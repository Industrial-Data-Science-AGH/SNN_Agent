"""P1 acceptance: an incomplete model, a mismatched encoder_hash and a bad port
all end in a controlled error before Start.

Each test names the code it expects, so a reworded message does not break the
suite and a silently changed rejection reason does.
"""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from contracts.validation import content_hash, fixture
from snn_runtime import LuiRuntime, RuntimeLoadError, RuntimeStateError, load_manifest

# ----------------------------------------------------------------- accepted

def test_demo_fixture_is_accepted():
    """W0's scripted fixture stays loadable: no integrator, so no physics checks."""
    model = load_manifest(fixture("model-manifest"))
    assert model.integrator == "none"
    assert model.is_scripted
    assert model.score_kind == "unavailable"


def test_real_network_round_trips_through_the_contract(lui8):
    model = load_manifest(lui8)
    assert model.decision_neuron == "D"
    assert model.neuron_order == ("H0", "H1", "H2", "H3", "G0", "G1", "G2", "D")
    assert all(len(b) == 3 for b in model.bindings.values()), "every board uses J1..J3"
    assert model.unused_channels == (), "all 7 encoder features are wired"
    assert model.dt_us == 10_000


def test_uncalibrated_until_a_board_is_measured(lui8):
    """Potentials are fractions of V_th, not volts, so scores say so."""
    model = load_manifest(lui8)
    assert model.potential_unit == "a.u."
    assert model.calibration == "uncalibrated"
    assert model.score_kind == "uncalibrated"


def test_sign_is_reapplied_for_the_integrator(lui8):
    """The contract stores magnitude only; the simulator wants one signed number."""
    model = load_manifest(lui8)
    by_port = {b.target_port: b for b in model.bindings["D"]}
    assert by_port[1].weight > 0 and by_port[1].sign == "inhibitory"
    assert by_port[1].signed_weight < 0
    assert by_port[2].signed_weight > 0


# ------------------------------------------------- incomplete or inconsistent

def test_missing_required_block_is_rejected(lui8, mutate):
    broken = mutate(lui8, lambda m: m.pop("decoder"))
    with pytest.raises(RuntimeLoadError) as err:
        load_manifest(broken)
    assert err.value.code == "INVALID_SCHEMA"


def test_encoder_hash_mismatch_is_rejected(lui8):
    broken = dict(lui8)
    broken["encoder_hash"] = "sha256:" + "0" * 64
    with pytest.raises(RuntimeLoadError) as err:
        load_manifest(broken)
    assert err.value.code == "INVALID_CONTRACT"


def test_channel_renamed_under_a_stale_hash_is_rejected(lui8):
    """The classic silent break: profile edited, hash left alone."""
    broken = {**lui8, "encoder_profile": {**lui8["encoder_profile"]}}
    broken["encoder_profile"]["channel_map"] = [
        {**c, "channel": "hf_lo_v2"} if c["channel"] == "hf_lo" else c
        for c in lui8["encoder_profile"]["channel_map"]
    ]
    with pytest.raises(RuntimeLoadError):
        load_manifest(broken)


def test_empty_topology_is_not_a_classifier(lui8, mutate):
    broken = mutate(
        lui8,
        lambda m: m["topology"].update({"neurons": [], "connections": []}),
    )
    with pytest.raises(RuntimeLoadError) as err:
        load_manifest(broken)
    assert err.value.code == "EMPTY_TOPOLOGY"


# --------------------------------------------------------------------- ports

def test_port_outside_the_board_is_rejected(lui8, mutate):
    broken = mutate(lui8, lambda m: m["topology"]["connections"][0].update({"target_port": 4}))
    with pytest.raises(RuntimeLoadError) as err:
        load_manifest(broken)
    assert err.value.code == "INVALID_SCHEMA"


def test_two_wires_into_one_port_are_rejected(lui8, mutate):
    def collide(m):
        first = m["topology"]["connections"][0]
        for edge in m["topology"]["connections"][1:]:
            if edge["target_id"] == first["target_id"]:
                edge["target_port"] = first["target_port"]
                return
        raise AssertionError("fixture has no second wire into the same board")

    broken = mutate(lui8, collide)
    with pytest.raises(RuntimeLoadError) as err:
        load_manifest(broken)
    assert err.value.code == "INVALID_CONTRACT"


# --------------------------------------------------------------------- clock

def test_dt_disagreeing_with_the_encoder_is_rejected(lui8, mutate):
    """dt_us and hop/sample_rate are two independent fields saying the same thing."""
    broken = mutate(lui8, lambda m: m["runtime"].update({"dt_us": 20_000}))
    with pytest.raises(RuntimeLoadError) as err:
        load_manifest(broken)
    assert err.value.code == "DT_ENCODER_MISMATCH"


def test_the_known_0_16_percent_drift_is_tolerated(lui8):
    """dt_us = 10000 vs a 9984 us frame: real, documented, below tolerance."""
    model = load_manifest(lui8)
    assert model.dt_us == 10_000


def test_step_larger_than_a_time_constant_is_rejected(lui8, mutate):
    broken = mutate(lui8, lambda m: m["topology"]["neurons"][0].update({"tau_syn_us": 5_000}))
    with pytest.raises(RuntimeLoadError) as err:
        load_manifest(broken)
    assert err.value.code == "DT_NOT_STABLE"


def test_unknown_integrator_is_rejected(lui8, mutate):
    broken = mutate(lui8, lambda m: m["runtime"].update({"integrator": "euler-maruyama"}))
    with pytest.raises(RuntimeLoadError) as err:
        load_manifest(broken)
    assert err.value.code == "UNKNOWN_INTEGRATOR"


# ------------------------------------------------------------------- decoder

def test_window_off_the_step_grid_is_rejected(lui8, mutate):
    broken = mutate(lui8, lambda m: m["decoder"].update({"window_us": 15_000}))
    with pytest.raises(RuntimeLoadError) as err:
        load_manifest(broken)
    assert err.value.code == "DECODER_WINDOW"


def test_cooldown_shorter_than_the_window_is_rejected(lui8, mutate):
    broken = mutate(
        lui8,
        lambda m: m["decoder"].update({"window_us": 100_000, "cooldown_us": 50_000}),
    )
    with pytest.raises(RuntimeLoadError) as err:
        load_manifest(broken)
    assert err.value.code == "DECODER_COOLDOWN"


def test_zero_threshold_would_alarm_on_silence(lui8, mutate):
    broken = mutate(lui8, lambda m: m["decoder"].update({"threshold": 0}))
    with pytest.raises(RuntimeLoadError) as err:
        load_manifest(broken)
    assert err.value.code == "DECODER_THRESHOLD"


# ----------------------------------------------------------- decision neuron

def test_two_terminal_neurons_are_ambiguous(lui8, mutate):
    """The contract has no decision_neuron field; the runtime must not guess."""
    broken = mutate(
        lui8,
        lambda m: m["topology"]["connections"].remove(
            next(e for e in m["topology"]["connections"] if e["source_id"] == "G2")
        ),
    )
    with pytest.raises(RuntimeLoadError) as err:
        load_manifest(broken)
    assert err.value.code == "AMBIGUOUS_DECISION_NEURON"


# --------------------------------------------------------------- calibration

def test_volts_without_a_calibration_id_are_rejected(lui8, mutate):
    def claim_volts(m):
        m["runtime"]["potential_unit"] = "V"
        for neuron in m["topology"]["neurons"]:
            neuron["potential_unit"] = "V"

    broken = mutate(lui8, claim_volts)
    with pytest.raises(RuntimeLoadError) as err:
        load_manifest(broken)
    assert err.value.code == "CALIBRATION_MISSING"


# ----------------------------------------------------------------- artifacts

def test_artifact_must_exist_on_disk(lui8, tmp_path):
    with pytest.raises(RuntimeLoadError) as err:
        load_manifest(lui8, artifact_root=tmp_path)
    assert err.value.code == "ARTIFACT_MISSING"


def test_artifact_must_hash_to_what_was_declared(lui8, tmp_path):
    (tmp_path / lui8["artifacts"][0]["path"]).write_bytes(b"not the checkpoint")
    with pytest.raises(RuntimeLoadError) as err:
        load_manifest(lui8, artifact_root=tmp_path)
    assert err.value.code == "ARTIFACT_HASH_MISMATCH"


# ------------------------------------------------------------------ champion

def test_our_best_model_cannot_yet_claim_champion(lui8, mutate):
    """Missing training_run_id and evaluation_hash, so the contract refuses."""
    broken = mutate(lui8, lambda m: m.update({"status": "evaluated_champion"}))
    with pytest.raises(RuntimeLoadError) as err:
        load_manifest(broken)
    assert err.value.code == "INVALID_CONTRACT"


# ------------------------------------------------------------- runtime state

def test_calls_before_load_are_refused():
    runtime = LuiRuntime()
    with pytest.raises(RuntimeStateError) as err:
        runtime.step({})
    assert err.value.code == "NOT_LOADED"


def test_step_before_reset_is_refused(lui8):
    runtime = LuiRuntime(allow_unverified_artifacts=True)
    runtime.load(lui8)
    with pytest.raises(RuntimeStateError) as err:
        runtime.step({})
    assert err.value.code == "NOT_STARTED"


def test_reset_validates_the_session_clock(lui8):
    runtime = LuiRuntime(allow_unverified_artifacts=True)
    runtime.load(lui8)
    with pytest.raises(RuntimeStateError) as err:
        runtime.reset(epoch=0, source_time_us=0)
    assert err.value.code == "INVALID_EPOCH"
    runtime.reset(epoch=1, source_time_us=0)
    assert runtime.started


def test_a_rejected_package_does_not_replace_a_good_one(lui8, mutate):
    runtime = LuiRuntime(allow_unverified_artifacts=True)
    runtime.load(lui8)
    good = runtime.model.model_hash
    with pytest.raises(RuntimeLoadError):
        runtime.load(mutate(lui8, lambda m: m.pop("topology")))
    assert runtime.model.model_hash == good


def test_model_hash_is_the_canonical_hash_of_the_manifest(lui8):
    assert load_manifest(lui8).model_hash == content_hash(lui8)


# ------------------------------------------------- isolation and memory

def test_mutating_the_callers_manifest_after_load_changes_nothing_that_was_accepted(lui8):
    manifest = copy.deepcopy(lui8)
    model = load_manifest(manifest)
    accepted_hash, accepted_threshold = model.model_hash, model.decoder["threshold"]
    manifest["decoder"]["threshold"] = 0
    manifest["topology"]["neurons"].clear()
    assert model.decoder["threshold"] == accepted_threshold and len(model.manifest["topology"]["neurons"]) > 0
    assert content_hash(json.loads(json.dumps(model.manifest, default=dict))) == accepted_hash


def test_the_accepted_package_cannot_be_edited_through_the_result(lui8):
    model = load_manifest(lui8)
    with pytest.raises(TypeError):
        model.decoder["threshold"] = 0
    with pytest.raises(TypeError):
        model.manifest["decoder"]["window_us"] = 1
    with pytest.raises(TypeError):
        model.manifest["topology"]["neurons"][0]["v_threshold"] = 9  # a nested mapping inside a tuple
    with pytest.raises((TypeError, AttributeError)):
        model.manifest["topology"]["neurons"].append({})  # lists were frozen to tuples
    with pytest.raises(TypeError):
        model.channel_index["peak"] = 99
    with pytest.raises(TypeError):
        model.bindings["extra"] = ()
    assert model.manifest["decoder"] == lui8["decoder"]  # still equal in content to what was given


def test_load_does_not_touch_the_callers_dict(lui8):
    before = copy.deepcopy(lui8)
    load_manifest(lui8)
    assert lui8 == before


def test_artifacts_are_hashed_from_a_stream_not_read_whole(lui8, tmp_path, monkeypatch):
    payload = b"weights" * 600_000  # about 4 MiB
    (tmp_path / lui8["artifacts"][0]["path"]).write_bytes(payload)
    declared = copy.deepcopy(lui8)
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    declared["artifacts"][0]["sha256"] = declared["provenance"]["checkpoint_hash"] = digest  # the contract ties the two

    def refuse(self):
        raise AssertionError("the whole artifact was read into memory")

    monkeypatch.setattr(Path, "read_bytes", refuse)
    assert load_manifest(declared, artifact_root=tmp_path).model_id == lui8["model_id"]


# --------------------------------------------- a running runtime verifies weights

def declared_file(lui8, tmp_path, payload=b"the trained weights"):
    (tmp_path / lui8["artifacts"][0]["path"]).write_bytes(payload)
    declared = copy.deepcopy(lui8)
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    declared["artifacts"][0]["sha256"] = declared["provenance"]["checkpoint_hash"] = digest
    return declared


def test_a_runtime_without_an_artifact_root_refuses_a_package_it_cannot_verify(lui8):
    runtime = LuiRuntime()
    with pytest.raises(RuntimeLoadError) as err:
        runtime.load(lui8)
    assert err.value.code == "ARTIFACT_ROOT_REQUIRED"
    with pytest.raises(RuntimeStateError) as state:
        runtime.reset(epoch=1, source_time_us=0)
    assert state.value.code == "NOT_LOADED"  # nothing half-started


def test_a_runtime_with_a_root_verifies_then_accepts(lui8, tmp_path):
    runtime = LuiRuntime(artifact_root=str(tmp_path))
    with pytest.raises(RuntimeLoadError) as err:
        runtime.load(lui8)
    assert err.value.code == "ARTIFACT_MISSING"
    runtime.load(declared_file(lui8, tmp_path))
    runtime.reset(epoch=1, source_time_us=0)
    assert runtime.started


def test_a_wrong_weights_file_is_refused_by_a_runtime_with_a_root(lui8, tmp_path):
    declared = declared_file(lui8, tmp_path)
    (tmp_path / declared["artifacts"][0]["path"]).write_bytes(b"swapped after packaging")
    with pytest.raises(RuntimeLoadError) as err:
        LuiRuntime(artifact_root=str(tmp_path)).load(declared)
    assert err.value.code == "ARTIFACT_HASH_MISMATCH"


def test_the_standalone_loader_still_inspects_a_manifest_without_weights(lui8):
    assert load_manifest(lui8).model_id == lui8["model_id"]  # dashboards and contract tests hold no weights
    with pytest.raises(RuntimeLoadError) as err:
        load_manifest(lui8, require_artifacts=True)
    assert err.value.code == "ARTIFACT_ROOT_REQUIRED"


def test_a_package_without_artifacts_needs_no_root(lui8):
    bare = copy.deepcopy(lui8)
    bare["artifacts"] = []
    bare["provenance"]["checkpoint_hash"] = None
    try:
        load_manifest(bare, require_artifacts=True)
    except RuntimeLoadError as err:
        assert err.code != "ARTIFACT_ROOT_REQUIRED"  # whatever else the contract says, not the missing root


def test_the_environment_factory_wires_the_root_and_the_opt_out(lui8, tmp_path):
    from snn_runtime.runtime import from_environment

    with pytest.raises(RuntimeLoadError) as err:
        from_environment({}).load(lui8)
    assert err.value.code == "ARTIFACT_ROOT_REQUIRED"
    from_environment({"SNN_MODEL_ARTIFACT_ROOT": str(tmp_path)}).load(declared_file(lui8, tmp_path))
    from_environment({"SNN_ALLOW_UNVERIFIED_ARTIFACTS": "1"}).load(lui8)
    with pytest.raises(RuntimeLoadError):
        from_environment({"SNN_ALLOW_UNVERIFIED_ARTIFACTS": "true"}).load(lui8)  # only the literal 1 opts out


# ------------------------------------------------------- call order

@pytest.mark.parametrize("call", ["step", "snapshot", "checkpoint", "restore"])
def test_every_stateful_call_checks_the_session_order_before_it_does_anything(lui8, call):
    args = {"step": ({},), "restore": (b"",)}.get(call, ())
    runtime = LuiRuntime(allow_unverified_artifacts=True)
    with pytest.raises(RuntimeStateError) as err:
        getattr(runtime, call)(*args)
    assert err.value.code == "NOT_LOADED"
    runtime.load(lui8)
    with pytest.raises(RuntimeStateError) as err:
        getattr(runtime, call)(*args)
    assert err.value.code == "NOT_STARTED"  # restore included: it cannot bypass reset()
    runtime.reset(epoch=1, source_time_us=0)
    with pytest.raises(NotImplementedError):
        getattr(runtime, call)(*args)  # only now does it reach the task that delivers it
