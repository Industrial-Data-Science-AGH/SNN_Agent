"""P1 acceptance: an incomplete model, a mismatched encoder_hash and a bad port
all end in a controlled error before Start.

Each test names the code it expects, so a reworded message does not break the
suite and a silently changed rejection reason does.
"""

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
    runtime = LuiRuntime()
    runtime.load(lui8)
    with pytest.raises(RuntimeStateError) as err:
        runtime.step({})
    assert err.value.code == "NOT_STARTED"


def test_reset_validates_the_session_clock(lui8):
    runtime = LuiRuntime()
    runtime.load(lui8)
    with pytest.raises(RuntimeStateError) as err:
        runtime.reset(epoch=0, source_time_us=0)
    assert err.value.code == "INVALID_EPOCH"
    runtime.reset(epoch=1, source_time_us=0)
    assert runtime.started


def test_a_rejected_package_does_not_replace_a_good_one(lui8, mutate):
    runtime = LuiRuntime()
    runtime.load(lui8)
    good = runtime.model.model_hash
    with pytest.raises(RuntimeLoadError):
        runtime.load(mutate(lui8, lambda m: m.pop("topology")))
    assert runtime.model.model_hash == good


def test_model_hash_is_the_canonical_hash_of_the_manifest(lui8):
    assert load_manifest(lui8).model_hash == content_hash(lui8)
