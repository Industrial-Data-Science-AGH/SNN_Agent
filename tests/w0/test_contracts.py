import copy
import json

import pytest
from jsonschema import Draft202012Validator

from contracts.validation import ROOT, ContractError, content_hash, fixture, schema, validate


def test_all_schemas_and_fixtures():
    for path in (ROOT / "v1").glob("*.schema.json"):
        Draft202012Validator.check_schema(json.loads(path.read_text()))
    for name, contract in json.loads((ROOT / "fixtures/index.json").read_text()).items():
        validate(contract, fixture(name), manifest=fixture("model-manifest"))


@pytest.mark.parametrize(
    "change,code",
    [
        ({"schema_version": "2.0"}, "VERSION_MISMATCH"),
        ({"encoder_hash": "sha256:" + "0" * 64}, "ENCODER_MISMATCH"),
        ({"spikes": [{"dt_us": 0, "channel": "autocorr_lag1"}]}, "UNKNOWN_CHANNEL"),
        ({"spikes": [{"dt_us": 250000, "channel": "zcr"}]}, "INVALID_CONTRACT"),
        ({"spikes": [{"dt_us": 2, "channel": "zcr"}, {"dt_us": 1, "channel": "zcr"}]}, "INVALID_CONTRACT"),
        ({"source_end_us": 250000}, "INVALID_CONTRACT"),
        ({"batch_seq": -1}, "INVALID_SCHEMA"),
        ({"unexpected": "value"}, "INVALID_SCHEMA"),
    ],
)
def test_reject_invalid_batch(change, code):
    value = fixture("spike") | change
    with pytest.raises(ContractError) as error:
        validate("SpikeBatch", value, manifest=fixture("model-manifest"))
    assert error.value.code == code


def test_hash_is_cross_language_canonical():
    assert content_hash({"b": 1.0, "a": 2}) == content_hash({"a": 2, "b": 1})
    for bad in [float("nan"), float("inf"), 2**54]:
        with pytest.raises(ContractError):
            content_hash({"bad": bad})


def test_encoder_semantics_and_hash():
    m = fixture("model-manifest")
    m["encoder_profile"]["channel_map"][1]["feature"] = "hjorth_mobility"
    with pytest.raises(ContractError, match="hash mismatch"):
        validate("ModelManifest", m)
    profile = fixture("encoder-profile")
    profile["channel_map"][1]["index"] = 0
    with pytest.raises(ContractError, match="indices"):
        validate("EncoderProfile", profile)


@pytest.mark.parametrize("count", [0, 1, 8, 50])
def test_topology_sizes(count):
    m = fixture("model-manifest")
    node = copy.deepcopy(m["topology"]["neurons"][0])
    m["topology"]["neurons"] = [node | {"neuron_id": f"n{i}"} for i in range(count)]
    m["topology"]["connections"] = []
    validate("ModelManifest", m)


def test_manifest_rejects_51_duplicate_port_and_dangling_edge():
    for bad in ["too_many", "duplicate_port", "dangling"]:
        m = fixture("model-manifest")
        if bad == "too_many":
            n = m["topology"]["neurons"][0]
            m["topology"]["neurons"] = [n | {"neuron_id": f"n{i}"} for i in range(51)]
        elif bad == "duplicate_port":
            m["topology"]["connections"] *= 2
        else:
            m["topology"]["connections"][0]["target_id"] = "missing"
        with pytest.raises(ContractError):
            validate("ModelManifest", m)


def test_unavailable_vision_never_confirms_glass():
    v = fixture("vision-unavailable")
    v["glass_visible"] = True
    with pytest.raises(ContractError, match="unknown"):
        validate("VisionResult", v)


@pytest.mark.parametrize("expires", ["not-a-date", "2026-09-24T00:00:40Z", "2026-09-23T00:00:00Z"])
def test_command_expiry(expires):
    c = fixture("capture-command") | {"expires_at": expires}
    with pytest.raises(ContractError):
        validate("CaptureCommand", c)


def test_evaluated_champion_cannot_be_demo():
    m = fixture("model-manifest") | {"status": "evaluated_champion"}
    with pytest.raises(ContractError):
        validate("ModelManifest", m)


def test_schema_version_and_no_remote_refs():
    for p in (ROOT / "v1").glob("*.schema.json"):
        s = schema(p.stem.removesuffix(".schema"))
        assert s["properties"]["schema_version"] == {"const": "1.0"}
        assert "$ref" not in json.dumps(s)  # fully offline, portable schema bundle


def test_demo_artifacts_match_manifest_bytes():
    import hashlib

    manifest = fixture("model-manifest")
    for a in manifest["artifacts"]:
        raw = (ROOT / "fixtures" / a["path"]).read_bytes()
        assert "sha256:" + hashlib.sha256(raw).hexdigest() == a["sha256"]
    assert fixture("session-create")["model_hash"] == content_hash(manifest)
    assert fixture("session-create")["encoder_hash"] == content_hash(fixture("encoder-profile"))
