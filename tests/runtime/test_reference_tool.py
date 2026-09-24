"""The reference converter must not emit a manifest the runtime would refuse, or one whose declared checkpoint hash
is not the named file's. The export is rebuilt from the reference manifest, so no file outside the gate is needed."""

import hashlib
import json

import pytest

from snn_runtime import load_manifest
from snn_runtime.tools.make_reference_manifest import main

CHAMPION_ARGS = [
    "--status", "evaluated_champion", "--training-run-id", "run-1", "--evaluation-hash", "sha256:" + "e" * 64,
    "--dataset-manifest-hash", "sha256:" + "d" * 64, "--seed", "0",
]  # fmt: skip


@pytest.fixture
def export(lui8, tmp_path):
    """An hw_*.json-shaped export equivalent to the reference manifest (the inverse of the documented mapping)."""
    boards = {n["neuron_id"]: {"tau_mem_ms": n["tau_mem_us"] / 1000, "tau_syn_ms": n["tau_syn_us"] / 1000,
                               "v_leak": n["v_leak"], "synapses": []} for n in lui8["topology"]["neurons"]}  # fmt: skip
    for edge in lui8["topology"]["connections"]:
        inhibitory = edge["sign"] == "inhibitory"
        boards[edge["target_id"]]["synapses"].append({
            "from": edge["source_id"], "port": f"J{edge['target_port']}", "sign": "-" if inhibitory else "+",
            "w_sim": -edge["weight"] if inhibitory else edge["weight"],
        })  # fmt: skip
    data = {"dt_s": lui8["runtime"]["dt_us"] / 1_000_000, "v_th": 1.0, "boards": boards,
            "channels": [c["channel"] for c in lui8["encoder_profile"]["channel_map"]]}  # fmt: skip
    path = tmp_path / "hw_export.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def cli(lui8, export, tmp_path):
    profile = lui8["encoder_profile"]
    out = tmp_path / "out.json"

    def run(*extra):
        args = ["--export", str(export), "--out", str(out), "--model-id", "lui8-test", "--artifact-path", "w.pt",
                "--profile-id", profile["profile_id"], "--encoder-sha", profile["implementation_sha"],
                "--encoder-config-sha256", profile["config_sha256"], "--topology-version",
                lui8["topology"]["topology_version"], *extra]  # fmt: skip
        return main(args), out

    return run


def test_a_consistent_conversion_writes_a_manifest_the_runtime_accepts(cli):
    code, out = cli("--artifact-sha256", "sha256:" + "a" * 64)
    assert code == 0 and load_manifest(json.loads(out.read_text())).model_id == "lui8-test"


def test_a_named_checkpoint_is_hashed_from_the_file_and_declared_in_both_places(cli, tmp_path):
    weights = tmp_path / "v2.pt"
    weights.write_bytes(b"trained weights")
    code, out = cli("--checkpoint", str(weights))
    manifest = json.loads(out.read_text())
    digest = "sha256:" + hashlib.sha256(b"trained weights").hexdigest()
    assert code == 0 and manifest["artifacts"][0]["sha256"] == manifest["provenance"]["checkpoint_hash"] == digest


def test_a_large_checkpoint_is_hashed_from_a_stream(cli, tmp_path, monkeypatch):
    from pathlib import Path

    weights = tmp_path / "big.pt"
    weights.write_bytes(b"w" * 3_000_000)
    real = Path.read_bytes

    def guarded(self):
        assert self.name != "big.pt", "the checkpoint was read whole"  # the small export may still be read normally
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    code, out = cli("--checkpoint", str(weights))
    assert code == 0 and json.loads(out.read_text())["artifacts"][0]["sha256"] == "sha256:" + hashlib.sha256(b"w" * 3_000_000).hexdigest()


def test_a_checkpoint_and_an_explicit_digest_together_are_refused_even_if_they_agree(cli, tmp_path, capsys):
    weights = tmp_path / "v2.pt"
    weights.write_bytes(b"trained weights")
    digest = "sha256:" + hashlib.sha256(b"trained weights").hexdigest()
    for given in (digest, "sha256:" + "0" * 64):
        with pytest.raises(SystemExit) as exit_:
            cli("--checkpoint", str(weights), "--artifact-sha256", given)
        assert exit_.value.code == 2
    assert not (tmp_path / "out.json").exists() and "alternatives" in capsys.readouterr().err


def test_a_champion_without_its_provenance_is_refused_and_no_file_is_written(cli, capsys):
    code, out = cli("--artifact-sha256", "sha256:" + "a" * 64, "--status", "evaluated_champion")
    assert code == 1 and not out.exists()
    assert "INVALID_CONTRACT" in capsys.readouterr().err


def test_a_champion_with_all_of_its_provenance_is_written(cli):
    code, out = cli("--artifact-sha256", "sha256:" + "a" * 64, *CHAMPION_ARGS)
    assert code == 0 and json.loads(out.read_text())["status"] == "evaluated_champion"


def test_a_decoder_the_runtime_would_reject_is_not_written(cli, capsys):
    code, out = cli("--artifact-sha256", "sha256:" + "a" * 64, "--k", "0")
    assert code == 1 and not out.exists() and "DECODER_THRESHOLD" in capsys.readouterr().err
