import hashlib
from unittest.mock import patch

import numpy as np
import pytest

from comparison import evaluate, extract


def test_audio_failure_and_empty_clip_are_not_dropped():
    with patch.object(extract, "frame_features", side_effect=OSError("broken audio")), pytest.raises(OSError):
        extract._one(("broken.wav", "mcu"))
    with patch.object(extract, "frame_features", return_value=np.empty((0, 14))), pytest.raises(ValueError, match="No frames"):
        extract._one(("empty.wav", "mcu"))


def test_cache_rejects_missing_clips_and_object_arrays(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("id,split,group_id,label\na,train,g,positive\n")
    monkeypatch.setattr(evaluate, "MANIFEST", manifest)
    payload = {"features": np.zeros((1, 14)), "lengths": np.array([1]),
                   "label": np.array([True]), "kind": np.array(["positive"]),
                   "group": np.array(["g"]), "clip": np.array(["a"]),
                   "manifest_sha256": np.array(hashlib.sha256(manifest.read_bytes()).hexdigest())}
    path = tmp_path / "mcu-train.npz"
    np.savez(path, **payload)
    assert evaluate.load(tmp_path, "mcu", "train").lengths.tolist() == [1]
    payload["clip"] = np.array(["wrong"])
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="complete manifest"):
        evaluate.load(tmp_path, "mcu", "train")
    payload["clip"] = np.array(["a"])
    payload["features"] = np.array([{"untrusted": "object"}], dtype=object)
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="Object arrays"):
        evaluate.load(tmp_path, "mcu", "train")
    payload.pop("manifest_sha256")
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="manifest missing"):
        evaluate.load(tmp_path, "mcu", "train")


def test_manifest_rejects_group_leakage(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("id,split,group_id,label\na,train,g,positive\nb,test,g,negative\n")
    with pytest.raises(ValueError, match="group leakage"):
        extract.read_manifest(manifest)
