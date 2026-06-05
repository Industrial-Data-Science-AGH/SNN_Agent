"""Unit tests for agent.camera: shape/dtype, BGR conversion, always-closed, clip write.

picamera2 is never imported at module top — it is monkey-patched inside tests.
opencv-python-headless is available on Mac so the save_clip cv2 paths run for real.
"""

import sys
import types

import numpy as np
import pytest

from agent import camera, config

_H, _W, _N = 480, 640, 3  # representative resolution; kept small for fast CI


# Helpers / shared fake camera


class FakeCam:
    """Minimal Picamera2 stand-in that returns synthetic RGB frames."""

    def __init__(self, h: int = _H, w: int = _W, fill: int = 100) -> None:
        self._h = h
        self._w = w
        self._fill = fill
        self.configured = False
        self.started = False
        self.closed = False

    def create_still_configuration(self) -> dict:
        return {}

    def configure(self, _cfg: dict) -> None:
        self.configured = True

    def start(self) -> None:
        self.started = True

    def capture_array(self) -> np.ndarray:
        return np.full((self._h, self._w, 3), self._fill, dtype=np.uint8)

    def close(self) -> None:
        self.closed = True


def _inject_picamera2(monkeypatch: pytest.MonkeyPatch, cam: FakeCam) -> None:
    """Inject a fake picamera2 stub into sys.modules so the lazy import succeeds.

    camera.py does ``from picamera2 import Picamera2`` lazily inside capture().
    monkeypatch.setattr("picamera2.Picamera2", ...) requires picamera2 to be
    importable first.  We satisfy that by pre-seeding sys.modules with a stub
    module whose Picamera2 attribute is a factory that returns *cam*.
    """
    stub = types.ModuleType("picamera2")
    stub.Picamera2 = lambda: cam  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "picamera2", stub)


@pytest.fixture()
def fake_cam(monkeypatch: pytest.MonkeyPatch) -> FakeCam:
    """Inject a stub picamera2 module and return the FakeCam instance."""
    cam = FakeCam()
    _inject_picamera2(monkeypatch, cam)
    return cam


# capture() — shape, dtype, BGR conversion


def test_capture_returns_shape_and_dtype(fake_cam: FakeCam) -> None:
    frames = camera.capture(n_frames=_N)
    assert frames.shape == (_N, _H, _W, 3)
    assert frames.dtype == np.uint8


def test_capture_converts_rgb_to_bgr(monkeypatch: pytest.MonkeyPatch) -> None:
    """picamera2 delivers RGB; capture() must flip channel order to BGR."""
    # Asymmetric RGB: R=10, G=20, B=30.  After [..,::-1]: idx0=B=30, idx2=R=10.
    rgb = np.zeros((_H, _W, 3), dtype=np.uint8)
    rgb[:, :, 0] = 10  # R channel
    rgb[:, :, 1] = 20  # G channel
    rgb[:, :, 2] = 30  # B channel

    cam = FakeCam()
    cam.capture_array = lambda: rgb.copy()  # type: ignore[method-assign]
    _inject_picamera2(monkeypatch, cam)

    frames = camera.capture(n_frames=1)
    # BGR[0] should be the original B value (30); BGR[2] the original R value (10)
    assert int(frames[0, 0, 0, 0]) == 30
    assert int(frames[0, 0, 0, 2]) == 10


# capture() — camera always closed


def test_capture_always_closes_camera(fake_cam: FakeCam) -> None:
    camera.capture(n_frames=_N)
    assert fake_cam.closed is True


def test_capture_closes_camera_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Even when capture_array() raises, the camera handle must be released."""
    cam = FakeCam()

    def _boom() -> np.ndarray:
        raise RuntimeError("sensor error")

    cam.capture_array = _boom  # type: ignore[method-assign]
    _inject_picamera2(monkeypatch, cam)

    with pytest.raises(RuntimeError, match="sensor error"):
        camera.capture(n_frames=1)
    assert cam.closed is True


# save_clip() — file written and non-empty


def test_save_clip_writes_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory
) -> None:
    monkeypatch.setattr(config, "CLIPS_DIR", tmp_path)
    frames = np.zeros((_N, _H, _W, 3), dtype=np.uint8)
    out = camera.save_clip(frames)
    assert out.exists()
    assert out.stat().st_size > 0


def test_save_clip_accepts_explicit_path(tmp_path: pytest.TempPathFactory) -> None:
    from pathlib import Path

    dest = Path(str(tmp_path)) / "subdir" / "test_clip.mp4"
    frames = np.zeros((_N, _H, _W, 3), dtype=np.uint8)
    out = camera.save_clip(frames, path=dest)
    assert out == dest
    assert dest.exists()
