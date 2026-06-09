"""Camera capture via picamera2 (CSI) or OpenCV VideoCapture (USB).

Top-level imports remain hardware-free; picamera2 and cv2 are imported lazily
inside functions.  The active backend is selected at call time via
config.CAMERA_BACKEND so tests can monkeypatch the mode.
"""

import time
from pathlib import Path

import numpy as np

from agent import config


def capture(n_frames: int = config.CAPTURE_FRAMES_N) -> np.ndarray:
    """Capture frames from the configured camera backend.

    Dispatches to _capture_csi or _capture_usb based on config.CAMERA_BACKEND
    (read at call time so tests can monkeypatch).

    Args:
        n_frames: Number of frames to capture.

    Returns:
        Array of shape (n_frames, H, W, 3) in BGR uint8.

    Raises:
        ValueError: If config.CAMERA_BACKEND is not 'csi' or 'usb'.
    """
    backend = config.CAMERA_BACKEND
    if backend == "csi":
        return _capture_csi(n_frames)
    if backend == "usb":
        return _capture_usb(n_frames)
    raise ValueError(
        f"Unknown CAMERA_BACKEND: {backend!r} (expected 'csi' or 'usb')"
    )


def _capture_csi(n_frames: int) -> np.ndarray:
    """Capture frames from the CSI camera via picamera2.

    picamera2 delivers RGB arrays; this function flips each frame to BGR so
    all downstream cv2 operations have a consistent channel order.
    """
    from picamera2 import Picamera2  # type: ignore[import-untyped]

    cam = Picamera2()
    try:
        cam.configure(cam.create_still_configuration())
        cam.start()
        frames = []
        for i in range(n_frames):
            if i > 0 and config.CAPTURE_INTERVAL_S > 0:
                time.sleep(config.CAPTURE_INTERVAL_S)
            frame = cam.capture_array()          # RGB uint8 (H, W, 3)
            frames.append(frame[..., ::-1])      # RGB → BGR
        return np.stack(frames).astype(np.uint8)
    finally:
        cam.close()


def _capture_usb(n_frames: int) -> np.ndarray:
    """Capture frames from a USB webcam via OpenCV VideoCapture.

    cv2.VideoCapture returns BGR natively — no channel flip needed.
    Warmup frames are discarded to skip dark/garbage frames on open.
    """
    import cv2  # type: ignore[import-untyped]  # lazy: not required for CSI path

    cap = cv2.VideoCapture(config.CAMERA_USB_INDEX)
    try:
        if not cap.isOpened():
            raise RuntimeError(
                f"USB camera index {config.CAMERA_USB_INDEX} could not be opened"
                " (check /dev/video*, 'video' group)"
            )
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # best-effort; V4L2 may ignore
        for _ in range(config.CAMERA_USB_WARMUP_FRAMES):
            cap.read()
        frames = []
        for i in range(n_frames):
            if i > 0 and config.CAPTURE_INTERVAL_S > 0:
                time.sleep(config.CAPTURE_INTERVAL_S)
                for _ in range(config.CAMERA_USB_DRAIN_FRAMES):
                    cap.grab()
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("USB camera read failed")
            frames.append(frame)
        return np.stack(frames).astype(np.uint8)
    finally:
        cap.release()


def save_clip(frames: np.ndarray, path: Path | None = None) -> Path:
    """Write a frame stack to an mp4 clip file.

    Args:
        frames: Array of shape (n_frames, H, W, 3) in BGR uint8.
        path: Destination file path. Defaults to a timestamped file under
            config.CLIPS_DIR.

    Returns:
        Path to the written clip file.
    """
    import cv2  # type: ignore[import-untyped]

    if path is None:
        path = config.CLIPS_DIR / f"clip_{int(time.time())}.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)

    H, W = frames.shape[1], frames.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (W, H))
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()

    return path
