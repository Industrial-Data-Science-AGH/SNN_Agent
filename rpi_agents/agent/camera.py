"""Camera capture via picamera2 (P2).

Top-level imports remain hardware-free; picamera2 and cv2 are imported lazily
inside functions.
"""

import time
from pathlib import Path

import numpy as np

from agent import config


def capture(n_frames: int = config.CAPTURE_FRAMES_N) -> np.ndarray:
    """Capture frames from the CSI camera via picamera2.

    picamera2 delivers RGB arrays; this function flips each frame to BGR so
    all downstream cv2 operations have a consistent channel order.

    Args:
        n_frames: Number of frames to capture.

    Returns:
        Array of shape (n_frames, H, W, 3) in BGR uint8.
    """
    from picamera2 import Picamera2  # type: ignore[import-untyped]

    cam = Picamera2()
    try:
        cam.configure(cam.create_still_configuration())
        cam.start()
        frames = []
        for _ in range(n_frames):
            frame = cam.capture_array()          # RGB uint8 (H, W, 3)
            frames.append(frame[..., ::-1])      # RGB → BGR
        return np.stack(frames).astype(np.uint8)
    finally:
        cam.close()


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
