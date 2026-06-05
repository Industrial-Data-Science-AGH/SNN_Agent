"""Camera capture via picamera2 (P2).

Stubs for hardware integration. Top-level imports remain hardware-free;
picamera2 is imported lazily inside functions.
"""

import numpy as np


def capture(n_frames: int) -> np.ndarray:
    """Capture frames from CSI camera via picamera2.

    Args:
        n_frames: Number of frames to capture.

    Returns:
        Array of shape (n_frames, height, width, 3) in BGR uint8.

    Raises:
        NotImplementedError: Phase P2 implementation pending.
    """
    del n_frames
    raise NotImplementedError("P2: camera.capture()")
