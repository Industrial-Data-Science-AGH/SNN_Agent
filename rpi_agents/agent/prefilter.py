"""Local motion + person prefilter: gates vision API calls (P2).

Stubs for hardware integration. Top-level imports remain hardware-free;
cv2 and sklearn are imported lazily inside functions.
"""

import numpy as np

from agent.types import PrefilterResult


def run(frames: np.ndarray) -> PrefilterResult:
    """Run motion + person detection on frame sequence.

    Cheap local processing (optical flow, shape detection) to gate Gemini calls.
    Caches bounding box / shape on escalate=True for vision.

    Args:
        frames: Array of shape (n_frames, height, width, 3) in BGR uint8.

    Returns:
        PrefilterResult with motion, person, escalate, and confidence score.

    Raises:
        NotImplementedError: Phase P2 implementation pending.
    """
    del frames
    raise NotImplementedError("P2: prefilter.run()")
