"""Vision classification: Gemini API + failsafe (P3).

Stubs for hardware integration. Top-level imports remain hardware-free;
google.genai is imported lazily inside functions.
"""

import numpy as np

from agent.types import VisionVerdict


def verdict(snapshot: np.ndarray) -> VisionVerdict:
    """Classify snapshot as intrusion or false alarm via Gemini.

    Sends single RGB image to Gemini 2.0 Flash with prompt.
    On timeout or error, returns failsafe ALARM verdict.

    Args:
        snapshot: RGB image array, shape (height, width, 3) uint8.

    Returns:
        VisionVerdict with is_intrusion, confidence, reason, and source.

    Raises:
        NotImplementedError: Phase P3 implementation pending.
    """
    del snapshot
    raise NotImplementedError("P3: vision.verdict()")
