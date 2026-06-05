"""Notifications: email + clip save (P4).

Stubs for hardware integration. Top-level imports remain hardware-free;
smtplib and email libs are imported lazily inside functions.
"""

import numpy as np


def notify(reason: str, snapshot: np.ndarray) -> None:
    """Send email alert with snapshot attachment.

    Args:
        reason: Human-readable reason for alarm (Decision.reason).
        snapshot: RGB image array for attachment.

    Raises:
        NotImplementedError: Phase P4 implementation pending.
    """
    del reason, snapshot
    raise NotImplementedError("P4: notifier.notify()")


def save_clip(frames: np.ndarray) -> None:
    """Save alarm clip to var/clips/ (git-ignored directory).

    Args:
        frames: Array of shape (n_frames, height, width, 3) uint8.

    Raises:
        NotImplementedError: Phase P4 implementation pending.
    """
    del frames
    raise NotImplementedError("P4: notifier.save_clip()")
