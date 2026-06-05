"""Notifications: email + clip save (P3).

Top-level imports remain hardware-free; smtplib, email, and cv2 are imported
lazily inside notify().
"""

import logging
from pathlib import Path

import numpy as np

from agent import camera, config

logger = logging.getLogger(__name__)


def _encode_jpeg(snapshot: np.ndarray) -> bytes:
    """Encode an RGB snapshot to JPEG bytes (flips to BGR for cv2)."""
    import cv2  # type: ignore[import-untyped]

    ok, buf = cv2.imencode(".jpg", snapshot[..., ::-1])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def notify(reason: str, snapshot: np.ndarray) -> None:
    """Send a Gmail SMTP_SSL alert with the snapshot attached.

    Args:
        reason: Human-readable reason for alarm.
        snapshot: RGB image array for attachment.

    Raises:
        smtplib.SMTPException: On send failure (caller owns retry/queue).
        OSError: On network/connection failure (caller owns retry/queue).
    """
    import smtplib
    from email.message import EmailMessage

    s = config.load_settings()
    msg = EmailMessage()
    msg["From"] = s.gmail_user
    msg["To"] = s.alert_to
    msg["Subject"] = f"{config.ALERT_SUBJECT_PREFIX} Intrusion detected"
    msg.set_content(reason)
    msg.add_attachment(
        _encode_jpeg(snapshot),
        maintype="image",
        subtype="jpeg",
        filename="snapshot.jpg",
    )
    try:
        with smtplib.SMTP_SSL(
            config.GMAIL_SMTP_HOST, config.GMAIL_SMTP_PORT, timeout=15
        ) as srv:
            srv.login(s.gmail_user, s.gmail_app_password)
            srv.send_message(msg)
    except (smtplib.SMTPException, OSError) as exc:
        logger.error("Email send failed: %s", exc)
        raise


def save_clip(frames: np.ndarray) -> Path:
    """Save alarm clip to var/clips/ by delegating to camera.save_clip.

    Args:
        frames: Array of shape (n_frames, height, width, 3) uint8.

    Returns:
        Path to the written clip file.
    """
    return camera.save_clip(frames)
