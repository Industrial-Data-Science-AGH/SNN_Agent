"""Configuration and settings for the RPi 5 agent.

Module constants (GPIO pins, thresholds, paths, timeouts) live at module level.
Secrets are loaded lazily via load_settings() from environment variables.

All paths are configurable via environment; defaults use XDG conventions.
Hardware constants documented as "bring-up TBD" are noted.
"""

import os
from dataclasses import dataclass
from pathlib import Path


LED_PIN: int = 17
"""GPIO pin for red LED (active high). Bring-up TBD."""

BUZZER_PIN: int = 27
"""GPIO pin for piezo buzzer (active high). Bring-up TBD."""

WAKE_CONFIRM_PIN: int = 3
"""GPIO pin receiving SNN trigger (active low; must level-shift 5V→3.3V).
Bring-up TBD."""


PREFILTER_STATIC_FRAMES_N: int = 3
"""Number of consecutive static frames to confirm no motion."""

PREFILTER_MOTION_THRESHOLD: float = 0.05
"""Optical flow magnitude threshold [0.0, 1.0] to detect motion."""


VAR_DIR: Path = Path(os.getenv(
    "SNN_AGENT_VAR_DIR",
    str(Path.home() / ".local" / "var" / "snn-agent")
))
"""Directory for logs, clips, event records."""

CLIPS_DIR: Path = VAR_DIR / "clips"
"""Subdirectory for saved alarm video clips (git-ignored)."""

EVENT_LOG: Path = VAR_DIR / "event.log"
"""Event journal (tap, spike, wake, alarm, notification)."""

PREFILTER_PERSON_CONF: float = 0.45
"""Minimum MobileNet-SSD confidence to count a 'person' detection."""

PREFILTER_PERSON_ENABLED: bool = True
"""If False, use motion-only gate (no person DNN). Still permissive."""

PERSON_MODEL_DIR: Path = Path(os.getenv("SNN_AGENT_MODEL_DIR", str(VAR_DIR / "models")))
"""Directory containing MobileNetSSD_deploy.prototxt + .caffemodel (git-ignored)."""


GEMINI_MODEL: str = "gemini-2.0-flash"
"""Gemini model name for vision classification."""

GEMINI_TIMEOUT_S: float = 10.0
"""Timeout for Gemini API calls (seconds). On timeout, default to ALARM."""


CAPTURE_FRAMES_N: int = 10
"""Number of frames to capture for prefilter + snapshot for vision."""

COOLDOWN_S: float = 5.0
"""Seconds to wait before re-armed after wake (debounce SNN chatter)."""


POWER_MODE: str = os.getenv("SNN_AGENT_POWER_MODE", "warm")
"""Power state strategy: 'warm' (stay on, dev mode) or 'halt' (sleep → wake).
Set via SNN_AGENT_POWER_MODE env var."""


RECALL_MIN: float = 0.98
"""Minimum recall (TP / (TP+FN)) on eval set."""

FALSE_ALARM_MAX: float = 0.20
"""Maximum false-alarm rate (FP / (FP+TN)) on eval set."""

VISION_CALLS_MAX_PER_NONINTRUSION: float = 0.30
"""Maximum vision API calls per non-intrusion event."""


# SECRETS LOADER (lazy, no module-level import of python-dotenv)

@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment or .env file.

    Never log or print these values; they contain API keys and credentials.
    """
    gemini_api_key: str
    gmail_user: str
    gmail_app_password: str
    alert_to: str


def load_settings() -> Settings:
    """Load secrets from environment or ~/.config/snn-agent/.env.

    Returns:
        Settings dataclass with API key, email creds, alert recipient.

    Raises:
        ValueError: If any required key is empty or missing.

    Notes:
        - On the Pi, ~/.config/snn-agent/.env is created at deploy time.
        - On dev machines, these env vars are optional (None → raises ValueError).
        - python-dotenv is imported lazily here; not required for module import.
    """
    # Try to load from .env file if it exists; don't fail if absent.
    env_file = Path.home() / ".config" / "snn-agent" / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv  # type: ignore[import-untyped]
            load_dotenv(env_file)
        except ImportError:
            # python-dotenv not installed; continue with os.environ.
            pass

    # All of these must be present and non-empty.
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    gmail_user = os.getenv("GMAIL_USER", "").strip()
    gmail_pw = os.getenv("GMAIL_APP_PASSWORD", "").strip()
    alert_to = os.getenv("ALERT_TO", "").strip()

    if not all([api_key, gmail_user, gmail_pw, alert_to]):
        raise ValueError(
            "Missing required secrets: GEMINI_API_KEY, GMAIL_USER, "
            "GMAIL_APP_PASSWORD, ALERT_TO. "
            "Set them in ~/.config/snn-agent/.env or as environment variables."
        )

    return Settings(
        gemini_api_key=api_key,
        gmail_user=gmail_user,
        gmail_app_password=gmail_pw,
        alert_to=alert_to,
    )
