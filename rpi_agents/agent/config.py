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

BUZZER_ENABLED: bool = os.getenv("SNN_AGENT_BUZZER_ENABLED", "true").strip().lower() != "false"
"""If False, buzzer hardware is skipped entirely (LED-only bring-up).
Set SNN_AGENT_BUZZER_ENABLED=false on the Pi while the piezo isn't wired yet."""

WAKE_CONFIRM_PIN: int = 3
"""GPIO pin receiving SNN trigger (active low; must level-shift 5V→3.3V).
Bring-up TBD."""


PREFILTER_STATIC_FRAMES_N: int = 3
"""Number of consecutive static frames to confirm no motion."""

PREFILTER_MOTION_THRESHOLD: float = float(os.getenv("SNN_AGENT_MOTION_THRESHOLD", "0.08"))
"""Optical flow magnitude threshold [0.0, 1.0] to detect motion.
Default 0.08 sits above the measured RealSense static-noise floor (~0.045).
Set via SNN_AGENT_MOTION_THRESHOLD env var."""


VAR_DIR: Path = Path(
    os.getenv("SNN_AGENT_VAR_DIR", str(Path.home() / ".local" / "var" / "snn-agent"))
)
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


GEMINI_MODEL: str = os.getenv("SNN_AGENT_GEMINI_MODEL", "gemini-3.1-flash-lite")
"""Gemini model name for vision classification.
Default is gemini-3.1-flash-lite (GA, vision-capable, free-tier ~1000 RPD).
gemini-2.0-flash free-tier quota is now 0 (deprecated).
Set via SNN_AGENT_GEMINI_MODEL env var."""

GEMINI_TIMEOUT_S: float = 10.0
"""Timeout for Gemini API calls (seconds). On timeout, default to ALARM."""

GEMINI_MAX_OUTPUT_TOKENS: int = 256
"""Maximum output tokens for Gemini API calls. Verdict JSON is tiny; cap cost."""

GMAIL_SMTP_HOST: str = "smtp.gmail.com"
"""Gmail SMTP server hostname for alert emails."""

GMAIL_SMTP_PORT: int = 465
"""Gmail SMTP_SSL port (implicit TLS)."""

ALERT_SUBJECT_PREFIX: str = "[Wake-Up AI]"
"""Subject-line prefix for all alarm email alerts."""


CAPTURE_FRAMES_N: int = 10
"""Number of frames to capture for prefilter + snapshot for vision."""

CAMERA_BACKEND: str = os.getenv("SNN_AGENT_CAMERA_BACKEND", "csi")
"""Camera capture backend: 'csi' (picamera2) or 'usb' (OpenCV VideoCapture).
Set via SNN_AGENT_CAMERA_BACKEND env var."""

CAMERA_USB_INDEX: int = int(os.getenv("SNN_AGENT_CAMERA_INDEX", "0"))
"""V4L2 device index for USB camera (/dev/video<N>).
Set via SNN_AGENT_CAMERA_INDEX env var."""

CAMERA_USB_WARMUP_FRAMES: int = int(os.getenv("SNN_AGENT_CAMERA_WARMUP", "5"))
"""Number of frames to discard on USB camera open (first frames are dark/garbage).
Set via SNN_AGENT_CAMERA_WARMUP env var."""

CAPTURE_INTERVAL_S: float = float(os.getenv("SNN_AGENT_CAPTURE_INTERVAL_S", "0.1"))
"""Sleep (seconds) between consecutive captured frames in both backends.
Ensures consecutive frames are temporally separated so inter-frame motion is detectable.
Set via SNN_AGENT_CAPTURE_INTERVAL_S env var."""

CAMERA_USB_DRAIN_FRAMES: int = int(os.getenv("SNN_AGENT_CAMERA_DRAIN", "4"))
"""Number of stale V4L2-buffered frames to discard (via grab()) after each inter-frame sleep.
Without this drain, read() returns a buffered frame from before the sleep.
Set via SNN_AGENT_CAMERA_DRAIN env var."""

COOLDOWN_S: float = 5.0
"""Seconds to wait before re-armed after wake (debounce SNN chatter)."""

ALARM_HOLD_S: float = 60.0
"""Seconds to blink the local LED (+buzzer, if enabled) before actuators.close() and re-halt."""

ALARM_BLINK_INTERVAL_S: float = 0.5
"""Half-period (seconds) for the alarm LED/buzzer blink during ALARM_HOLD_S."""

LOG_LEVEL: str = os.getenv("SNN_AGENT_LOG_LEVEL", "INFO")
"""Root log level for main (env-overridable). Set via SNN_AGENT_LOG_LEVEL env var."""


POWER_MODE: str = os.getenv("SNN_AGENT_POWER_MODE", "warm")
"""Power state strategy: 'warm' (stay on, dev mode) or 'halt' (sleep → wake).
Set via SNN_AGENT_POWER_MODE env var."""


# CLOUD SYNC (F03, ADR-0014, ADR-0015)

CLOUD_SYNC_ENABLED: bool = (
    os.getenv("SNN_AGENT_CLOUD_SYNC_ENABLED", "true").strip().lower() != "false"
)
"""If False, the Pi never attempts to push events to the cloud dashboard
(local event.log logging is unaffected either way). Set
SNN_AGENT_CLOUD_SYNC_ENABLED=false to disable entirely."""

CLOUD_SYNC_URL: str = os.getenv("SNN_AGENT_CLOUD_SYNC_URL", "").strip()
"""Azure ingest endpoint, e.g. https://<app>.<region>.azurecontainerapps.io/api/events.
Empty disables cloud sync regardless of CLOUD_SYNC_ENABLED (nothing to push
to) -- expected to be empty until T01's Terraform apply produces a real
FQDN. Must be https:// if set; plaintext HTTP is rejected below at import
time (F03 design, Security)."""

if CLOUD_SYNC_URL and not CLOUD_SYNC_URL.startswith("https://"):
    raise ValueError(f"SNN_AGENT_CLOUD_SYNC_URL must start with https://, got: {CLOUD_SYNC_URL!r}")

CLOUD_SYNC_TIMEOUT_S: tuple[float, float] = (
    float(os.getenv("SNN_AGENT_CLOUD_SYNC_CONNECT_TIMEOUT_S", "3")),
    float(os.getenv("SNN_AGENT_CLOUD_SYNC_READ_TIMEOUT_S", "5")),
)
"""(connect, read) timeout tuple for the cloud push, seconds. Never rely on
requests' default (wait forever) -- bounds one push attempt's worst case at
~8s (F03 design, Risks)."""

SYNC_QUEUE_PATH: Path = VAR_DIR / "sync_queue.jsonl"
"""Bounded local backlog of events that failed to push, retried on later
wake cycles (ADR-0015). Sibling to EVENT_LOG; this is a cloud-delivery
worklist only -- EVENT_LOG remains the complete, uncapped local record
regardless of what this queue drops."""

SYNC_QUEUE_MAX_SIZE: int = int(os.getenv("SNN_AGENT_SYNC_QUEUE_MAX_SIZE", "20"))
"""Max pending entries in SYNC_QUEUE_PATH; oldest dropped first once full."""

SYNC_QUEUE_MAX_FLUSH_PER_CYCLE: int = int(
    os.getenv("SNN_AGENT_SYNC_QUEUE_MAX_FLUSH_PER_CYCLE", "5")
)
"""Max queued entries flushed per wake cycle; a flush stops at the first
failure (a dead network fails every later attempt identically)."""

SYNC_QUEUE_MAX_ATTEMPTS: int = int(os.getenv("SNN_AGENT_SYNC_QUEUE_MAX_ATTEMPTS", "5"))
"""Failed-push attempts before a queued entry is dropped (logged) rather
than retried forever -- prevents one permanently-broken entry from blocking
every entry behind it (the queue is always processed oldest-first)."""


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
    cloud_sync_user: str | None = None
    """Shared Basic Auth username for the cloud dashboard (ADR-0009). Unlike
    the four fields above, this is optional: a missing credential disables
    the cloud push (cloud_sync.push() logs once and skips) rather than
    crashing the wake cycle -- see load_settings()."""
    cloud_sync_password: str | None = None
    """Shared Basic Auth password, same credential as cloud_sync_user."""


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

    # Optional: a missing cloud_sync credential disables the cloud push
    # (F03 design) rather than raising here like the four fields above.
    cloud_sync_user = os.getenv("CLOUD_SYNC_USER", "").strip() or None
    cloud_sync_password = os.getenv("CLOUD_SYNC_PASSWORD", "").strip() or None

    return Settings(
        gemini_api_key=api_key,
        gmail_user=gmail_user,
        gmail_app_password=gmail_pw,
        alert_to=alert_to,
        cloud_sync_user=cloud_sync_user,
        cloud_sync_password=cloud_sync_password,
    )
