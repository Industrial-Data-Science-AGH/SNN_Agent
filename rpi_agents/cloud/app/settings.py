"""Backend settings: every tunable in one frozen dataclass, validated on construction. No I/O.

Defaults follow the architecture document. Values that are decisions rather than tuning (the alarm policy, the
alarm plan, recipients) default to the safe choice: manual review only, no recipients.
"""

from __future__ import annotations

from dataclasses import dataclass

from rpi_agents.cloud.app.policy import MANUAL_REVIEW, POLICIES, AlarmPlan


@dataclass(frozen=True)
class Settings:
    policy_version: str = MANUAL_REVIEW
    alarm_plan: AlarmPlan = AlarmPlan()
    alarm_cooldown_s: float = 60.0
    alarm_ttl_s: float = 15.0  # contract: at most 30
    alarm_max_event_age_s: float = 120.0  # an event older than this can no longer raise an alarm
    capture_ttl_s: float = 10.0  # contract: at most 30
    capture_frames: int = 1  # contract: 1..3
    capture_max_bytes: int = 1_048_576
    event_cooldown_s: float = 20.0  # minimum time between two events of one device
    session_lease_s: float = 30.0  # a session silent for longer than this may be replaced
    allow_live: bool = False  # live sessions drive real outputs: opt in deliberately
    vision_max_attempts: int = 3
    vision_visibility_s: float = 60.0
    vision_retry_delay_s: float = 5.0
    notify_max_attempts: int = 3
    notify_retry_delay_s: float = 30.0
    publish_grace_s: float = 10.0  # the reconciler leaves younger outbox rows to the inline publisher
    recipients: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.policy_version not in POLICIES:
            raise ValueError(f"unknown policy {self.policy_version!r}")
        for name in ("alarm_ttl_s", "capture_ttl_s"):
            if not 1 <= getattr(self, name) <= 30:
                raise ValueError(f"{name} must be 1..30 seconds (contract limit)")
        if not 1 <= self.capture_frames <= 3 or not 1 <= self.capture_max_bytes <= 1_048_576:
            raise ValueError("capture_frames must be 1..3 and capture_max_bytes 1..1048576")
        if self.vision_max_attempts < 1 or self.notify_max_attempts < 1:
            raise ValueError("attempt limits must be at least 1")
        for name in ("alarm_cooldown_s", "alarm_max_event_age_s", "event_cooldown_s", "session_lease_s",
                     "vision_visibility_s", "vision_retry_delay_s", "notify_retry_delay_s", "publish_grace_s"):  # fmt: skip
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")
        if self.session_lease_s < 1 or self.vision_visibility_s < 1:
            raise ValueError("session_lease_s and vision_visibility_s must be at least 1")
