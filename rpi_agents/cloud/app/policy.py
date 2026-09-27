"""Alarm policy: the rule that turns an SNN trigger and a vision result into a decision. Pure code, no I/O.

Safety properties, all covered by an exhaustive test over every input combination:
- An SNN trigger alone never raises an alarm; vision must confirm it.
- A vision failure, an unknown answer, a poor or unknown image quality, or a malformed result is never an
  alarm and never a confident "no alarm": it goes to human review.
- Only one input combination can raise an alarm, and only under the armed policy: an `ok` result of good
  quality that shows both glass and a person, outside the cool-down after the previous alarm.

Two policies exist. `manual-review-only-v1` is the default, because the architecture leaves the arming rule
open until it is decided: it never raises an alarm. `armed-glass-and-person-v1` is the variant the
architecture names; enable it deliberately in the backend configuration. How glass-without-person and
person-without-glass are handled (both go to review) is a proposal awaiting that decision.

Events map onto the contract's Event.status: alarm_confirmed, no_alarm, review_required.
"""

from __future__ import annotations

from dataclasses import dataclass

MANUAL_REVIEW = "manual-review-only-v1"
ARMED = "armed-glass-and-person-v1"
POLICIES = (MANUAL_REVIEW, ARMED)
DEFAULT_COOLDOWN_S = 60.0


@dataclass(frozen=True)
class AlarmPlan:
    duration_ms: int = 10_000
    led: bool = True
    buzzer: bool = True

    def __post_init__(self) -> None:
        if not 1 <= self.duration_ms <= 30_000 or not (self.led or self.buzzer):
            raise ValueError("an alarm plan needs 1..30000 ms and at least one output")


@dataclass(frozen=True)
class PolicyDecision:
    status: str  # alarm_confirmed | no_alarm | review_required
    reason: str
    policy_version: str
    alarm: AlarmPlan | None  # present only for alarm_confirmed
    notify: bool  # a human should be told (e-mail); never implied by, or cancelled by, an alarm


def _review(policy: str, reason: str) -> PolicyDecision:
    return PolicyDecision("review_required", reason, policy, None, True)


def evaluate(
    policy_version: str,
    *,
    trigger: bool,
    vision: dict | None,
    seconds_since_last_alarm: float | None = None,
    cooldown_s: float = DEFAULT_COOLDOWN_S,
    plan: AlarmPlan = AlarmPlan(),
) -> PolicyDecision:
    if policy_version not in POLICIES:
        raise ValueError(f"unknown policy {policy_version!r}")
    if not trigger:
        return PolicyDecision("no_alarm", "NO_TRIGGER", policy_version, None, False)
    if not isinstance(vision, dict):
        return _review(policy_version, "VISION_UNAVAILABLE")
    status = vision.get("status")
    if status == "unavailable":
        return _review(policy_version, "VISION_UNAVAILABLE")
    if status != "ok":
        return _review(policy_version, "VISION_ERROR")
    glass, person, quality = vision.get("glass_visible"), vision.get("person_visible"), vision.get("image_quality")
    valid = {True, False, "unknown"}
    if not all(type(v) in (bool, str) and v in valid for v in (glass, person)) or quality not in ("good", "poor", "unknown"):
        return _review(policy_version, "VISION_INVALID")
    if "unknown" in (glass, person):
        return _review(policy_version, "VISION_UNKNOWN")
    if quality != "good":
        return _review(policy_version, "IMAGE_QUALITY")
    if glass and person:
        if policy_version != ARMED:
            return _review(policy_version, "MANUAL_REVIEW_POLICY")
        if seconds_since_last_alarm is not None and seconds_since_last_alarm < cooldown_s:
            return _review(policy_version, "ALARM_COOLDOWN")
        return PolicyDecision("alarm_confirmed", "GLASS_AND_PERSON", policy_version, plan, True)
    if glass:
        return _review(policy_version, "GLASS_WITHOUT_PERSON")
    if person:
        return _review(policy_version, "PERSON_WITHOUT_GLASS")
    return PolicyDecision("no_alarm", "NOTHING_VISIBLE", policy_version, None, False)
