import itertools
import subprocess
import sys

import pytest

from contracts.validation import fixture
from rpi_agents.cloud.app.policy import ARMED, MANUAL_REVIEW, AlarmPlan, evaluate

STATUSES = ["ok", "unavailable", "error"]
TRISTATE = [True, False, "unknown"]
QUALITIES = ["good", "poor", "unknown"]


def vision(status="ok", glass=True, person=True, quality="good"):
    return fixture("vision-unavailable") | {
        "status": status, "glass_visible": glass, "person_visible": person, "image_quality": quality,
        "error_code": None if status == "ok" else "X", "provenance": "synthetic",
    }  # fmt: skip


def oracle_alarm(policy, status, glass, person, quality, cooldown_active):
    """The only situation that may raise an alarm, written independently of the implementation."""
    return policy == ARMED and status == "ok" and quality == "good" and glass is True and person is True and not cooldown_active


@pytest.mark.parametrize("policy", [MANUAL_REVIEW, ARMED])
@pytest.mark.parametrize("cooldown_active", [False, True])
def test_every_input_combination_follows_the_safety_rules(policy, cooldown_active):
    seen_alarm = False
    for status, glass, person, quality in itertools.product(STATUSES, TRISTATE, TRISTATE, QUALITIES):
        d = evaluate(policy, trigger=True, vision=vision(status, glass, person, quality),
                     seconds_since_last_alarm=5.0 if cooldown_active else None)  # fmt: skip
        combo = (status, glass, person, quality)
        assert (d.status == "alarm_confirmed") == oracle_alarm(policy, *combo, cooldown_active), combo
        assert (d.alarm is not None) == (d.status == "alarm_confirmed"), combo
        assert d.notify == (d.status != "no_alarm"), combo  # a human is told about everything except a clear no
        confident_no = status == "ok" and quality == "good" and glass is False and person is False
        assert (d.status == "no_alarm") == confident_no, combo  # "no alarm" needs a confident, clean result
        assert d.status in ("alarm_confirmed", "no_alarm", "review_required") and d.policy_version == policy
        seen_alarm |= d.status == "alarm_confirmed"
    assert seen_alarm == (policy == ARMED and not cooldown_active)


def test_an_snn_trigger_alone_never_raises_an_alarm():
    for policy in (MANUAL_REVIEW, ARMED):
        for missing in (None, "text", 5, []):
            d = evaluate(policy, trigger=True, vision=missing)
            assert (d.status, d.reason, d.alarm) == ("review_required", "VISION_UNAVAILABLE", None)


def test_without_a_trigger_nothing_happens_whatever_vision_says():
    d = evaluate(ARMED, trigger=False, vision=vision())
    assert (d.status, d.reason, d.alarm, d.notify) == ("no_alarm", "NO_TRIGGER", None, False)


@pytest.mark.parametrize("glass,person", [(1, True), (True, 1), (0, False), ("true", True), (None, True), ("yes", "unknown"), ([], True)])
def test_malformed_vision_values_go_to_review_never_to_an_alarm(glass, person):
    d = evaluate(ARMED, trigger=True, vision=vision(glass=glass, person=person))
    assert (d.status, d.reason, d.alarm) == ("review_required", "VISION_INVALID", None)


def test_unknown_quality_or_status_values_are_not_trusted():
    assert evaluate(ARMED, trigger=True, vision=vision(quality="excellent")).reason == "VISION_INVALID"
    assert evaluate(ARMED, trigger=True, vision=vision(status="maybe")).reason == "VISION_ERROR"
    assert evaluate(ARMED, trigger=True, vision={}).reason == "VISION_ERROR"


def test_reasons_are_specific_enough_to_act_on():
    def why(**kw):
        return evaluate(ARMED, trigger=True, vision=vision(**kw)).reason

    assert why(glass=True, person=True) == "GLASS_AND_PERSON"
    assert why(glass=True, person=False) == "GLASS_WITHOUT_PERSON"
    assert why(glass=False, person=True) == "PERSON_WITHOUT_GLASS"
    assert why(glass=False, person=False) == "NOTHING_VISIBLE"
    assert why(glass="unknown") == "VISION_UNKNOWN" and why(quality="poor") == "IMAGE_QUALITY"
    assert evaluate(MANUAL_REVIEW, trigger=True, vision=vision()).reason == "MANUAL_REVIEW_POLICY"
    assert evaluate(ARMED, trigger=True, vision=vision(status="unavailable")).reason == "VISION_UNAVAILABLE"
    assert evaluate(ARMED, trigger=True, vision=vision(status="error")).reason == "VISION_ERROR"


def test_the_cooldown_blocks_repeat_alarms_but_not_the_first_one():
    assert evaluate(ARMED, trigger=True, vision=vision(), seconds_since_last_alarm=None).status == "alarm_confirmed"
    assert evaluate(ARMED, trigger=True, vision=vision(), seconds_since_last_alarm=61).status == "alarm_confirmed"
    d = evaluate(ARMED, trigger=True, vision=vision(), seconds_since_last_alarm=59)
    assert (d.status, d.reason, d.alarm) == ("review_required", "ALARM_COOLDOWN", None) and d.notify
    assert evaluate(ARMED, trigger=True, vision=vision(), seconds_since_last_alarm=5, cooldown_s=1).status == "alarm_confirmed"


def test_the_alarm_plan_is_carried_and_validated():
    d = evaluate(ARMED, trigger=True, vision=vision(), plan=AlarmPlan(duration_ms=2000, led=True, buzzer=False))
    assert d.alarm == AlarmPlan(2000, True, False)
    for bad in ({"duration_ms": 0}, {"duration_ms": 30001}, {"led": False, "buzzer": False}):
        with pytest.raises(ValueError):
            AlarmPlan(**bad)


def test_an_unknown_policy_is_refused():
    with pytest.raises(ValueError):
        evaluate("armed-v2", trigger=True, vision=vision())


def test_the_module_imports_without_site_packages():
    subprocess.run([sys.executable, "-S", "-c", "import rpi_agents.cloud.app.policy"], check=True)
