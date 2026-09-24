import threading
import time

import pytest

from contracts.validation import validate
from rpi_agents.cloud.app.notify import DeliveryUnknown, NotificationFailed
from rpi_agents.cloud.app.policy import ARMED
from rpi_agents.cloud.app.records import (
    CORE,
    DEVICES_PK,
    IMAGES,
    Q_NOTIFY,
    Q_NOTIFY_POISON,
    Q_VISION,
    Q_VISION_POISON,
    T_DEVICES,
    T_EVENTS,
    T_VISION_RUNS,
)
from rpi_agents.cloud.app.storage import PreconditionFailed
from rpi_agents.cloud.app.vision import Observation, VisionFailed, VisionUnavailable
from tests.w0.backend_env import Env
from tests.w0.fakes import FakeNotifier, FakeVision

OWNER = ("owner@example.com",)


def obs(glass=True, person=True, quality="good", text="Broken glass near a person."):
    return Observation(glass, person, quality, text)


def armed(**kw):
    kw = {"policy_version": ARMED, "recipients": OWNER, "event_cooldown_s": 0} | kw
    return Env(**kw)


def meta(env, event_id):
    return env.storage.tables.get(T_EVENTS, CORE, f"event:{event_id}").data["meta"]


def kinds(event):
    return [c["type"] for c in event["commands"]]


# ----------------------------------------------------------------------------------- the main flows


def test_glass_and_a_person_under_the_armed_policy_becomes_an_alarm_command_and_an_email():
    env = armed()
    state, event_id, capture = env.pending_event()
    vision, notifier = FakeVision([obs()]), FakeNotifier()
    env.worker(vision, notifier).run_once()
    event = env.events.get(event_id)
    validate("Event", event)
    assert event["status"] == "alarm_confirmed" and kinds(event) == ["capture", "alarm"]
    assert (event["vision"]["status"], event["vision"]["glass_visible"], event["vision"]["authorization"]) == ("ok", True, "unknown")
    alarm = event["commands"][1]
    validate("AlarmCommand", alarm)
    assert alarm["policy_version"] == ARMED and alarm["parameters"] == {"duration_ms": 10_000, "led": True, "buzzer": True}
    assert alarm["expires_at"] == "2026-09-24T12:00:15Z" and alarm["mode"] == "demo"
    assert alarm["command_id"] in {c["command_id"] for c in env.commands.poll("demo-pi")}  # visible to the device
    (mail,) = notifier.sent
    assert (mail.event_id, mail.status, mail.reason, mail.recipients) == (event_id, "alarm_confirmed", "GLASS_AND_PERSON", OWNER)
    assert mail.jpeg is not None and mail.observation == "Broken glass near a person."
    assert meta(env, event_id)["notifications"]["email"]["status"] == "sent"
    assert env.storage.tables.get(T_DEVICES, DEVICES_PK, "demo-pi").data["last_alarm_ms"] is not None


def test_the_models_rationale_and_token_usage_are_kept_with_the_run_but_not_in_the_event():
    env = armed()
    state, event_id, capture = env.pending_event()
    usage = {"prompt_tokens": 300, "completion_tokens": 700, "reasoning_tokens": 400}
    answer = Observation(True, True, "good", "Broken glass near a person.", "Shards under a window; a figure stands beside it.", usage)
    env.worker(FakeVision([answer]), FakeNotifier()).run_once()
    run = env.storage.tables.get(T_VISION_RUNS, event_id, meta(env, event_id)["image_ids"][0]["sha256"]).data
    assert run["state"] == "done" and run["rationale"] == answer.rationale and run["usage"] == usage
    assert "rationale" not in env.events.get(event_id)["vision"]  # the shared contract is unchanged


def test_the_default_policy_never_raises_an_alarm_but_still_tells_a_human():
    env = Env(recipients=OWNER)
    state, event_id, capture = env.pending_event()
    notifier = FakeNotifier()
    env.worker(FakeVision([obs()]), notifier).run_once()
    event = env.events.get(event_id)
    assert (event["status"], kinds(event)) == ("review_required", ["capture"])
    assert notifier.sent[0].reason == "MANUAL_REVIEW_POLICY"


def test_nothing_visible_is_a_clean_no_alarm_and_sends_no_email():
    env = armed()
    state, event_id, capture = env.pending_event()
    notifier = FakeNotifier()
    env.worker(FakeVision([obs(glass=False, person=False)]), notifier).run_once()
    event = env.events.get(event_id)
    assert (event["status"], kinds(event), notifier.sent) == ("no_alarm", ["capture"], [])


@pytest.mark.parametrize("glass,person,quality", [(True, False, "good"), (False, True, "good"), ("unknown", True, "good"), (True, True, "poor")])
def test_anything_short_of_a_clear_glass_and_person_goes_to_a_human_not_to_an_alarm(glass, person, quality):
    env = armed()
    state, event_id, capture = env.pending_event()
    notifier = FakeNotifier()
    env.worker(FakeVision([obs(glass, person, quality)]), notifier).run_once()
    event = env.events.get(event_id)
    assert (event["status"], kinds(event)) == ("review_required", ["capture"]) and len(notifier.sent) == 1


@pytest.mark.parametrize("mode,provenance", [("demo", "demo"), ("replay", "synthetic")])
def test_the_vision_result_carries_the_provenance_of_the_session(mode, provenance):
    env = armed()
    state, event_id, capture = env.pending_event(mode=mode)
    env.worker(FakeVision([obs()]), FakeNotifier()).run_once()
    assert env.events.get(event_id)["vision"]["provenance"] == provenance


# -------------------------------------------------------------------------------- vision failures


def test_a_flaky_provider_is_retried_with_a_delay_and_the_event_stays_pending_meanwhile():
    env = armed(vision_retry_delay_s=5)
    state, event_id, capture = env.pending_event()
    vision = FakeVision([VisionUnavailable("VISION_TIMEOUT"), VisionUnavailable("VISION_SERVER"), obs()])
    worker = env.worker(vision, FakeNotifier())
    worker.run_once()
    assert env.events.get(event_id)["status"] == "analyzing" and len(vision.calls) == 1
    assert worker.run_once()["vision"] == 0 and len(vision.calls) == 1  # inside the delay: nothing is retried yet
    env.clock.advance(6)
    worker.run_once()
    assert env.events.get(event_id)["status"] == "analyzing" and len(vision.calls) == 2
    env.clock.advance(6)
    worker.run_once()
    assert env.events.get(event_id)["status"] == "alarm_confirmed" and len(vision.calls) == 3
    assert env.storage.queues.depth(Q_VISION) == 0 and env.storage.queues.depth(Q_VISION_POISON) == 0


def test_when_retries_run_out_the_job_is_set_aside_but_the_event_still_goes_to_a_human():
    env = armed()
    state, event_id, capture = env.pending_event()
    vision, notifier = FakeVision([VisionUnavailable("VISION_TIMEOUT")] * 3), FakeNotifier()
    worker = env.worker(vision, notifier)
    for _ in range(3):
        worker.run_once()
        env.clock.advance(6)
    event = env.events.get(event_id)
    assert event["status"] == "review_required" and (event["vision"]["status"], event["vision"]["error_code"]) == ("unavailable", "VISION_TIMEOUT")
    assert kinds(event) == ["capture"] and len(vision.calls) == 3
    assert env.storage.queues.depth(Q_VISION) == 0
    (poisoned,) = env.storage.queues.receive(Q_VISION_POISON, visibility_s=30)
    assert poisoned.body["event_id"] == event_id and poisoned.body["reason"] == "VISION_TIMEOUT"
    assert notifier.sent[0].reason == "VISION_UNAVAILABLE"  # a human learns that something happened and could not be judged


@pytest.mark.parametrize("failure,code", [(VisionFailed("VISION_REFUSED"), "VISION_REFUSED"), (VisionFailed("VISION_BAD_RESPONSE"), "VISION_BAD_RESPONSE")])
def test_a_permanent_vision_failure_is_not_retried_and_is_never_an_alarm(failure, code):
    env = armed()
    state, event_id, capture = env.pending_event()
    vision = FakeVision([failure])
    env.worker(vision, FakeNotifier()).run_once()
    event = env.events.get(event_id)
    assert (event["status"], event["vision"]["status"], event["vision"]["error_code"], kinds(event)) == ("review_required", "error", code, ["capture"])
    assert len(vision.calls) == 1 and env.storage.queues.depth(Q_VISION) == 0


def test_a_missing_image_blob_resolves_to_review_instead_of_looping():
    env = armed()
    state, event_id, capture = env.pending_event()
    env.storage.blobs.delete(IMAGES, f"{event_id}/0.jpg")
    vision = FakeVision([])
    env.worker(vision, FakeNotifier()).run_once()
    event = env.events.get(event_id)
    assert (event["status"], event["vision"]["error_code"]) == ("review_required", "IMAGE_MISSING") and vision.calls == []


# ---------------------------------------------------------------------------------- duplicates


def test_a_duplicate_delivery_is_recognised_and_analysed_once():
    env = armed()
    state, event_id, capture = env.pending_event()
    env.storage.queues.send(Q_VISION, {"event_id": event_id, "image_id": f"{event_id}-0"})  # at-least-once: a second copy
    vision, notifier = FakeVision([obs(), obs()]), FakeNotifier()
    env.worker(vision, notifier).run_once()
    assert len(vision.calls) == 1 and len(notifier.sent) == 1 and env.storage.queues.depth(Q_VISION) == 0


def test_a_worker_that_dies_mid_job_is_taken_over_after_the_run_goes_stale_and_the_event_resolves_once():
    env = armed(vision_visibility_s=60, alarm_max_event_age_s=600)  # the takeover happens 122 s after the event
    state, event_id, capture = env.pending_event()
    vision, notifier = FakeVision([obs(), obs()]), FakeNotifier()
    worker = env.worker(vision, notifier)
    real = worker._resolve
    worker._resolve = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("worker crashed"))
    worker.run_once()  # analysed, then crashed before recording anything
    worker._resolve = real
    assert env.events.get(event_id)["status"] == "analyzing" and len(vision.calls) == 1
    env.clock.advance(61)
    worker.run_once()  # redelivered, but the first run is only 61 s old: not stale, so it is left alone
    assert len(vision.calls) == 1 and env.events.get(event_id)["status"] == "analyzing"
    env.clock.advance(61)
    worker.run_once()  # now 122 s: stale, taken over
    assert len(vision.calls) == 2 and env.events.get(event_id)["status"] == "alarm_confirmed"
    assert len(notifier.sent) == 1 and kinds(env.events.get(event_id)).count("alarm") == 1


def test_orphan_jobs_are_dropped():
    env = armed()
    env.storage.queues.send(Q_VISION, {"event_id": "ghost", "image_id": "ghost-0"})
    env.storage.queues.send(Q_NOTIFY, {"event_id": "ghost", "reason": "X"})
    env.worker(FakeVision([]), FakeNotifier()).run_once()
    assert env.storage.queues.depth(Q_VISION) == 0 and env.storage.queues.depth(Q_NOTIFY) == 0


def test_a_handler_that_keeps_crashing_is_eventually_set_aside_and_its_event_still_reaches_a_human():
    env = armed()
    state, event_id, capture = env.pending_event()
    vision = FakeVision([RuntimeError("bug")] * 20)
    worker = env.worker(vision, FakeNotifier())
    for _ in range(12):
        worker.run_once()
        env.clock.advance(61 * 2)
    event = env.events.get(event_id)
    assert (event["status"], event["vision"]["error_code"]) == ("review_required", "WORKER_ERROR")
    assert env.storage.queues.depth(Q_VISION) == 0
    (poisoned,) = env.storage.queues.receive(Q_VISION_POISON, visibility_s=30)
    assert poisoned.body["reason"] == "TOO_MANY_DELIVERIES"


# ------------------------------------------------------------------------------ leases and claims


def test_the_queue_message_is_kept_invisible_while_a_slow_model_call_runs():
    env = armed(vision_visibility_s=60)
    state, event_id, capture = env.pending_event()
    release = threading.Event()
    vision = FakeVision([obs()], block=release)
    worker = env.worker(vision, FakeNotifier(), renew_interval_s=0.02)
    thread = threading.Thread(target=worker.run_once)
    thread.start()
    try:
        assert vision.entered.wait(5)
        for _ in range(2):  # two visibility periods pass while the call is still running
            time.sleep(0.15)
            env.clock.advance(40)
        time.sleep(0.15)
        assert env.storage.queues.receive(Q_VISION, visibility_s=60) == []  # nobody else got the job: it was renewed
    finally:
        release.set()
        thread.join(10)
    assert env.events.get(event_id)["status"] == "alarm_confirmed"


def test_a_worker_that_lost_its_claim_throws_its_result_away_and_the_new_owner_finishes_the_job():
    env = armed(vision_visibility_s=60)
    state, event_id, capture = env.pending_event()
    release = threading.Event()
    slow = env.worker(FakeVision([obs()], block=release), FakeNotifier(), renew_interval_s=0.02)
    thread = threading.Thread(target=slow.run_once)
    thread.start()
    try:
        assert slow.vision.entered.wait(5)
        env.clock.advance(61)  # the slow worker's claim lapses ...
        (stolen,) = env.storage.queues.receive(Q_VISION, visibility_s=60)  # ... and another worker receives the job
        thread.join(5)
        assert not thread.is_alive()  # the slow worker noticed on its next renewal and gave up
        assert env.events.get(event_id)["status"] == "analyzing"  # its result was not applied
        env.storage.queues.delete(Q_VISION, stolen)
        env.storage.queues.send(Q_VISION, stolen.body)  # the new owner's own delivery of the same job
        fast = env.worker(FakeVision([obs()]), FakeNotifier())
        fast.run_once()
    finally:
        release.set()
    assert env.events.get(event_id)["status"] == "alarm_confirmed"
    assert env.storage.tables.get(T_VISION_RUNS, event_id, meta(env, event_id)["image_ids"][0]["sha256"]).data["state"] == "done"


# ---------------------------------------------------------------- the alarm guards (downgrades)


def test_an_alarm_is_downgraded_when_its_session_is_no_longer_active():
    env = armed()
    state, event_id, capture = env.pending_event()
    env.sessions.stop("demo-pi", state["session_id"], env.stop_body(state))
    notifier = FakeNotifier()
    env.worker(FakeVision([obs()]), notifier).run_once()
    event = env.events.get(event_id)
    assert (event["status"], kinds(event)) == ("review_required", ["capture"])
    assert meta(env, event_id)["policy"]["reason"] == "SESSION_NOT_ACTIVE" and len(notifier.sent) == 1


def test_an_event_that_is_too_old_can_no_longer_raise_an_alarm():
    env = armed(alarm_max_event_age_s=120)
    state, event_id, capture = env.pending_event()
    env.clock.advance(121)
    env.worker(FakeVision([obs()]), FakeNotifier()).run_once()
    assert meta(env, event_id)["policy"]["reason"] == "EVENT_TOO_OLD" and env.events.get(event_id)["status"] == "review_required"


def test_two_events_cannot_both_alarm_inside_the_cooldown():
    env = armed(alarm_cooldown_s=60)
    state = env.open_session()
    ids = []
    for seq in range(3):
        event_id, _ = env.trigger(state, seq)
        env.upload(event_id)
        ids.append(event_id)
        env.worker(FakeVision([obs()]), FakeNotifier()).run_once()
        env.clock.advance(30 if seq == 0 else 40)
    statuses = [env.events.get(i)["status"] for i in ids]
    assert statuses == ["alarm_confirmed", "review_required", "alarm_confirmed"]
    assert meta(env, ids[1])["policy"]["reason"] == "ALARM_COOLDOWN"


def test_a_concurrent_alarm_claimed_between_the_check_and_our_claim_still_wins_the_cooldown():
    env = armed(alarm_cooldown_s=60)
    state, event_id, capture = env.pending_event()
    worker = env.worker(FakeVision([obs()]), FakeNotifier())
    real = worker._session_active

    def another_worker_claims_first(event):  # runs after the policy looked, before this worker claims
        device = env.storage.tables.get(T_DEVICES, DEVICES_PK, "demo-pi")
        env.storage.tables.replace(T_DEVICES, DEVICES_PK, "demo-pi", device.data | {"last_alarm_ms": int(env.clock.seconds() * 1000)}, device.etag)
        return real(event)

    worker._session_active = another_worker_claims_first
    worker.run_once()
    assert env.events.get(event_id)["status"] == "review_required"
    assert meta(env, event_id)["policy"]["reason"] == "ALARM_COOLDOWN" and kinds(env.events.get(event_id)) == ["capture"]


def test_a_conflict_while_recording_the_decision_is_retried_without_the_alarm_blocking_itself():
    env = armed()
    state, event_id, capture = env.pending_event()
    tables, real = env.storage.tables, env.storage.tables.transaction
    calls = {"n": 0}

    def flaky(table, pk, ops):
        if table == T_EVENTS and any(op.rk.startswith("event:") and op.kind == "replace" for op in ops) and calls["n"] == 0:
            calls["n"] += 1
            raise PreconditionFailed("someone touched the event")
        return real(table, pk, ops)

    tables.transaction = flaky
    env.worker(FakeVision([obs()]), FakeNotifier()).run_once()
    assert calls["n"] == 1 and env.events.get(event_id)["status"] == "alarm_confirmed"  # not "ALARM_COOLDOWN" from its own claim


# --------------------------------------------------------------------------------- notifications


def test_without_recipients_nothing_is_sent_and_that_is_recorded():
    env = armed(recipients=())
    state, event_id, capture = env.pending_event()
    notifier = FakeNotifier()
    env.worker(FakeVision([obs()]), notifier).run_once()
    assert notifier.sent == [] and meta(env, event_id)["notifications"]["email"]["status"] == "not_configured"
    assert env.events.get(event_id)["status"] == "alarm_confirmed"


def test_a_failing_mailbox_is_retried_then_reported_and_never_cancels_the_alarm():
    env = armed(notify_retry_delay_s=30)
    state, event_id, capture = env.pending_event()
    notifier = FakeNotifier([NotificationFailed("SMTP_UNREACHABLE", True)] * 3)
    worker = env.worker(FakeVision([obs()]), notifier)
    worker.run_once()
    assert meta(env, event_id)["notifications"]["email"]["status"] == "retrying"
    for _ in range(2):
        env.clock.advance(31)
        worker.run_once()
    assert meta(env, event_id)["notifications"]["email"]["status"] == "failed" and len(notifier.sent) == 3
    assert env.storage.queues.depth(Q_NOTIFY_POISON) == 1
    assert "alarm" in kinds(env.events.get(event_id)) and env.events.get(event_id)["status"] == "alarm_confirmed"


def test_a_transient_email_failure_that_recovers_is_sent_once():
    env = armed()
    state, event_id, capture = env.pending_event()
    notifier = FakeNotifier([NotificationFailed("SMTP_452", True)])
    worker = env.worker(FakeVision([obs()]), notifier)
    worker.run_once()
    env.clock.advance(31)
    worker.run_once()
    assert meta(env, event_id)["notifications"]["email"]["status"] == "sent" and len(notifier.sent) == 2  # one failed try, one success


def test_a_permanent_email_failure_is_not_retried():
    env = armed()
    state, event_id, capture = env.pending_event()
    notifier = FakeNotifier([NotificationFailed("SMTP_550", False)])
    env.worker(FakeVision([obs()]), notifier).run_once()
    entry = meta(env, event_id)["notifications"]["email"]
    assert (entry["status"], entry["code"], len(notifier.sent)) == ("failed", "SMTP_550", 1)
    assert env.storage.queues.depth(Q_NOTIFY) == 0 and env.storage.queues.depth(Q_NOTIFY_POISON) == 0


def test_an_ambiguous_email_outcome_is_recorded_as_unknown_and_never_resent():
    env = armed()
    state, event_id, capture = env.pending_event()
    notifier = FakeNotifier([DeliveryUnknown("timeout after the body")])
    worker = env.worker(FakeVision([obs()]), notifier)
    worker.run_once()
    env.storage.queues.send(Q_NOTIFY, {"event_id": event_id, "reason": "X"})  # even a duplicate job must not resend it
    env.clock.advance(300)
    worker.run_once()
    assert meta(env, event_id)["notifications"]["email"]["status"] == "delivery_unknown" and len(notifier.sent) == 1


def test_a_duplicate_notification_job_after_success_sends_nothing_more():
    env = armed()
    state, event_id, capture = env.pending_event()
    notifier = FakeNotifier()
    worker = env.worker(FakeVision([obs()]), notifier)
    worker.run_once()
    env.storage.queues.send(Q_NOTIFY, {"event_id": event_id, "reason": "X"})
    worker.run_once()
    assert len(notifier.sent) == 1


# ------------------------------------------------------------------------------------- the loop


def test_the_reconciler_publishes_rows_a_crash_left_behind_before_the_worker_processes_them():
    env = armed()
    env.publisher._publish = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("queue down"))
    state, event_id, capture = env.pending_event()  # the upload's vision job was recorded but not enqueued
    assert env.storage.queues.depth(Q_VISION) == 0
    del env.publisher._publish
    vision = FakeVision([obs()])
    worker = env.worker(vision, FakeNotifier())
    assert worker.run_once()["vision"] == 0  # too young for the reconciler: the request that wrote it may still be on it
    env.clock.advance(11)
    counts = worker.run_once()
    assert counts["published"] >= 1 and len(vision.calls) == 1 and env.events.get(event_id)["status"] == "alarm_confirmed"


def test_drained_is_false_until_everything_is_published_and_processed():
    env = armed()
    worker = env.worker(FakeVision([obs()]), FakeNotifier())
    assert worker.drained()
    state, event_id, capture = env.pending_event()
    assert not worker.drained()
    worker.run_once()
    assert worker.drained()


def test_run_forever_survives_a_failing_round_and_stops_when_asked():
    env = armed()
    worker = env.worker(FakeVision([]), FakeNotifier())
    real, calls = worker.run_once, {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("storage hiccup")
        return real(*a, **k)

    worker.run_once = flaky
    stop = threading.Event()
    thread = threading.Thread(target=worker.run_forever, args=(stop, 0.01))
    thread.start()
    time.sleep(0.2)
    stop.set()
    thread.join(5)
    assert not thread.is_alive() and calls["n"] >= 2
