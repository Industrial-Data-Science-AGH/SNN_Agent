import threading

import pytest

from contracts.validation import ContractError, validate
from rpi_agents.cloud.app.records import CORE, T_BATCHES, T_DEVICE_COMMANDS, T_EVENTS, T_SESSIONS
from tests.w0.backend_env import QUIET, TRIGGER, Env

SPIKES = [(1200, "zcr")]


def code(exc_info):
    return (exc_info.value.code, exc_info.value.status)


def opened(**settings):
    env = Env(**settings)
    return env, env.open_session()


def session_record(env, state):
    return env.storage.tables.get(T_SESSIONS, "demo-pi", state["session_id"]).data


def test_a_silent_batch_is_acknowledged_stepped_once_and_recorded_durably():
    env, state = opened()
    ack = env.send(state, 0)
    validate("BatchAck", ack)
    assert (ack["status"], ack["received_seq"], ack["processed_seq"], ack["durable_seq"]) == ("running", 0, 0, 0)
    assert (ack["gaps"], ack["commands"], ack["decision"]["trigger"]) == ([], [], False)
    assert len(env.created[0].steps) == 1
    record = session_record(env, state)
    assert (record["received_seq"], record["source_time_us"]) == (0, 250_000)
    stored = env.storage.tables.get(T_BATCHES, state["session_id"], "000000000000").data
    assert stored["state"] == "done" and stored["ack"] == ack


def test_an_exact_retry_returns_the_same_ack_and_never_steps_the_runtime_again():
    env, state = opened()
    first = env.send(state, 0, spikes=SPIKES)
    assert env.send(state, 0, spikes=SPIKES) == first
    assert len(env.created[0].steps) == 1
    with pytest.raises(ContractError) as error:
        env.send(state, 0, spikes=[(5, "zcr")])  # same sequence, different content
    assert code(error) == ("IDEMPOTENCY_CONFLICT", 409)


def test_a_retry_of_a_finished_batch_still_returns_its_ack_after_the_session_stopped_or_the_backend_restarted():
    env, state = opened()
    first = env.send(state, 0)
    env.sessions.stop("demo-pi", state["session_id"], env.stop_body(state))
    assert env.send(state, 0) == first
    env2, state2 = opened()
    ack = env2.send(state2, 0)
    env2.ctx.runtimes.clear()  # the process restarted
    assert env2.send(state2, 0) == ack


def test_a_missing_batch_is_an_explicit_gap_and_a_gap_never_triggers():
    env, state = opened()
    env.ctx.runtimes[state["session_id"]].decisions = [QUIET, TRIGGER]  # the runtime WOULD trigger on the gapped batch
    env.send(state, 0)
    ack = env.send(state, 2)  # batch 1 never arrived
    assert ack["status"] == "gap" and ack["decision"]["status"] == "gap" and ack["decision"]["trigger"] is False
    (gap,) = ack["gaps"]
    assert (gap["reason"], gap["source_start_us"], gap["source_end_us"]) == ("missing_batch", 250_000, 500_000)
    assert ack["commands"] == [] and env.storage.tables.query(T_EVENTS, CORE, rk_prefix="event:") == []


@pytest.mark.parametrize("quality", [{"dropped": 3}, {"clipped": True}])
def test_lost_events_and_clipping_are_reported_as_gaps_over_the_batch_itself(quality):
    env, state = opened()
    ack = env.send(state, 0, **quality)
    (gap,) = ack["gaps"]
    assert gap["reason"] in ("dropped_events", "adc_clipped") and (gap["source_start_us"], gap["source_end_us"]) == (0, 250_000)
    assert ack["status"] == "gap"


def test_out_of_order_and_overlapping_batches_are_refused():
    env, state = opened()
    env.send(state, 0)
    env.send(state, 1)
    with pytest.raises(ContractError) as error:
        env.send(state, 0, spikes=SPIKES, request_id="late")  # older sequence, different content
    assert code(error) == ("IDEMPOTENCY_CONFLICT", 409)
    with pytest.raises(ContractError) as error:
        env.send(state, 5, start=100_000, end=350_000)  # starts before the time already covered
    assert code(error) == ("OUT_OF_ORDER", 409)
    env.send(state, 2)
    with pytest.raises(ContractError) as error:
        env.ingest.ingest("demo-pi", state["session_id"], env.batch(state, 1, request_id="dup", spikes=SPIKES))
    assert error.value.status == 409


def test_a_trigger_records_the_event_the_command_and_the_outbox_together_and_publishes_them():
    env, state = opened()
    env.ctx.runtimes[state["session_id"]].decisions = [TRIGGER]
    ack = env.send(state, 0, spikes=SPIKES)
    (command,) = ack["commands"]
    validate("CaptureCommand", command)
    assert (command["type"], command["mode"], command["parameters"]) == ("capture", "demo", {"frames": 1, "max_bytes": 1_048_576})
    assert ack["decision"]["trigger"] is True and ack["decision"]["event_id"] == command["event_id"]
    event = env.storage.tables.get(T_EVENTS, CORE, f"event:{command['event_id']}").data["event"]
    validate("Event", event)
    assert (event["status"], event["vision"], event["demo"]) == ("photo_requested", None, True)
    assert env.storage.tables.get(T_EVENTS, CORE, f"cmd:{command['command_id']}") is not None
    assert env.publisher.drained()  # the outbox row was published inline and deleted
    indexed = env.storage.tables.get(T_DEVICE_COMMANDS, "demo-pi", command["command_id"]).data
    assert indexed["status"] == "issued" and indexed["command"] == command


def test_a_trigger_alone_never_raises_an_alarm_it_only_requests_a_photo():
    env, state = opened()
    env.ctx.runtimes[state["session_id"]].decisions = [TRIGGER]
    ack = env.send(state, 0, spikes=SPIKES)
    assert [c["type"] for c in ack["commands"]] == ["capture"]


def test_the_command_lifetime_and_limits_come_from_the_settings_and_live_sessions_get_live_commands():
    env = Env(capture_ttl_s=7, capture_frames=2, capture_max_bytes=500_000, allow_live=True)
    state = env.open_session(mode="live")
    env.ctx.runtimes[state["session_id"]].decisions = [TRIGGER]
    (command,) = env.send(state, 0, spikes=SPIKES)["commands"]
    assert command["mode"] == "live" and command["parameters"] == {"frames": 2, "max_bytes": 500_000}
    assert command["issued_at"] == "2026-09-24T12:00:00Z" and command["expires_at"] == "2026-09-24T12:00:07Z"


def test_the_event_cooldown_suppresses_a_second_event_and_the_decision_says_so():
    env, state = opened(event_cooldown_s=20)
    env.ctx.runtimes[state["session_id"]].decisions = [TRIGGER, TRIGGER, TRIGGER]
    first = env.send(state, 0, spikes=SPIKES)
    second = env.send(state, 1, spikes=SPIKES)
    assert first["commands"] and second["commands"] == []
    assert second["decision"]["trigger"] is False and second["decision"]["event_id"] is None  # no event, so no trigger
    env.clock.advance(21)
    assert env.send(state, 2, spikes=SPIKES)["commands"]
    assert len(env.storage.tables.query(T_EVENTS, CORE, rk_prefix="event:")) == 2


def test_warm_up_never_triggers():
    env, state = opened()
    env.ctx.runtimes[state["session_id"]].decisions = [{**TRIGGER, "status": "warmup"}]
    ack = env.send(state, 0, spikes=SPIKES)
    assert (ack["status"], ack["decision"]["status"], ack["decision"]["trigger"], ack["commands"]) == ("warmup", "warmup", False, [])


@pytest.mark.parametrize(
    "bad",
    [{"trigger": "yes", "status": "valid"}, {"trigger": True, "status": "on fire"}, {"status": "valid"}, "text", None,
     {"trigger": True, "status": "valid", "score": True}, {"trigger": True, "status": "valid", "score": "high"}],
)
def test_an_invalid_runtime_answer_ends_the_session_instead_of_being_guessed_at(bad):
    env, state = opened()
    env.ctx.runtimes[state["session_id"]].decisions = [bad]
    with pytest.raises(ContractError) as error:
        env.send(state, 0)
    assert code(error) == ("RUNTIME_ERROR", 503)
    assert session_record(env, state)["state"] == "stopped" and session_record(env, state)["stop_reason"] == "runtime_error"
    with pytest.raises(ContractError) as error:
        env.send(state, 1)
    assert error.value.code == "SESSION_STOPPED"


def test_a_runtime_that_raises_ends_the_session_and_leaks_nothing_about_the_error():
    env, state = opened()
    env.ctx.runtimes[state["session_id"]].step_error = RuntimeError("secret internal detail")
    with pytest.raises(ContractError) as error:
        env.send(state, 0)
    assert "secret" not in str(error.value) and error.value.status == 503


def test_a_crash_between_stepping_and_finishing_ends_the_session_on_retry_never_a_second_step():
    env, state = opened()
    tables, real = env.storage.tables, env.storage.tables.replace

    def dying(table, pk, rk, data, etag):
        if table == T_BATCHES:
            raise KeyboardInterrupt  # the process dies after the runtime stepped and before the ack was stored
        return real(table, pk, rk, data, etag)

    tables.replace = dying
    with pytest.raises(KeyboardInterrupt):
        env.send(state, 0, spikes=SPIKES)
    tables.replace = real
    steps_before = len(env.created[0].steps)
    with pytest.raises(ContractError) as error:
        env.send(state, 0, spikes=SPIKES)  # the retry
    assert code(error) == ("SESSION_LOST", 409) and len(env.created[0].steps) == steps_before
    assert session_record(env, state)["state"] == "stopped"
    assert env.open_session(request_id="create-2")["epoch"] == 2  # the device recovers with a new epoch


def test_after_a_backend_restart_the_next_batch_ends_the_session_explicitly():
    env, state = opened()
    env.send(state, 0)
    env.ctx.runtimes.clear()
    with pytest.raises(ContractError) as error:
        env.send(state, 1)
    assert code(error) == ("SESSION_LOST", 409) and session_record(env, state)["stop_reason"] == "backend_restarted"


@pytest.mark.parametrize(
    "change,expected",
    [({"epoch": 2}, ("SESSION_MISMATCH", 409)), ({"boot_id": "other-boot"}, ("SESSION_MISMATCH", 409)),
     ({"device_id": "other-pi"}, ("SESSION_MISMATCH", 409)),
     ({"encoder_hash": "sha256:" + "0" * 64}, ("ENCODER_MISMATCH", 409)),
     ({"spikes": [(1, "autocorr_lag1")]}, ("UNKNOWN_CHANNEL", 422)),
     ({"schema_version": "2.0"}, ("VERSION_MISMATCH", 409)), ({"extra": 1}, ("INVALID_SCHEMA", 422))],
)
def test_a_batch_that_does_not_belong_to_this_session_is_refused_and_nothing_is_stepped(change, expected):
    env, state = opened()
    with pytest.raises(ContractError) as error:
        env.ingest.ingest("demo-pi", state["session_id"], env.batch(state, 0, **change))
    assert code(error) == expected and env.created[0].steps == []


def test_a_stopped_session_accepts_nothing_new():
    env, state = opened()
    env.sessions.stop("demo-pi", state["session_id"], env.stop_body(state))
    with pytest.raises(ContractError) as error:
        env.send(state, 0)
    assert code(error) == ("SESSION_STOPPED", 409)


def test_two_identical_requests_at_once_step_the_runtime_once_and_get_the_same_ack():
    env, state = opened()
    results, errors = [], []

    def post():
        try:
            results.append(env.send(state, 0, spikes=SPIKES))
        except Exception as exc:  # pragma: no cover - would fail the assertions below
            errors.append(exc)

    threads = [threading.Thread(target=post) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and len(env.created[0].steps) == 1 and all(r == results[0] for r in results)


def test_a_device_cannot_write_into_another_devices_session():
    env, state = opened()
    from rpi_agents.cloud.app.sessions import provision_device

    provision_device(env.ctx, "other-pi")
    with pytest.raises(ContractError) as error:
        env.ingest.ingest("other-pi", state["session_id"], env.batch(state, 0, device_id="other-pi"))
    assert error.value.status in (404, 409) and env.created[0].steps == []


def test_a_failing_inline_publish_never_fails_the_request_and_the_row_survives_for_the_reconciler():
    env, state = opened()
    env.ctx.runtimes[state["session_id"]].decisions = [TRIGGER]

    def broken(*a, **k):
        raise RuntimeError("queue down")

    env.publisher._publish = broken
    ack = env.send(state, 0, spikes=SPIKES)
    assert ack["commands"] and not env.publisher.drained()
    assert env.storage.tables.query(T_DEVICE_COMMANDS, "demo-pi") == []  # not visible to the device yet


def test_the_runtime_gets_its_own_copy_of_the_batch_never_the_callers_object():
    env, state = opened()
    body = env.batch(state, 0, spikes=SPIKES)
    env.ingest.ingest("demo-pi", state["session_id"], body)
    (seen,) = env.created[0].steps
    assert seen == body and seen is not body and seen["spikes"] is not body["spikes"]
