import pytest

from contracts.validation import ContractError, validate
from rpi_agents.cloud.app.records import T_DEVICE_COMMANDS
from rpi_agents.cloud.app.sessions import provision_device
from tests.w0.backend_env import Env


def code(exc_info):
    return (exc_info.value.code, exc_info.value.status)


def with_command(**settings):
    env = Env(**settings)
    state = env.open_session()
    event_id, command = env.trigger(state)
    return env, state, event_id, command


def test_a_fresh_command_is_handed_to_its_device_and_only_to_it():
    env, state, event_id, command = with_command()
    assert env.commands.poll("demo-pi") == [command]
    provision_device(env.ctx, "other-pi")
    assert env.commands.poll("other-pi") == []


def test_a_command_is_no_longer_handed_out_after_it_expires():
    env, state, event_id, command = with_command(capture_ttl_s=10)
    env.clock.advance(9)
    assert env.commands.poll("demo-pi") == [command]
    env.clock.advance(2)
    assert env.commands.poll("demo-pi") == []


def test_a_command_of_a_stopped_or_replaced_session_is_never_handed_out():
    env, state, event_id, command = with_command()
    env.sessions.stop("demo-pi", state["session_id"], env.stop_body(state))
    assert env.commands.poll("demo-pi") == []
    env2, state2, _, command2 = with_command(session_lease_s=1)
    env2.clock.advance(2)
    env2.sessions.create("demo-pi", env2.create_body(request_id="create-2"))  # a new epoch replaces the old session
    assert env2.commands.poll("demo-pi") == []


def test_the_command_moves_forward_through_accepted_to_completed_with_a_real_image():
    env, state, event_id, command = with_command()
    validate("CommandAck", env.ack(command, "accepted"))
    assert env.commands.poll("demo-pi") == [command]  # accepted but not finished: still visible, the device de-duplicates
    image = env.upload(event_id)
    done = env.ack(command, "completed", image_id=image["image_id"])
    assert done["status"] == "completed"
    assert env.commands.poll("demo-pi") == []
    row = env.storage.tables.get(T_DEVICE_COMMANDS, "demo-pi", command["command_id"]).data
    assert row["status"] == "completed" and [a["status"] for a in row["acks"]] == ["accepted", "completed"]


@pytest.mark.parametrize("image_id", [None, "no-such-image", "other-event-0"])
def test_a_completed_capture_must_name_an_image_that_was_stored_for_that_event(image_id):
    env, state, event_id, command = with_command()
    with pytest.raises(ContractError) as error:
        env.ack(command, "completed", image_id=image_id)
    assert code(error) == ("UNKNOWN_IMAGE", 409)
    assert env.storage.tables.get(T_DEVICE_COMMANDS, "demo-pi", command["command_id"]).data["status"] == "issued"


def test_a_terminal_state_is_final():
    env, state, event_id, command = with_command()
    env.ack(command, "failed")
    for status in ("accepted", "completed", "expired", "failed"):
        with pytest.raises(ContractError) as error:
            env.ack(command, status, request_id=f"another-{status}", image_id="x-0")
        assert code(error) == ("COMMAND_TERMINAL", 409)


def test_an_exact_repeat_of_an_ack_returns_the_stored_answer_even_after_the_command_finished():
    env, state, event_id, command = with_command()
    first = env.ack(command, "accepted")
    assert env.ack(command, "accepted") == first
    env.ack(command, "failed")
    assert env.ack(command, "accepted") == first  # the same request id: the same answer, not "terminal"
    with pytest.raises(ContractError) as error:
        env.ack(command, "accepted", error_code="Y_CHANGED")
    assert code(error) == ("IDEMPOTENCY_CONFLICT", 409)


def test_accepting_after_expiry_is_refused_but_reporting_expiry_is_allowed():
    env, state, event_id, command = with_command(capture_ttl_s=10)
    env.clock.advance(11)
    with pytest.raises(ContractError) as error:
        env.ack(command, "accepted")
    assert code(error) == ("COMMAND_EXPIRED", 409)
    assert env.ack(command, "expired")["status"] == "expired"


def test_accepting_on_a_stopped_session_is_refused():
    env, state, event_id, command = with_command()
    env.sessions.stop("demo-pi", state["session_id"], env.stop_body(state))
    with pytest.raises(ContractError) as error:
        env.ack(command, "accepted")
    assert code(error) == ("SESSION_STOPPED", 409)


@pytest.mark.parametrize(
    "changes,expected",
    [({"epoch": 9}, ("SESSION_MISMATCH", 409)), ({"session_id": "other"}, ("SESSION_MISMATCH", 409)),
     ({"device_id": "other-pi"}, ("DEVICE_MISMATCH", 409)), ({"surprise": 1}, ("INVALID_SCHEMA", 422)),
     ({"status": "failed", "error_code": None}, ("INVALID_CONTRACT", 422))],
)
def test_an_ack_that_does_not_belong_to_the_command_is_refused(changes, expected):
    env, state, event_id, command = with_command()
    changes = dict(changes)
    status = changes.pop("status", "accepted")
    with pytest.raises(ContractError) as error:
        env.commands.acknowledge("demo-pi", command["command_id"], env.ack_body(command, status, **changes))
    assert code(error) == expected


def test_the_path_must_name_the_command_and_unknown_or_foreign_commands_are_not_found():
    env, state, event_id, command = with_command()
    with pytest.raises(ContractError) as error:
        env.commands.acknowledge("demo-pi", "another-id", env.ack_body(command, "accepted"))
    assert code(error) == ("COMMAND_MISMATCH", 409)
    provision_device(env.ctx, "other-pi")
    body = env.ack_body(command, "accepted", device_id="other-pi")
    with pytest.raises(ContractError) as error:
        env.commands.acknowledge("other-pi", command["command_id"], body)  # another device cannot touch it
    assert code(error) == ("NOT_FOUND", 404)


def test_an_alarm_command_completes_without_an_image():
    env, state, event_id, command = with_command()
    alarm = command | {"type": "alarm", "command_id": "alarm-1", "policy_version": "armed-glass-and-person-v1",
                       "parameters": {"duration_ms": 1000, "led": True, "buzzer": False}}  # fmt: skip
    env.storage.tables.insert(T_DEVICE_COMMANDS, "demo-pi", "alarm-1", {"command": alarm, "status": "issued", "acks": []})
    assert "alarm-1" in {c["command_id"] for c in env.commands.poll("demo-pi")}
    assert env.ack(alarm, "accepted")["status"] == "accepted"
    assert env.ack(alarm, "completed")["status"] == "completed"
