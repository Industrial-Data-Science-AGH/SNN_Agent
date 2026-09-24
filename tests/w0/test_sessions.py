import pytest

from contracts.validation import ContractError, validate
from rpi_agents.cloud.app.records import DEVICES_PK, T_DEVICES, T_SESSIONS
from rpi_agents.cloud.app.storage import PreconditionFailed
from tests.w0.backend_env import Env


def code(exc_info):
    return (exc_info.value.code, exc_info.value.status)


def test_creating_a_session_issues_epoch_one_and_starts_the_runtime():
    env = Env()
    state = env.open_session()
    validate("SessionState", state)
    assert (state["epoch"], state["state"], state["received_seq"], state["demo"]) == (1, "running", None, True)
    (runtime,) = env.created
    assert runtime.loaded == env.manifest and runtime.resets == [(1, 0)]
    assert env.ctx.runtimes[state["session_id"]] is runtime
    device = env.storage.tables.get(T_DEVICES, DEVICES_PK, "demo-pi").data
    assert (device["epoch"], device["active_session_id"]) == (1, state["session_id"])


def test_an_exact_retry_returns_the_same_session_and_never_a_second_one():
    env = Env()
    body = env.create_body()
    first = env.sessions.create("demo-pi", body)
    assert env.sessions.create("demo-pi", body) == first and len(env.created) == 1
    with pytest.raises(ContractError) as error:
        env.sessions.create("demo-pi", body | {"source_start_us": 5})
    assert code(error) == ("IDEMPOTENCY_CONFLICT", 409)


def test_a_second_session_is_refused_while_the_first_is_alive_and_allowed_after_its_lease():
    env = Env(session_lease_s=30)
    first = env.open_session()
    with pytest.raises(ContractError) as error:
        env.sessions.create("demo-pi", env.create_body(request_id="create-2"))
    assert code(error) == ("SESSION_ACTIVE", 409)
    env.clock.advance(31)
    second = env.sessions.create("demo-pi", env.create_body(request_id="create-2"))
    assert second["epoch"] == 2 and second["session_id"] != first["session_id"]
    old = env.storage.tables.get(T_SESSIONS, "demo-pi", first["session_id"]).data
    assert (old["state"], old["stop_reason"]) == ("stopped", "lease_expired")
    assert first["session_id"] not in env.ctx.runtimes and second["session_id"] in env.ctx.runtimes


def test_stopping_is_idempotent_frees_the_device_and_drops_the_runtime():
    env = Env()
    state = env.open_session()
    body = env.stop_body(state)
    stopped = env.sessions.stop("demo-pi", state["session_id"], body)
    assert stopped["state"] == "stopped" and env.sessions.stop("demo-pi", state["session_id"], body) == stopped
    assert state["session_id"] not in env.ctx.runtimes
    assert env.storage.tables.get(T_DEVICES, DEVICES_PK, "demo-pi").data["active_session_id"] is None
    again = env.sessions.create("demo-pi", env.create_body(request_id="create-2"))  # no lease wait needed
    assert again["epoch"] == 2


def test_stop_checks_device_session_and_epoch():
    env = Env()
    state = env.open_session()
    for change in ({"epoch": 2}, {"session_id": "other"}, {"device_id": "other-pi"}):
        with pytest.raises(ContractError) as error:
            env.sessions.stop("demo-pi", state["session_id"], env.stop_body(state, request_id="s-x", **change))
        assert error.value.status in (409, 404)
    with pytest.raises(ContractError) as error:
        env.sessions.stop("demo-pi", "missing", env.stop_body(state, request_id="s-y", session_id="missing"))
    assert code(error) == ("NOT_FOUND", 404)
    assert env.sessions.get("demo-pi", state["session_id"])["state"] == "running"


@pytest.mark.parametrize(
    "changes,expected",
    [({"model_hash": "sha256:" + "0" * 64}, ("MODEL_MISMATCH", 409)),
     ({"encoder_hash": "sha256:" + "1" * 64}, ("ENCODER_MISMATCH", 409)),
     ({"mode": "live"}, ("LIVE_DISABLED", 409)),
     ({"device_id": "other-pi"}, ("DEVICE_MISMATCH", 409)),
     ({"surprise": 1}, ("INVALID_SCHEMA", 422))],
)  # fmt: skip
def test_a_create_request_that_does_not_match_this_backend_is_refused(changes, expected):
    env = Env()
    with pytest.raises(ContractError) as error:
        env.sessions.create("demo-pi", env.create_body(**changes))
    assert code(error) == expected and env.created == []  # refused before a runtime was even created
    assert env.storage.tables.get(T_DEVICES, DEVICES_PK, "demo-pi").data["epoch"] == 0  # nothing was allocated


def test_live_sessions_need_an_explicit_opt_in():
    env = Env(allow_live=True)
    state = env.open_session(mode="live")
    assert state["demo"] is False and state["mode"] == "live"


def test_a_device_that_was_never_provisioned_cannot_open_a_session():
    env = Env()
    with pytest.raises(ContractError) as error:
        env.sessions.create("stranger", env.create_body(device_id="stranger"))
    assert code(error) == ("UNKNOWN_DEVICE", 403)


@pytest.mark.parametrize("failing", ["load_error", "reset_error"])
def test_a_runtime_that_refuses_leaves_no_session_and_no_epoch_behind(failing):
    env = Env()
    env.runtime_kwargs = {failing: RuntimeError("incompatible")}
    with pytest.raises(ContractError) as error:
        env.open_session()
    assert code(error) == ("RUNTIME_REFUSED", 409) and "incompatible" not in error.value.args[0]
    assert env.storage.tables.query(T_SESSIONS, "demo-pi") == [] and env.ctx.runtimes == {}
    assert env.storage.tables.get(T_DEVICES, DEVICES_PK, "demo-pi").data["epoch"] == 0


def test_epochs_only_increase_across_many_sessions():
    env = Env(session_lease_s=1)
    epochs = []
    for i in range(4):
        epochs.append(env.sessions.create("demo-pi", env.create_body(request_id=f"create-{i}"))["epoch"])
        env.clock.advance(2)
    assert epochs == [1, 2, 3, 4]


def test_a_lost_race_for_the_epoch_is_retried_not_duplicated():
    env = Env()
    tables, real = env.storage.tables, env.storage.tables.replace
    calls = {"n": 0}

    def flaky(table, pk, rk, data, etag):
        if table == T_DEVICES and calls["n"] == 0:
            calls["n"] += 1
            raise PreconditionFailed("someone else changed it")
        return real(table, pk, rk, data, etag)

    tables.replace = flaky
    state = env.open_session()
    assert state["epoch"] == 1 and calls["n"] == 1  # the second attempt won; only one session exists
    assert len(env.storage.tables.query(T_SESSIONS, "demo-pi")) == 1


def test_a_stale_device_pointer_after_a_crash_does_not_lock_the_device_out():
    env = Env()
    env.open_session()
    env.storage.tables.delete(T_SESSIONS, "demo-pi", env.storage.tables.query(T_SESSIONS, "demo-pi")[0].rk)
    second = env.sessions.create("demo-pi", env.create_body(request_id="create-2"))  # pointer names a missing session
    assert second["epoch"] == 2


def test_interrupting_a_session_ends_it_and_frees_the_device():
    env = Env()
    state = env.open_session()
    env.sessions.interrupt("demo-pi", state["session_id"], "runtime_state_unknown")
    record = env.storage.tables.get(T_SESSIONS, "demo-pi", state["session_id"]).data
    assert (record["state"], record["stop_reason"]) == ("stopped", "runtime_state_unknown")
    assert env.sessions.create("demo-pi", env.create_body(request_id="create-2"))["epoch"] == 2
