import io
import logging

import pytest

from contracts.validation import ContractError
from rpi_agents.cloud.app import auth
from rpi_agents.cloud.app.auth import (
    OperatorAuth,
    RedactingFilter,
    authenticate_device,
    check_csrf,
    client_ip,
    hash_password,
    issue_device_token,
    redact,
    verify_password,
)
from rpi_agents.cloud.app.records import DEVICES_PK, T_DEVICES
from tests.w0.backend_env import Env

PASSWORD = "correct horse battery staple"
VERIFIER = hash_password(PASSWORD, log2_n=14)


def make_auth(env=None, **kw):
    env = env or Env()
    return env, OperatorAuth(env.ctx, username="operator", password_hash=VERIFIER, **kw)


def code(exc_info):
    return (exc_info.value.code, exc_info.value.status)


# ------------------------------------------------------------------------------------------ passwords


def test_a_password_verifies_only_against_its_own_verifier():
    assert verify_password(PASSWORD, VERIFIER) and not verify_password("wrong", VERIFIER) and not verify_password("", VERIFIER)
    assert VERIFIER.startswith("scrypt$14$8$1$") and PASSWORD not in VERIFIER
    assert hash_password(PASSWORD, log2_n=14) != VERIFIER  # a fresh salt every time
    assert verify_password(PASSWORD, hash_password(PASSWORD, log2_n=14))


@pytest.mark.parametrize("stored", ["", "plain-text", "scrypt$14$8$1$AAAA", "scrypt$x$8$1$AAAAAAAAAAAAAAAA$AAAA", "sha256$1$2$3$4$5",
                                    None, "scrypt$10$8$1$AAAAAAAAAAAAAAAA$" + "A" * 43 + "=", "scrypt$14$8$1$!!!!$!!!!"])  # fmt: skip
def test_malformed_or_weak_verifiers_never_verify_and_never_crash(stored):
    assert verify_password(PASSWORD, stored) is False


def test_weak_or_empty_passwords_are_refused_when_hashing():
    with pytest.raises(ValueError):
        hash_password("")
    with pytest.raises(ValueError, match="cost"):
        hash_password("x", log2_n=13)


# ------------------------------------------------------------------------------ device credentials


def test_a_device_token_is_returned_once_stored_only_as_a_hash_and_authenticates_only_its_device():
    env = Env()
    token = issue_device_token(env.ctx, "demo-pi")
    device_id, secret = token.split("~")
    assert device_id == "demo-pi" and len(secret) >= 32
    row = env.storage.tables.get(T_DEVICES, DEVICES_PK, "demo-pi").data
    assert secret not in str(row) and len(row["token_hash"]) == 64  # only a hash
    assert authenticate_device(env.ctx, token) == "demo-pi"


@pytest.mark.parametrize("bad", [None, "", "demo-pi", "demo-pi~", "demo-pi~short", "~" + "a" * 40, "demo-pi~" + "a" * 43, "a b~" + "a" * 43,
                                 "demo-pi~" + "a" * 43 + " ", "Bearer x"])  # fmt: skip
def test_malformed_or_wrong_tokens_authenticate_nobody(bad):
    env = Env()
    issue_device_token(env.ctx, "demo-pi")
    assert authenticate_device(env.ctx, bad) is None


def test_a_token_cannot_be_moved_to_another_device_and_unknown_or_disabled_devices_fail():
    env = Env()
    token = issue_device_token(env.ctx, "demo-pi")
    other = issue_device_token(env.ctx, "other-pi")
    secret = token.split("~")[1]
    assert authenticate_device(env.ctx, f"other-pi~{secret}") is None  # the secret belongs to demo-pi
    assert authenticate_device(env.ctx, other) == "other-pi"
    assert authenticate_device(env.ctx, "ghost~" + secret) is None
    row = env.storage.tables.get(T_DEVICES, DEVICES_PK, "demo-pi")
    env.storage.tables.replace(T_DEVICES, DEVICES_PK, "demo-pi", row.data | {"active": False}, row.etag)
    assert authenticate_device(env.ctx, token) is None


def test_rotating_a_credential_invalidates_the_old_one():
    env = Env()
    old = issue_device_token(env.ctx, "demo-pi")
    new = issue_device_token(env.ctx, "demo-pi")
    assert old != new and authenticate_device(env.ctx, old) is None and authenticate_device(env.ctx, new) == "demo-pi"


def test_a_device_without_a_credential_cannot_be_authenticated_with_an_empty_secret():
    env = Env()  # demo-pi is provisioned without a token hash
    assert authenticate_device(env.ctx, "demo-pi~" + "a" * 43) is None


# ------------------------------------------------------------------------------------ operator login


@pytest.mark.parametrize("username,verifier", [("", VERIFIER), ("operator", ""), ("operator", "plaintext"),
                                               ("operator", "scrypt$10$8$1$AAAAAAAAAAAAAAAA$" + "A" * 43 + "=")])  # fmt: skip
def test_there_are_no_default_credentials_and_weak_verifiers_are_refused_at_startup(username, verifier):
    with pytest.raises(ValueError):
        OperatorAuth(Env().ctx, username=username, password_hash=verifier)


def test_a_successful_login_creates_a_server_side_session_whose_token_is_never_stored():
    env, operator = make_auth()
    token, web = operator.login("operator", PASSWORD, "1.2.3.4")
    assert web.actor == "shared_operator" and web.csrf_token != token and len(token) >= 40
    rows = env.storage.tables.query(auth.T_WEB_SESSIONS, auth.WEB_PK)
    assert len(rows) == 1 and token not in str(rows[0].data) and token not in rows[0].rk  # only the hash is stored
    assert operator.session(token) == web


def test_a_wrong_username_and_a_wrong_password_are_indistinguishable_and_cost_the_same_work(monkeypatch):
    env, operator = make_auth()
    calls = []
    real = auth.verify_password
    monkeypatch.setattr(auth, "verify_password", lambda password, stored: calls.append(stored) or real(password, stored))
    with pytest.raises(ContractError) as wrong_password:
        operator.login("operator", "nope", "1.1.1.1")
    with pytest.raises(ContractError) as wrong_user:
        operator.login("intruder", PASSWORD, "1.1.1.1")
    assert code(wrong_password) == code(wrong_user) == ("INVALID_CREDENTIALS", 401)
    assert str(wrong_password.value) == str(wrong_user.value) and len(calls) == 2  # a verification ran in both cases


def test_the_correct_password_for_the_wrong_username_fails():
    env, operator = make_auth()
    with pytest.raises(ContractError) as error:
        operator.login("Operator", PASSWORD, "1.1.1.1")  # case matters
    assert error.value.code == "INVALID_CREDENTIALS"


def test_repeated_failures_lock_the_client_out_even_for_the_right_password_and_it_expires():
    env, operator = make_auth(max_failures=3, lockout_s=300)
    for _ in range(3):
        with pytest.raises(ContractError):
            operator.login("operator", "bad", "9.9.9.9")
    with pytest.raises(ContractError) as error:
        operator.login("operator", PASSWORD, "9.9.9.9")
    assert code(error) == ("TOO_MANY_ATTEMPTS", 429) and operator.retry_after_s("9.9.9.9") >= 1
    assert operator.login("operator", PASSWORD, "8.8.8.8")  # another client is unaffected
    env.clock.advance(301)
    assert operator.login("operator", PASSWORD, "9.9.9.9")


def test_a_successful_login_clears_the_failure_count():
    env, operator = make_auth(max_failures=3)
    for _ in range(2):
        with pytest.raises(ContractError):
            operator.login("operator", "bad", "7.7.7.7")
    operator.login("operator", PASSWORD, "7.7.7.7")
    for _ in range(2):  # two more failures are not the fifth
        with pytest.raises(ContractError):
            operator.login("operator", "bad", "7.7.7.7")
    assert operator.login("operator", PASSWORD, "7.7.7.7")


def test_failures_spread_over_many_clients_hit_the_global_limit():
    env, operator = make_auth(max_failures=3, global_max_failures=6, window_s=300)
    for i in range(6):
        with pytest.raises(ContractError):
            operator.login("operator", "bad", f"10.0.0.{i}")
    with pytest.raises(ContractError) as error:
        operator.login("operator", PASSWORD, "10.9.9.9")  # a new client, the right password, still refused
    assert code(error) == ("TOO_MANY_ATTEMPTS", 429)
    env.clock.advance(301)
    assert operator.login("operator", PASSWORD, "10.9.9.9")


def test_failures_outside_the_window_do_not_add_up():
    env, operator = make_auth(max_failures=3, window_s=60)
    for _ in range(2):
        with pytest.raises(ContractError):
            operator.login("operator", "bad", "6.6.6.6")
    env.clock.advance(61)
    for _ in range(2):
        with pytest.raises(ContractError):
            operator.login("operator", "bad", "6.6.6.6")
    assert operator.login("operator", PASSWORD, "6.6.6.6")


def test_a_session_ends_at_its_hard_lifetime_even_when_it_is_used_constantly():
    env, operator = make_auth(ttl_s=4000, idle_ttl_s=1800)
    token, web = operator.login("operator", PASSWORD, "1.1.1.1")
    for _ in range(3):
        env.clock.advance(1200)
        assert operator.session(token) is not None  # activity keeps the idle timer fresh (3600 s in)
    env.clock.advance(1200)  # 4800 s: past the 4000 s hard limit, although it was used a moment ago
    assert operator.session(token) is None and operator.session(token) is None


def test_an_idle_session_expires():
    env, operator = make_auth(ttl_s=86400, idle_ttl_s=600)
    token, _ = operator.login("operator", PASSWORD, "1.1.1.1")
    env.clock.advance(601)
    assert operator.session(token) is None


def test_logout_deletes_the_session_on_the_server_so_a_copied_cookie_stops_working():
    env, operator = make_auth()
    token, _ = operator.login("operator", PASSWORD, "1.1.1.1")
    operator.logout(token)
    assert operator.session(token) is None and env.storage.tables.query(auth.T_WEB_SESSIONS, auth.WEB_PK) == []
    operator.logout(token)
    operator.logout(None)  # harmless


def test_logging_out_one_session_leaves_another_alone_and_two_logins_do_not_share_a_token():
    env, operator = make_auth()
    a, _ = operator.login("operator", PASSWORD, "1.1.1.1")
    b, _ = operator.login("operator", PASSWORD, "2.2.2.2")
    assert a != b
    operator.logout(a)
    assert operator.session(a) is None and operator.session(b) is not None


@pytest.mark.parametrize("token", [None, "", "x" * 300, "not-a-real-token", "\x00"])
def test_garbage_session_tokens_never_authenticate(token):
    env, operator = make_auth()
    operator.login("operator", PASSWORD, "1.1.1.1")
    assert operator.session(token) is None


def test_the_csrf_token_must_match_exactly():
    env, operator = make_auth()
    _, web = operator.login("operator", PASSWORD, "1.1.1.1")
    assert check_csrf(web, web.csrf_token)
    for bad in (None, "", "x", web.csrf_token + "x", web.csrf_token[:-1], web.csrf_token.upper()):
        assert not check_csrf(web, bad)


@pytest.mark.parametrize(
    "xff,peer,trusted,expected",
    [(None, "10.0.0.1", 1, "10.0.0.1"), ("1.1.1.1", "10.0.0.1", 0, "10.0.0.1"), ("1.1.1.1, 2.2.2.2", "10.0.0.1", 1, "2.2.2.2"),
     ("evil, 1.1.1.1, 2.2.2.2", "10.0.0.1", 2, "1.1.1.1"), ("1.1.1.1", "10.0.0.1", 3, "10.0.0.1"), ("", "10.0.0.1", 1, "10.0.0.1")],
)  # fmt: skip
def test_only_the_trusted_tail_of_forwarded_for_is_believed(xff, peer, trusted, expected):
    assert client_ip(xff, peer, trusted) == expected  # a spoofed left-hand entry never picks the rate-limit bucket


# ---------------------------------------------------------------------------------------- redaction


@pytest.mark.parametrize(
    "text,secret",
    [("Authorization: Bearer abcdef123456789", "abcdef123456789"), ("authorization=Bearer abcdef123456789", "abcdef123456789"),
     ("got header bearer AbCdEf0123456789xyz here", "AbCdEf0123456789xyz"), ("Cookie: __Host-snn_session=SECRETCOOKIEVALUE123", "SECRETCOOKIEVALUE123"),
     ("Set-Cookie: sid=abc123456789; HttpOnly", "abc123456789"), ('{"password": "hunter2hunter2"}', "hunter2hunter2"),
     ("password=hunter2hunter2&x=1", "hunter2hunter2"), ("api-key: KEYKEYKEY12345", "KEYKEYKEY12345"), ("X-CSRF-Token: csrfvalue123456", "csrfvalue123456"),
     ("device token demo-pi~" + "A" * 43 + " ok", "A" * 43), ("https://x/blob?sv=2024&sig=AbCdEfGhIjKl%2Bmn&se=1", "AbCdEfGhIjKl%2Bmn"),
     ("X-IDENTITY-HEADER: hdr-secret-value", "hdr-secret-value"), ("Authorization: Basic dXNlcjpwYXNzd29yZA==", "dXNlcjpwYXNzd29yZA=="),
     ("Authorization: Token abcdef0123456789", "abcdef0123456789"), ("authorization=Digest response=xyz789", "xyz789")],
)
def test_credentials_are_scrubbed_from_text(text, secret):
    assert secret not in redact(text) and "[redacted" in redact(text)


def test_ordinary_text_is_left_alone():
    text = "session created for boot 70548e48 with 116 batches; event evt-1 is review_required"
    assert redact(text) == text


def test_the_logging_filter_scrubs_messages_arguments_and_exceptions():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(RedactingFilter())
    log = logging.getLogger("redaction-test")
    log.setLevel(logging.INFO)
    log.addHandler(handler)
    try:
        log.info("login for %s with password=%s", "operator", "hunter2hunter2")
        log.info("request headers: %r", {"Authorization": "Bearer SECRETTOKEN123456"})
        try:
            raise RuntimeError("upstream said Authorization: Bearer LEAKEDTOKEN1234567")
        except RuntimeError:
            log.exception("call failed")
    finally:
        log.removeHandler(handler)
    output = stream.getvalue()
    assert "hunter2hunter2" not in output and "SECRETTOKEN123456" not in output and "LEAKEDTOKEN1234567" not in output
    assert "operator" in output and "call failed" in output




def test_every_secret_comparison_is_constant_time(monkeypatch):
    calls = []
    real = auth.hmac.compare_digest
    monkeypatch.setattr(auth.hmac, "compare_digest", lambda a, b: calls.append(1) or real(a, b))
    verify_password(PASSWORD, VERIFIER)
    assert len(calls) == 1
    env, operator = make_auth()
    calls.clear()
    _, web = operator.login("operator", PASSWORD, "1.1.1.1")
    assert len(calls) == 2  # the username and the password verifier
    calls.clear()
    check_csrf(web, web.csrf_token)
    assert len(calls) == 1
    calls.clear()
    token = issue_device_token(env.ctx, "demo-pi")
    authenticate_device(env.ctx, token)
    authenticate_device(env.ctx, "ghost~" + "a" * 43)  # an unknown device still runs a comparison
    assert len(calls) == 2
