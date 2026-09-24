import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from contracts.validation import fixture, validate
from rpi_agents.cloud.app.api import ApiSettings, Services, create_app
from rpi_agents.cloud.app.auth import OperatorAuth, hash_password, issue_device_token
from rpi_agents.cloud.app.sessions import provision_device
from rpi_agents.cloud.app.status import StatusService
from tests.w0.backend_env import TRIGGER, Env
from tests.w0.fakes import make_jpeg

PASSWORD = "correct horse battery staple"
VERIFIER = hash_password(PASSWORD, log2_n=14)
BASE = "https://testserver"


class Rig:
    def __init__(self, api=None, **settings):
        self.env = Env(**settings)
        env = self.env
        self.services = Services(env.sessions, env.ingest, env.commands, env.images, env.events, StatusService(env.ctx), env.ctx)
        self.operator = OperatorAuth(env.ctx, username="operator", password_hash=VERIFIER, max_failures=3)
        self.app = create_app(self.services, self.operator, api or ApiSettings(trusted_proxies=0))
        self.client = TestClient(self.app, base_url=BASE)
        self.token = issue_device_token(env.ctx, "demo-pi")

    def dev(self, key=None, token=None):
        headers = {"Authorization": f"Bearer {token or self.token}"}
        return headers | ({"Idempotency-Key": key} if key else {})

    def post(self, path, body, key=None, token=None):
        return self.client.post(path, json=body, headers=self.dev(key or body.get("request_id"), token))

    def open(self):
        r = self.post("/v1/sessions", self.env.create_body())
        assert r.status_code == 201, r.text
        return r.json()

    def login(self):
        r = self.client.post("/auth/login", json={"username": "operator", "password": PASSWORD})
        assert r.status_code == 200, r.text
        return r.json()["csrf_token"]


@pytest.fixture
def rig():
    return Rig()


def error_code(response):
    return response.json()["error"]["code"]


# ------------------------------------------------------------------------------- cross-cutting


def test_every_response_carries_the_security_headers_and_nothing_secret_leaks_from_health(rig):
    r = rig.client.get("/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok", "schema_version": "1.0"}
    for response in (r, rig.client.get("/v1/events"), rig.client.get("/no/such/route")):
        h = response.headers
        assert h["cache-control"] == "no-store" and h["x-content-type-options"] == "nosniff"
        assert h["x-frame-options"] == "DENY" and h["referrer-policy"] == "no-referrer"
        assert "default-src 'none'" in h["content-security-policy"] and "max-age" in h["strict-transport-security"]


def test_interactive_docs_are_off_by_default_and_available_only_when_asked_for():
    assert [Rig().client.get(p).status_code for p in ("/docs", "/redoc", "/openapi.json")] == [404, 404, 404]
    assert Rig(ApiSettings(trusted_proxies=0, enable_docs=True)).client.get("/openapi.json").status_code == 200


def test_only_the_configured_hosts_are_served():
    rig = Rig(ApiSettings(trusted_proxies=0, allowed_hosts=("api.example.com",)))
    assert rig.client.get("/auth/session").status_code == 400  # the test client's host is not on the list
    assert rig.client.get("/v1/events").status_code == 400 and rig.client.post("/auth/login", json={}).status_code == 400
    good = TestClient(rig.app, base_url="https://api.example.com")
    assert good.get("/auth/session").status_code == 401  # served: the host is fine, only the sign-in is missing
    assert TestClient(rig.app, base_url="https://api.example.com:8443").get("/auth/session").status_code == 401


def test_the_platform_health_probe_is_answered_whatever_host_it_uses():
    rig = Rig(ApiSettings(trusted_proxies=0, allowed_hosts=("api.example.com",)))
    for base in ("http://10.244.1.7:8000", "http://100.64.100.2:8000", "https://api.example.com"):
        r = TestClient(rig.app, base_url=base).get("/healthz")
        assert r.status_code == 200 and r.json() == {"status": "ok", "schema_version": "1.0"}
    assert rig.client.get("/healthz/../v1/events").status_code in (400, 404)  # the exemption is for that one path only


def test_an_unexpected_error_reveals_nothing_about_the_server(rig):
    state = rig.open()
    rig.services.ingest.ingest = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("secret internal path /etc/x"))
    r = rig.post(f"/v1/sessions/{state['session_id']}/batches", rig.env.batch(state, 0))
    assert r.status_code == 500 and "secret" not in r.text and error_code(r) == "INTERNAL"


# -------------------------------------------------------------------------------------- login


def test_login_sets_a_hardened_cookie_and_returns_only_the_csrf_token(rig):
    r = rig.client.post("/auth/login", json={"username": "operator", "password": PASSWORD})
    assert r.status_code == 200 and set(r.json()) == {"schema_version", "csrf_token", "expires_at", "actor"} and r.json()["actor"] == "shared_operator"
    cookie = r.headers["set-cookie"]
    assert cookie.startswith("__Host-snn_session=") and all(a in cookie for a in ("Secure", "HttpOnly", "SameSite=strict", "Path=/"))
    assert "Domain" not in cookie
    token = cookie.split("=", 1)[1].split(";", 1)[0]
    assert token not in r.text  # the session token is only ever in the cookie
    assert rig.client.get("/auth/session").json()["authenticated"] is True


def test_the_development_mode_uses_a_plain_cookie_and_no_hsts():
    rig = Rig(ApiSettings(trusted_proxies=0, insecure_dev=True))
    r = rig.client.post("/auth/login", json={"username": "operator", "password": PASSWORD})
    assert r.headers["set-cookie"].startswith("snn_session=") and "Secure" not in r.headers["set-cookie"]
    assert "strict-transport-security" not in r.headers


@pytest.mark.parametrize("body", [{"username": "operator", "password": "nope"}, {"username": "intruder", "password": PASSWORD},
                                  {"username": "operator"}, {"password": PASSWORD}, {"username": 5, "password": PASSWORD}, {},
                                  {"username": "operator", "password": "x" * 2000}])  # fmt: skip
def test_bad_credentials_are_one_uniform_401_and_set_no_cookie(rig, body):
    r = rig.client.post("/auth/login", json=body)
    assert (r.status_code, error_code(r)) == (401, "INVALID_CREDENTIALS") and "set-cookie" not in r.headers


def test_repeated_failures_lock_the_client_with_retry_after(rig):
    for _ in range(3):
        rig.client.post("/auth/login", json={"username": "operator", "password": "nope"})
    r = rig.client.post("/auth/login", json={"username": "operator", "password": PASSWORD})
    assert (r.status_code, error_code(r)) == (429, "TOO_MANY_ATTEMPTS") and int(r.headers["retry-after"]) >= 1


def test_login_input_is_parsed_strictly(rig):
    assert rig.client.post("/auth/login", content="username=operator", headers={"content-type": "text/plain"}).status_code == 415
    assert rig.client.post("/auth/login", content='{"username":"a","username":"b","password":"x"}', headers={"content-type": "application/json"}).status_code == 422
    assert rig.client.post("/auth/login", content="[1]", headers={"content-type": "application/json"}).status_code == 422
    assert rig.client.post("/auth/login", content=b"{" + b" " * 70000 + b"}", headers={"content-type": "application/json"}).status_code == 413


def test_logout_needs_the_csrf_token_and_a_same_origin_request_and_ends_the_session_for_good(rig):
    csrf = rig.login()
    assert rig.client.post("/auth/logout").status_code == 403  # no CSRF token
    assert rig.client.post("/auth/logout", headers={"X-CSRF-Token": "wrong"}).status_code == 403
    assert rig.client.post("/auth/logout", headers={"X-CSRF-Token": csrf, "Origin": "https://evil.example"}).status_code == 403
    assert rig.client.get("/auth/session").status_code == 200  # still signed in after the refused attempts
    stolen = rig.client.cookies.get("__Host-snn_session")
    assert rig.client.post("/auth/logout", headers={"X-CSRF-Token": csrf, "Origin": BASE}).status_code == 204
    assert rig.client.get("/auth/session").status_code == 401
    other = TestClient(rig.app, base_url=BASE)
    other.cookies.set("__Host-snn_session", stolen)
    assert other.get("/v1/events").status_code == 401  # the copied cookie is dead on the server


def test_two_browsers_can_be_signed_in_with_the_same_shared_login(rig):
    rig.login()
    second = TestClient(rig.app, base_url=BASE)
    assert second.post("/auth/login", json={"username": "operator", "password": PASSWORD}).status_code == 200
    assert second.get("/v1/events").status_code == 200 and rig.client.get("/v1/events").status_code == 200


# ----------------------------------------------------------------------------- device credentials


@pytest.mark.parametrize("header", [None, "", "Bearer", "Bearer ", "Basic abc", "Bearer nope", "Bearer demo-pi~" + "a" * 43, "bearer"])
def test_device_routes_reject_missing_or_wrong_credentials_without_detail(rig, header):
    headers = {} if header is None else {"Authorization": header}
    for method, path in (("get", "/v1/devices/demo-pi/commands"), ("post", "/v1/sessions")):
        r = getattr(rig.client, method)(path, headers=headers | {"Idempotency-Key": "x"}, **({"json": {}} if method == "post" else {}))
        assert (r.status_code, error_code(r)) == (401, "UNAUTHORIZED")


def test_a_device_token_cannot_read_operator_data_and_an_operator_cookie_cannot_act_as_a_device(rig):
    assert rig.client.get("/v1/events", headers=rig.dev()).status_code == 401  # bearer is not a login
    rig.login()
    r = rig.client.post("/v1/sessions", json=rig.env.create_body(), headers={"Idempotency-Key": "create-1"})
    assert r.status_code == 401  # a cookie is not a device credential


def test_a_device_can_only_reach_its_own_device_paths(rig):
    provision_device(rig.env.ctx, "other-pi")
    other = issue_device_token(rig.env.ctx, "other-pi")
    assert rig.client.get("/v1/devices/demo-pi/commands", headers=rig.dev(token=other)).status_code == 403
    assert rig.client.get("/v1/devices/other-pi/commands", headers=rig.dev()).status_code == 403
    assert rig.client.get("/v1/devices/demo-pi/commands", headers=rig.dev()).status_code == 200


# --------------------------------------------------------------------------------- device flow


def test_idempotency_key_must_match_the_request_id_and_retries_return_the_same_session(rig):
    body = rig.env.create_body()
    assert rig.client.post("/v1/sessions", json=body, headers=rig.dev()).status_code == 422  # no key
    assert rig.client.post("/v1/sessions", json=body, headers=rig.dev("other")).status_code == 422
    first = rig.post("/v1/sessions", body)
    assert first.status_code == 201 and rig.post("/v1/sessions", body).json() == first.json()


def test_another_device_cannot_touch_a_session_or_an_event_it_does_not_own(rig):
    state = rig.open()
    provision_device(rig.env.ctx, "other-pi")
    other = issue_device_token(rig.env.ctx, "other-pi")
    stop = {"schema_version": "1.0", "request_id": "s1", "device_id": "other-pi", "session_id": state["session_id"], "epoch": 1}
    assert rig.post(f"/v1/sessions/{state['session_id']}/stop", stop, token=other).status_code == 404
    batch = rig.env.batch(state, 0, device_id="other-pi")
    assert rig.post(f"/v1/sessions/{state['session_id']}/batches", batch, token=other).status_code in (404, 409)
    assert rig.client.get(f"/v1/sessions/{state['session_id']}", headers=rig.dev(token=other)).status_code == 404
    assert rig.client.get(f"/v1/sessions/{state['session_id']}", headers=rig.dev()).status_code == 200


def test_the_whole_alarm_chain_over_http(rig):
    state = rig.open()
    rig.env.ctx.runtimes[state["session_id"]].decisions = [TRIGGER]
    ack = rig.post(f"/v1/sessions/{state['session_id']}/batches", rig.env.batch(state, 0, spikes=[(1200, "zcr")])).json()
    validate("BatchAck", ack)
    (capture,) = ack["commands"]
    polled = rig.client.get("/v1/devices/demo-pi/commands", headers=rig.dev()).json()["items"]
    assert polled == [capture]
    accepted = rig.env.ack_body(capture, "accepted")
    assert rig.post(f"/v1/commands/{capture['command_id']}/ack", accepted).status_code == 200

    jpeg = make_jpeg(640, 480)
    headers = rig.dev(f"{capture['event_id']}-0") | {"Content-Type": "image/jpeg", "X-Image-Index": "0", "X-Image-Sha256": hashlib.sha256(jpeg).hexdigest(),
                                                   "X-Captured-At": "2026-09-24T12:00:04Z"}  # fmt: skip
    up = rig.client.post(f"/v1/events/{capture['event_id']}/image", content=jpeg, headers=headers)
    assert up.status_code == 200 and up.json()["status"] == "queued"
    assert rig.client.post(f"/v1/events/{capture['event_id']}/image", content=jpeg, headers=headers).json() == up.json()  # a retry
    done = rig.env.ack_body(capture, "completed", image_id=up.json()["image_id"])
    assert rig.post(f"/v1/commands/{capture['command_id']}/ack", done).status_code == 200

    assert rig.client.get(f"/v1/events/{capture['event_id']}").status_code == 401  # the operator is not signed in yet
    for path in (f"/v1/events/{capture['event_id']}/images/0", "/v1/events"):  # photos of the home need the operator login
        assert rig.client.get(path).status_code == 401
        assert rig.client.get(path, headers=rig.dev()).status_code == 401  # and a device credential is not enough
    rig.login()
    event = rig.client.get(f"/v1/events/{capture['event_id']}").json()
    assert event["status"] == "analyzing" and rig.client.get("/v1/events").json()["items"][0]["event_id"] == capture["event_id"]
    image = rig.client.get(f"/v1/events/{capture['event_id']}/images/0")
    assert image.content == jpeg and image.headers["content-type"] == "image/jpeg"
    assert image.headers["x-content-type-options"] == "nosniff" and "sandbox" in image.headers["content-security-policy"]
    assert image.headers["cache-control"] == "private, no-store"
    assert rig.client.get(f"/v1/events/{capture['event_id']}/images/2").status_code == 404
    assert rig.client.get("/v1/events/nope").status_code == 404


@pytest.mark.parametrize(
    "changes,status,code",
    [({"Content-Type": "image/png"}, 415, "CONTENT_TYPE"), ({"X-Image-Index": "x"}, 422, "BAD_IMAGE_INDEX"), ({"X-Image-Index": "5"}, 422, "IDEMPOTENCY_REQUIRED"),
     ({"X-Image-Sha256": "0" * 64}, 422, "HASH_MISMATCH"), ({"X-Captured-At": "yesterday"}, 422, "BAD_TIMESTAMP"), ({"Idempotency-Key": "wrong"}, 422, "IDEMPOTENCY_REQUIRED")],
)  # fmt: skip
def test_image_uploads_are_checked_at_the_http_boundary(rig, changes, status, code):
    state = rig.open()
    rig.env.ctx.runtimes[state["session_id"]].decisions = [TRIGGER]
    (capture,) = rig.post(f"/v1/sessions/{state['session_id']}/batches", rig.env.batch(state, 0, spikes=[(1, "zcr")])).json()["commands"]
    jpeg = make_jpeg()
    headers = rig.dev(f"{capture['event_id']}-0") | {"Content-Type": "image/jpeg", "X-Image-Index": "0", "X-Image-Sha256": hashlib.sha256(jpeg).hexdigest(),
                                                   "X-Captured-At": "2026-09-24T12:00:04Z"} | changes  # fmt: skip
    r = rig.client.post(f"/v1/events/{capture['event_id']}/image", content=jpeg, headers=headers)
    assert (r.status_code, error_code(r)) == (status, code)


def test_oversized_bodies_are_cut_off_while_they_are_read(rig):
    state = rig.open()
    huge = json.dumps(rig.env.batch(state, 0, request_id="big")) + " " * 70000
    r = rig.client.post(f"/v1/sessions/{state['session_id']}/batches", content=huge, headers=rig.dev("big") | {"Content-Type": "application/json"})
    assert (r.status_code, error_code(r)) == (413, "PAYLOAD_TOO_LARGE")
    big = b"\xff\xd8" + b"\x00" * 1_100_000 + b"\xff\xd9"
    r = rig.client.post("/v1/events/e1/image", content=big, headers=rig.dev("e1-0") | {"Content-Type": "image/jpeg", "X-Image-Index": "0", "X-Image-Sha256": "0" * 64, "X-Captured-At": "2026-09-24T12:00:04Z"})
    assert (r.status_code, error_code(r)) == (413, "PAYLOAD_TOO_LARGE")


@pytest.mark.parametrize("content,ctype,status", [("{bad", "application/json", 422), ('{"a":NaN}', "application/json", 422), ('{"a":1,"a":2}', "application/json", 422),
                                                   ("[]", "application/json", 422), ("{}", "text/plain", 415)])  # fmt: skip
def test_json_bodies_are_parsed_strictly(rig, content, ctype, status):
    r = rig.client.post("/v1/sessions", content=content, headers=rig.dev("x") | {"Content-Type": ctype})
    assert r.status_code == status


@pytest.mark.parametrize("bad", ["a b", "x" * 65, "-lead", "semi;colon"])
def test_malformed_ids_in_paths_are_refused_before_any_lookup(rig, bad):
    assert rig.client.get(f"/v1/devices/{bad}/commands", headers=rig.dev()).status_code in (404, 422)
    r = rig.client.post(f"/v1/sessions/{bad}/stop", json={"request_id": "r"}, headers=rig.dev("r"))
    assert r.status_code in (404, 422)


# ------------------------------------------------------------------------------------- status


def test_a_device_reports_its_status_and_only_the_operator_or_that_device_can_read_it(rig):
    status = fixture("device-status")
    assert rig.post("/v1/devices/demo-pi/status", status).status_code == 200
    assert rig.client.get("/v1/devices/demo-pi/status").status_code == 401
    assert rig.client.get("/v1/devices/demo-pi/status", headers=rig.dev()).json()["uptime_s"] == status["uptime_s"]
    rig.login()
    assert rig.client.get("/v1/devices/demo-pi/status").json()["state"] == "running"
    assert rig.client.get("/v1/devices/ghost/status").status_code == 404
    assert rig.post("/v1/devices/other-pi/status", status).status_code == 403  # a device cannot report for another
