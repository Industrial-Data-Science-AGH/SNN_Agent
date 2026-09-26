import json
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from rpi_agents.agent.api import (
    ApiClient,
    Backoff,
    Outcome,
    Response,
    TransportError,
    UrllibTransport,
    classify,
    is_local_host,
    validate_base_url,
)


class QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):  # clients that hang up early are expected here
        pass


class Server:
    """Local HTTP server that records requests and answers from a script."""

    def __init__(self):
        self.requests, self.script = [], []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _serve(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length).decode("latin-1") if length else ""
                outer.requests.append((self.command, self.path, dict(self.headers), body))
                status, payload, extra = outer.script.pop(0) if outer.script else (200, {"ok": True}, {})
                raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(raw)))
                for name, value in extra.items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(raw)

            do_GET = do_POST = _serve

            def log_message(self, *args):
                pass

        self.httpd = QuietServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    s = Server()
    yield s
    s.close()


@pytest.mark.parametrize("url", ["https://api.example.com", "https://api.example.com:8443/", "http://127.0.0.1:8000",
                                 "http://localhost", "http://[::1]:9"])  # fmt: skip
def test_valid_base_urls(url):
    assert validate_base_url(url) == url.rstrip("/")


@pytest.mark.parametrize("url", ["http://api.example.com", "ftp://x", "https://u:p@api.example.com", "https://a/?q=1",
                                 "https://a/#f", "api.example.com", "https://"])  # fmt: skip
def test_unsafe_base_urls_are_refused(url):
    with pytest.raises(ValueError):
        validate_base_url(url)


def test_post_sends_json_idempotency_key_and_bearer_token(server):
    tr = UrllibTransport(server.url, token="s3cret")
    r = tr.request("POST", "/v1/x", {"request_id": "r1", "a": 1}, headers={"Idempotency-Key": "r1"})
    assert (r.status, r.body) == (200, {"ok": True})
    method, path, headers, body = server.requests[0]
    assert (method, path, json.loads(body)) == ("POST", "/v1/x", {"request_id": "r1", "a": 1})
    assert headers["Idempotency-Key"] == "r1" and headers["Authorization"] == "Bearer s3cret"
    assert headers["Content-Type"] == "application/json"


def test_http_errors_are_responses_with_parsed_body_and_retry_after(server):
    server.script += [(409, {"error": {"code": "SESSION_ACTIVE"}}, {}), (429, {}, {"Retry-After": "7"}),
                      (503, b"<html>bad gateway</html>", {})]  # fmt: skip
    tr = UrllibTransport(server.url)
    r = tr.request("GET", "/a")
    assert (r.status, r.body["error"]["code"]) == (409, "SESSION_ACTIVE")
    r = tr.request("GET", "/a")
    assert (r.status, r.retry_after_s) == (429, 7.0)
    r = tr.request("GET", "/a")
    assert (r.status, r.body) == (503, None)  # a non-JSON error page must not crash the caller


def test_redirects_are_not_followed_and_the_token_is_not_forwarded(server):
    server.script.append((302, {}, {"Location": "http://127.0.0.1:1/steal"}))
    r = UrllibTransport(server.url, token="s3cret").request("GET", "/a")
    assert r.status == 302 and len(server.requests) == 1


def test_no_token_means_no_authorization_header(server):
    UrllibTransport(server.url).request("GET", "/a")
    assert "Authorization" not in server.requests[0][2]


def test_oversized_response_is_bounded(server):
    server.script.append((200, b'{"x":"' + b"a" * (2 << 20) + b'"}', {}))
    assert UrllibTransport(server.url).request("GET", "/a").body is None
    limit = 1 << 20  # valid JSON, exactly one byte over the limit: must still be refused
    exact = b'{"x":"' + b"a" * (limit + 1 - 8) + b'"}'
    assert len(exact) == limit + 1
    server.script.append((200, exact, {}))
    assert UrllibTransport(server.url).request("GET", "/a").body is None
    server.script.append((200, b'{"x":"' + b"a" * (limit - 8) + b'"}', {}))  # exactly at the limit: accepted
    assert UrllibTransport(server.url).request("GET", "/a").body is not None


def test_connection_failures_and_timeouts_become_transport_errors(server):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()  # nothing listens here any more
    with pytest.raises(TransportError):
        UrllibTransport(f"http://127.0.0.1:{port}").request("GET", "/a", timeout_s=1)
    silent = socket.socket()
    silent.bind(("127.0.0.1", 0))
    silent.listen(1)  # accepts the connection at kernel level but never answers
    try:
        with pytest.raises(TransportError):
            UrllibTransport(f"http://127.0.0.1:{silent.getsockname()[1]}").request("GET", "/a", timeout_s=0.3)
    finally:
        silent.close()


@pytest.mark.parametrize(
    "status,outcome",
    [(200, Outcome.OK), (201, Outcome.OK), (401, Outcome.RETRY), (403, Outcome.RETRY), (408, Outcome.RETRY),
     (429, Outcome.RETRY), (500, Outcome.RETRY), (503, Outcome.RETRY), (400, Outcome.PERMANENT),
     (404, Outcome.PERMANENT), (409, Outcome.PERMANENT), (413, Outcome.PERMANENT), (422, Outcome.PERMANENT)],
)  # fmt: skip
def test_outcome_classification(status, outcome):
    assert classify(Response(status, None)).outcome is outcome


def test_classify_extracts_the_error_code():
    r = classify(Response(409, {"error": {"code": "SESSION_ACTIVE", "message": "x"}}))
    assert (r.outcome, r.code) == (Outcome.PERMANENT, "SESSION_ACTIVE")
    assert classify(Response(409, {"error": "text"})).code is None


def test_client_builds_paths_query_and_idempotency_headers(server):
    api = ApiClient(UrllibTransport(server.url))
    body = {"request_id": "create-1", "device_id": "d"}
    assert api.create_session(body, scenario="silence").outcome is Outcome.OK
    api.post_batch("s/../1", {"request_id": "b1"})
    api.poll_commands("demo-pi")
    paths = [(m, p) for m, p, *_ in server.requests]
    assert paths == [("POST", "/v1/sessions?scenario=silence"), ("POST", "/v1/sessions/s%2F..%2F1/batches"),
                     ("GET", "/v1/devices/demo-pi/commands")]  # fmt: skip
    assert server.requests[0][2]["Idempotency-Key"] == "create-1"
    assert "Idempotency-Key" not in server.requests[2][2]


def test_an_image_is_sent_as_raw_bytes_with_a_stable_idempotency_key_and_its_hash(server):
    api = ApiClient(UrllibTransport(server.url, token="s3cret"))
    jpeg = b"\xff\xd8" + bytes(range(256)) * 4 + b"\xff\xd9"
    server.script.append((200, {"image_id": "evt-0"}, {}))
    result = api.upload_image("evt", 0, jpeg, sha256="ab" * 32, captured_at="2026-09-24T12:00:04Z")
    assert (result.outcome, result.body) == (Outcome.OK, {"image_id": "evt-0"})
    method, path, headers, body = server.requests[0]
    assert (method, path) == ("POST", "/v1/events/evt/image")
    assert headers["Content-Type"] == "image/jpeg" and headers["Idempotency-Key"] == "evt-0"
    assert headers["X-Image-Index"] == "0" and headers["X-Image-Sha256"] == "ab" * 32 and headers["X-Captured-At"] == "2026-09-24T12:00:04Z"
    assert headers["Authorization"] == "Bearer s3cret" and body.encode("latin-1") == jpeg  # the bytes arrive untouched, not JSON-encoded


def test_a_request_cannot_carry_both_a_json_body_and_raw_bytes(server):
    with pytest.raises(ValueError, match="either"):
        UrllibTransport(server.url).request("POST", "/x", {"a": 1}, raw=b"x")


def test_client_maps_a_dead_server_to_retry_not_an_exception():
    class Dead:
        def request(self, *a, **k):
            raise TransportError("ConnectionRefusedError")

    result = ApiClient(Dead()).post_batch("s", {"request_id": "b"})
    assert (result.outcome, result.code, result.status) == (Outcome.RETRY, "TRANSPORT_ERROR", None)


def test_backoff_grows_is_capped_honours_hints_and_resets():
    b = Backoff(base=1.0, cap=8.0, rand=lambda: 1.0)  # no jitter reduction
    assert [b.next_delay() for _ in range(5)] == [1.0, 2.0, 4.0, 8.0, 8.0]
    assert b.next_delay(hint=20.0) == 20.0 and b.next_delay(hint=10_000) == 300.0
    b.reset()
    assert b.next_delay() == 1.0
    assert Backoff(base=2.0, cap=8.0, rand=lambda: 0.0).next_delay() == 1.0  # jitter floor is 50 %


def test_module_imports_without_site_packages():
    subprocess.run([sys.executable, "-S", "-c", "import rpi_agents.agent.api"], check=True)


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "::1", "169.254.129.1", "10.0.0.4", "192.168.1.9", "172.16.3.3", "100.64.100.2", "100.127.255.254", "fd00::1"])
def test_local_http_is_allowed_only_for_local_and_private_addresses(host):
    assert is_local_host(host)
    literal = f"[{host}]" if ":" in host else host
    assert validate_base_url(f"http://{literal}:42356/msi/token", local_http=True).startswith("http://")


@pytest.mark.parametrize("host", ["example.com", "8.8.8.8", "1.1.1.1", "100.63.255.255", "100.128.0.1", "2606:4700::1111", "169.254.169.254.evil.com", "localhost.evil.com", "", None])
def test_local_http_never_opens_plain_http_to_the_internet(host):
    assert not is_local_host(host)
    if host:
        with pytest.raises(ValueError, match="plain http"):
            validate_base_url(f"http://{host}/x", local_http=True)


def test_a_private_address_is_still_refused_without_the_local_http_flag():
    with pytest.raises(ValueError, match="plain http"):
        validate_base_url("http://169.254.129.1:8081/msi/token")
    with pytest.raises(ValueError, match="plain http"):
        UrllibTransport("http://10.0.0.4:8000")
