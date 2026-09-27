"""Scripted smoke client: python -m contracts.demo (mock running locally)."""

import argparse
import json
import uuid
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from contracts.validation import fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    url = urlparse(args.base_url)
    if (
        url.scheme != "http"
        or url.hostname not in ("127.0.0.1", "localhost")
        or url.username
        or url.password
        or url.path not in ("", "/")
        or url.query
        or url.fragment
    ):
        parser.error("The demo client only connects to an HTTP loopback mock")
    base = args.base_url.rstrip("/")

    def request(path, body=None):
        headers = {"Content-Type": "application/json"}
        if body is not None:
            headers["Idempotency-Key"] = body["request_id"]
        r = Request(
            base + path, data=json.dumps(body).encode() if body is not None else None, headers=headers
        )
        with urlopen(r, timeout=5) as response:
            return json.load(response)

    body = fixture("session-create") | {"device_id": f"demo-{uuid.uuid4().hex[:12]}"}
    session = request("/v1/sessions?scenario=vision_unavailable", body)
    ids = {k: session[k] for k in ("device_id", "session_id", "epoch", "boot_id")}
    route = "/v1/sessions/" + session["session_id"]
    for name in ("silence", "spike", "duplicate", "gap"):
        ack = request(route + "/batches", fixture(name) | ids)
        print(
            json.dumps(
                {
                    "fixture": name,
                    "status": ack["status"],
                    "trigger": ack["decision"]["trigger"],
                    "decision_id": ack["decision"]["decision_id"],
                    "durable_seq": ack["durable_seq"],
                    "demo": ack["demo"],
                }
            )
        )
    control = {k: session[k] for k in ("schema_version", "device_id", "session_id", "epoch")} | {
        "request_id": "stop-1"
    }
    print(json.dumps({"session": session["session_id"], "state": request(route + "/stop", control)["state"]}))


if __name__ == "__main__":
    main()
