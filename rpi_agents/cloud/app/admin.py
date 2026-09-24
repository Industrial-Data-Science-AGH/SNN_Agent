"""Administration commands: `python -m rpi_agents.cloud.app.admin <command>`.

  hash-password        read a password (hidden prompt, or one line on stdin) and print the scrypt verifier to store
                       as SNN_OPERATOR_PASSWORD_HASH. The password itself is never printed or stored.
  events               print the newest events as JSON lines: status, policy reason, what the vision model saw and its
                       rationale, token usage, notification result. For operators and for building the dashboard.
  issue-device-token   create or rotate a device credential and print the token ONCE. Only its hash is stored, so a
                       lost token cannot be recovered, only replaced. Put it in the device's private credential file.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import sys

from rpi_agents.cloud.app.auth import hash_password, issue_device_token
from rpi_agents.cloud.app.backend_config import BackendConfigError, build, from_env
from rpi_agents.cloud.app.keyvault import KeyVaultError, resolve
from rpi_agents.cloud.app.records import CORE, T_EVENTS, T_VISION_RUNS, event_rk

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


def _read_password() -> str:
    if sys.stdin.isatty():
        first, again = getpass.getpass("Password: "), getpass.getpass("Repeat: ")
        if first != again:
            raise SystemExit("the two entries differ")
        return first
    return sys.stdin.readline().rstrip("\r\n")


def describe_event(tables, event_id: str) -> dict | None:
    """One event as an operator sees it, including the vision model's own rationale (kept in the run record)."""
    found = tables.get(T_EVENTS, CORE, event_rk(event_id))
    if found is None:
        return None
    event, meta = found.data["event"], found.data["meta"]
    runs = [
        {"state": r.data.get("state"), "rationale": r.data.get("rationale"), "usage": r.data.get("usage")}
        for r in tables.query(T_VISION_RUNS, event_id, limit=10)
    ]
    vision = event.get("vision") or {}
    return {
        "event_id": event_id, "status": event["status"], "created_at": meta["created_at"], "mode": meta["mode"],
        "policy": meta.get("policy"), "trigger_score": event["decision"].get("score"),
        "vision": {k: vision.get(k) for k in ("status", "glass_visible", "person_visible", "image_quality", "observation",
                                              "model_deployment", "prompt_version", "provenance", "error_code")} if vision else None,
        "vision_runs": runs, "notifications": meta.get("notifications"), "commands": [c["type"] for c in event["commands"]],
        "images": len(meta.get("image_ids", [])),
    }  # fmt: skip


def main(argv: list[str] | None = None, environ=None, storage=None) -> int:
    parser = argparse.ArgumentParser(description="SNN Agent backend administration")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hash-password")
    token = sub.add_parser("issue-device-token")
    token.add_argument("device_id")
    listing = sub.add_parser("events")
    listing.add_argument("--limit", type=int, default=5)
    args = parser.parse_args(argv)

    if args.command == "hash-password":
        password = _read_password()
        if len(password) < 12:
            print("refusing a password shorter than 12 characters", file=sys.stderr)
            return 1
        print(hash_password(password))
        return 0

    if args.command == "issue-device-token" and not _ID.fullmatch(args.device_id):
        print("device id must match [A-Za-z0-9][A-Za-z0-9_.-]{0,63}", file=sys.stderr)
        return 1
    try:
        backend = build(from_env(resolve(os.environ if environ is None else environ)), storage=storage)
    except (BackendConfigError, KeyVaultError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    if args.command == "events":
        pointers = backend.ctx.storage.tables.query(T_EVENTS, CORE, rk_prefix="list:", limit=max(1, min(args.limit, 50)))
        for pointer in pointers:  # the pointer keys sort newest first
            print(json.dumps(describe_event(backend.ctx.storage.tables, pointer.data["event_id"]), sort_keys=True))
        return 0
    if backend.config.storage == "memory":
        print("refusing: a device credential stored in memory disappears with this process", file=sys.stderr)
        return 1
    print(issue_device_token(backend.ctx, args.device_id))
    return 0


if __name__ == "__main__":
    sys.exit(main())
