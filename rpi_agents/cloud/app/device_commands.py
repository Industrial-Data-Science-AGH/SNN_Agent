"""Commands as the device sees them: polling and acknowledgement.

A command is issued once (in the event transaction), made visible by the outbox publisher, and then moves
through a small state machine that only ever goes forward:

    issued -> accepted -> completed | failed | expired            (a terminal state is final)

The device may repeat an ack (retry): an exact repeat returns the stored answer. A command that has run out of
time is never handed out and can no longer be accepted; only the device's own `expired`/`failed` report is
taken after that. A `completed` capture must name an image that was really stored for that event.
"""

from __future__ import annotations

import copy

from contracts.validation import ContractError, content_hash, validate
from rpi_agents.cloud.app.context import Context
from rpi_agents.cloud.app.records import CORE, T_DEVICE_COMMANDS, T_EVENTS, T_REQUESTS, event_rk, parse_utc
from rpi_agents.cloud.app.sessions import SessionService
from rpi_agents.cloud.app.storage import Conflict, PreconditionFailed

_TERMINAL = ("completed", "failed", "expired")
_CAS_ATTEMPTS = 8


class DeviceCommandService:
    def __init__(self, ctx: Context, sessions: SessionService):
        self.ctx, self.sessions = ctx, sessions

    def poll(self, device_id: str) -> list[dict]:
        now, items = self.ctx.now(), []
        for row in self.ctx.storage.tables.query(T_DEVICE_COMMANDS, device_id, limit=100):
            command = row.data["command"]
            if row.data["status"] in _TERMINAL or parse_utc(command["expires_at"]) <= now:
                continue
            session = self.ctx.storage.tables.get("sessions", device_id, command["session_id"])
            if session is None or session.data["state"] != "running" or session.data["epoch"] != command["epoch"]:
                continue  # a command for a finished or replaced session must not act on a new one
            items.append(copy.deepcopy(command))
        return items

    def acknowledge(self, device_id: str, command_id: str, body: dict) -> dict:
        validate("CommandAck", body)
        if body["command_id"] != command_id:
            raise ContractError("COMMAND_MISMATCH", "Path and command ID differ", 409)
        if body["device_id"] != device_id:
            raise ContractError("DEVICE_MISMATCH", "Body device differs from the authenticated device", 409)
        with self.ctx.lock(f"commands:{device_id}"):
            key, digest = f"ack:{body['request_id']}", content_hash(body)
            stored = self.ctx.storage.tables.get(T_REQUESTS, device_id, key)
            if stored is not None:
                if stored.data["hash"] != digest:
                    raise ContractError("IDEMPOTENCY_CONFLICT", "Idempotency key reused with different content", 409)
                return copy.deepcopy(stored.data["result"])
            result = self._apply(device_id, command_id, body)
            try:
                self.ctx.storage.tables.insert(T_REQUESTS, device_id, key, {"hash": digest, "result": result})
            except Conflict:
                pass
            return copy.deepcopy(result)

    def _apply(self, device_id: str, command_id: str, body: dict) -> dict:
        tables = self.ctx.storage.tables
        for _ in range(_CAS_ATTEMPTS):
            row = tables.get(T_DEVICE_COMMANDS, device_id, command_id)
            if row is None:
                raise ContractError("NOT_FOUND", "Command not found", 404)
            command = row.data["command"]
            if (body["session_id"], body["epoch"]) != (command["session_id"], command["epoch"]):
                raise ContractError("SESSION_MISMATCH", "Session, device or epoch mismatch", 409)
            status = body["status"]
            if row.data["status"] in _TERMINAL:
                raise ContractError("COMMAND_TERMINAL", "Command is already finished", 409)
            if status == "accepted":
                if parse_utc(command["expires_at"]) <= self.ctx.now():
                    raise ContractError("COMMAND_EXPIRED", "Command expired", 409)
                session = tables.get("sessions", device_id, command["session_id"])
                if session is None or session.data["state"] != "running":
                    raise ContractError("SESSION_STOPPED", "Session is stopped", 409)
            elif status == "completed" and command["type"] == "capture":
                self._require_image(command, body)
            try:
                tables.replace(T_DEVICE_COMMANDS, device_id, command_id,
                               row.data | {"status": status, "acks": row.data["acks"] + [body]}, row.etag)  # fmt: skip
                return copy.deepcopy(body)
            except PreconditionFailed:
                continue
        raise ContractError("BUSY", "Could not record the acknowledgement; retry", 409)

    def _require_image(self, command: dict, body: dict) -> None:
        event = self.ctx.storage.tables.get(T_EVENTS, CORE, event_rk(command["event_id"]))
        known = {img["image_id"] for img in event.data["meta"]["image_ids"]} if event is not None else set()
        if body["image_id"] is None or body["image_id"] not in known:
            raise ContractError("UNKNOWN_IMAGE", "A completed capture must name an image stored for this event", 409)
