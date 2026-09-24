"""Session lifecycle: create, stop, state. Epochs, leases and idempotency.

- One live session per device. The epoch is a fencing token: it comes from a compare-and-swap counter on the
  device record and only ever increases, so anything stamped with an old epoch can be recognised and refused.
- A session that has been silent for longer than `session_lease_s` is abandoned: a new session may replace it,
  and the old one is marked stopped with reason lease_expired. This is how a device that lost its state
  (a wiped SD card, say) recovers instead of being locked out.
- The runtime is created and loaded BEFORE anything is written, so a refused model leaves no trace.
- Retrying an identical create/stop request returns the identical result; the same key with different
  content is a conflict.
"""

from __future__ import annotations

import copy
import re

from contracts.validation import ContractError, content_hash, validate
from rpi_agents.cloud.app.context import Context
from rpi_agents.cloud.app.records import (
    DEVICES_PK,
    T_DEVICES,
    T_REQUESTS,
    T_SESSIONS,
    parse_utc,
    utc_z,
)
from rpi_agents.cloud.app.storage import Conflict, PreconditionFailed

_CAS_ATTEMPTS = 8


def provision_device(ctx: Context, device_id: str, token_hash: str = "") -> None:
    """Register a device so it may open sessions. Idempotent."""
    try:
        ctx.storage.tables.insert(T_DEVICES, DEVICES_PK, device_id, {
            "device_id": device_id, "token_hash": token_hash, "active": True, "epoch": 0, "active_session_id": None,
        })  # fmt: skip
    except Conflict:
        pass


def session_state(record: dict, ctx: Context) -> dict:
    """The contract's SessionState for an internal session record."""
    return {
        "schema_version": "1.0", "device_id": record["device_id"], "session_id": record["session_id"],
        "epoch": record["epoch"], "boot_id": record["boot_id"], "mode": record["mode"],
        "model_hash": record["model_hash"], "encoder_hash": record["encoder_hash"],
        "state": "running" if record["state"] == "running" else "stopped",
        "source_time_us": record["source_time_us"], "received_seq": record["received_seq"],
        "processed_seq": record["processed_seq"], "durable_seq": record["durable_seq"],
        "limits": {"batch_bytes": 65536, "image_bytes": ctx.settings.capture_max_bytes,
                   "max_frames": ctx.settings.capture_frames},
        "demo": record["mode"] != "live",
    }  # fmt: skip


_STABLE_CODE = re.compile(r"[A-Z][A-Z0-9_]{2,47}")


def _reason(exc: Exception) -> str:
    """Why a runtime refused: its own stable code when it has one (RuntimeLoadError.code, e.g. DT_ENCODER_MISMATCH),
    else only the exception type. Never the message: that is prose and may quote model contents."""
    code = getattr(exc, "code", None)
    return code if isinstance(code, str) and _STABLE_CODE.fullmatch(code) else type(exc).__name__


class SessionService:
    def __init__(self, ctx: Context):
        self.ctx = ctx

    # ---------------------------------------------------------------- idempotency

    def _idempotent(self, device_id: str, key: str, body: dict, operation):
        """Run `operation` once per (device, key); an exact retry returns the stored result."""
        tables, digest = self.ctx.storage.tables, content_hash(body)
        stored = tables.get(T_REQUESTS, device_id, key)
        if stored is not None:
            if stored.data["hash"] != digest:
                raise ContractError("IDEMPOTENCY_CONFLICT", "Idempotency key reused with different content", 409)
            return copy.deepcopy(stored.data["result"])
        result = operation()
        try:
            tables.insert(T_REQUESTS, device_id, key, {"hash": digest, "result": result})
        except Conflict:  # a concurrent identical request finished first: return what it stored
            return copy.deepcopy(tables.get(T_REQUESTS, device_id, key).data["result"])
        return copy.deepcopy(result)

    # --------------------------------------------------------------------- create

    def create(self, device_id: str, body: dict) -> dict:
        validate("SessionCreate", body)
        if body["device_id"] != device_id:
            raise ContractError("DEVICE_MISMATCH", "Body device differs from the authenticated device", 409)
        with self.ctx.lock(f"device:{device_id}"):
            return self._idempotent(device_id, f"create:{body['request_id']}", body, lambda: self._create(device_id, body))

    def _create(self, device_id: str, body: dict) -> dict:
        ctx, tables = self.ctx, self.ctx.storage.tables
        if body["model_hash"] != content_hash(ctx.manifest):
            raise ContractError("MODEL_MISMATCH", "Unknown model", 409)
        if body["encoder_hash"] != ctx.manifest["encoder_hash"]:
            raise ContractError("ENCODER_MISMATCH", "Encoder does not match the model", 409)
        if body["mode"] == "live" and not ctx.settings.allow_live:
            raise ContractError("LIVE_DISABLED", "Live sessions are not enabled on this backend", 409)

        runtime = ctx.runtime_factory()
        try:  # a refused model must leave no trace
            runtime.load(ctx.manifest)
        except Exception as exc:
            raise ContractError("RUNTIME_REFUSED", f"Runtime refused the model: {_reason(exc)}", 409) from None

        for _ in range(_CAS_ATTEMPTS):
            device = tables.get(T_DEVICES, DEVICES_PK, device_id)
            if device is None or not device.data["active"]:
                raise ContractError("UNKNOWN_DEVICE", "Device is not provisioned", 403)
            self._retire_previous(device.data)
            epoch, session_id = device.data["epoch"] + 1, ctx.new_id()
            try:
                runtime.reset(epoch=epoch, source_time_us=body["source_start_us"])
            except Exception as exc:
                raise ContractError("RUNTIME_REFUSED", f"Runtime could not start: {_reason(exc)}", 409) from None
            try:
                tables.replace(T_DEVICES, DEVICES_PK, device_id,
                               device.data | {"epoch": epoch, "active_session_id": session_id}, device.etag)  # fmt: skip
            except PreconditionFailed:
                continue  # someone else changed the device record: read it again
            now = utc_z(ctx.now())
            record = {
                "device_id": device_id, "session_id": session_id, "epoch": epoch, "boot_id": body["boot_id"],
                "mode": body["mode"], "model_hash": body["model_hash"], "encoder_hash": body["encoder_hash"],
                "state": "running", "source_start_us": body["source_start_us"], "source_time_us": body["source_start_us"],
                "received_seq": None, "processed_seq": None, "durable_seq": None, "created_at": now,
                "last_seen_at": now, "stopped_at": None, "stop_reason": None,
            }  # fmt: skip
            tables.insert(T_SESSIONS, device_id, session_id, record)
            ctx.runtimes[session_id] = runtime
            return validate("SessionState", session_state(record, ctx))
        raise ContractError("BUSY", "Could not allocate an epoch; retry", 409)

    def _retire_previous(self, device: dict) -> None:
        """Refuse while the previous session is alive; retire it when its lease has run out."""
        session_id = device.get("active_session_id")
        if not session_id:
            return
        found = self.ctx.storage.tables.get(T_SESSIONS, device["device_id"], session_id)
        if found is None or found.data["state"] != "running":
            return  # the pointer is stale (a crash between the two writes): nothing to protect
        silent_s = (self.ctx.now() - parse_utc(found.data["last_seen_at"])).total_seconds()
        if silent_s <= self.ctx.settings.session_lease_s:
            raise ContractError("SESSION_ACTIVE", "Device already has an active session", 409)
        self._mark_stopped(found, "lease_expired")

    # ----------------------------------------------------------------------- stop

    def stop(self, device_id: str, session_id: str, body: dict) -> dict:
        validate("SessionControl", body)
        with self.ctx.lock(f"device:{device_id}"):
            return self._idempotent(device_id, f"stop:{body['request_id']}", body | {"path_session": session_id},
                                    lambda: self._stop(device_id, session_id, body))  # fmt: skip

    def _stop(self, device_id: str, session_id: str, body: dict) -> dict:
        found = self.ctx.storage.tables.get(T_SESSIONS, device_id, session_id)
        if found is None:
            raise ContractError("NOT_FOUND", "Session not found", 404)
        if (body["device_id"], body["session_id"], body["epoch"]) != (device_id, session_id, found.data["epoch"]):
            raise ContractError("SESSION_MISMATCH", "Session, device or epoch mismatch", 409)
        record = self._mark_stopped(found, "stopped_by_device") if found.data["state"] == "running" else found.data
        return validate("SessionState", session_state(record, self.ctx))

    def _mark_stopped(self, found, reason: str) -> dict:
        ctx, tables = self.ctx, self.ctx.storage.tables
        record = found.data | {"state": "stopped", "stopped_at": utc_z(ctx.now()), "stop_reason": reason}
        tables.replace(T_SESSIONS, record["device_id"], record["session_id"], record, found.etag)
        ctx.runtimes.pop(record["session_id"], None)
        for _ in range(_CAS_ATTEMPTS):
            device = tables.get(T_DEVICES, DEVICES_PK, record["device_id"])
            if device is None or device.data.get("active_session_id") != record["session_id"]:
                break
            try:
                tables.replace(T_DEVICES, DEVICES_PK, record["device_id"], device.data | {"active_session_id": None}, device.etag)
                break
            except PreconditionFailed:
                continue
        return record

    def interrupt(self, device_id: str, session_id: str, reason: str) -> None:
        """The runtime state can no longer be trusted (a crash mid-step): end the session so the device starts a
        new epoch instead of continuing on top of a state that might have been stepped twice."""
        found = self.ctx.storage.tables.get(T_SESSIONS, device_id, session_id)
        if found is not None and found.data["state"] == "running":
            self._mark_stopped(found, reason)

    # ---------------------------------------------------------------------- reads

    def get(self, device_id: str, session_id: str) -> dict:
        found = self.ctx.storage.tables.get(T_SESSIONS, device_id, session_id)
        if found is None:
            raise ContractError("NOT_FOUND", "Session not found", 404)
        return validate("SessionState", session_state(found.data, self.ctx))

    def record(self, device_id: str, session_id: str):
        found = self.ctx.storage.tables.get(T_SESSIONS, device_id, session_id)
        if found is None:
            raise ContractError("NOT_FOUND", "Session not found", 404)
        return found
