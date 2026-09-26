"""Batch ingest: SpikeBatch in, BatchAck out, with the SNN runtime, events and capture commands in between.

Order of operations, chosen so that a crash never makes the stateful runtime step twice:
  1. validate; an exact retry of a finished batch returns the stored ack without running anything;
  2. record the batch as "processing" (create-only, so a concurrent duplicate cannot pass);
  3. step the runtime;
  4. if it triggered: ONE transaction records the event, its capture command and the outbox rows;
  5. finalise the batch record with the ack and advance the session.
If the process dies between 2 and 5 the in-memory runtime is gone with it, and the retry finds a "processing"
row: the session is ended (SESSION_LOST) so the device opens a new epoch instead of continuing on a state
that may have been stepped twice. That is the documented behaviour for a backend restart: an explicit gap,
a new epoch, a new warm-up.

A trigger never raises an alarm here: it only requests a photo. Everything after that is vision and policy.
"""

from __future__ import annotations

import copy
import logging
from datetime import timedelta

from contracts.validation import ContractError, content_hash, validate
from rpi_agents.cloud.app.context import Context
from rpi_agents.cloud.app.publisher import Publisher, outbox_op
from rpi_agents.cloud.app.records import (
    CORE,
    DEVICES_PK,
    T_BATCHES,
    T_DEVICES,
    T_EVENTS,
    T_SESSIONS,
    cmd_rk,
    event_rk,
    list_rk,
    ms,
    seq_rk,
    utc_z,
)
from rpi_agents.cloud.app.sessions import SessionService
from rpi_agents.cloud.app.storage import Op, PreconditionFailed

log = logging.getLogger("snn_backend.ingest")
_CAS_ATTEMPTS = 8
_RUNTIME_STATUSES = ("valid", "warmup", "invalid")
_SCORE_KINDS = ("spike_count", "uncalibrated", "unavailable")
_PROVENANCE = ("demo", "simulated", "measured")


class IngestService:
    def __init__(self, ctx: Context, sessions: SessionService, publisher: Publisher):
        self.ctx, self.sessions, self.publisher = ctx, sessions, publisher

    def ingest(self, device_id: str, session_id: str, body: dict) -> dict:
        validate("SpikeBatch", body, manifest=self.ctx.manifest)
        if (body["device_id"], body["session_id"]) != (device_id, session_id):
            raise ContractError("SESSION_MISMATCH", "Session or device mismatch", 409)
        with self.ctx.lock(f"session:{session_id}"):
            ack = self._ingest_locked(device_id, session_id, body)
        try:
            self.publisher.publish_pending()  # immediate and best effort; the reconciler covers a crash right here
        except Exception:
            log.exception("inline publish failed; the reconciler will retry")
        return ack

    # --------------------------------------------------------------------------- the locked part

    def _ingest_locked(self, device_id: str, session_id: str, body: dict) -> dict:
        ctx, tables = self.ctx, self.ctx.storage.tables
        found = self.sessions.record(device_id, session_id)
        session, seq = found.data, body["batch_seq"]
        if (body["epoch"], body["boot_id"]) != (session["epoch"], session["boot_id"]):
            raise ContractError("SESSION_MISMATCH", "Epoch or boot mismatch: this session belongs to another epoch", 409)

        digest = content_hash(body)
        prior = tables.get(T_BATCHES, session_id, seq_rk(seq))
        if prior is not None:
            if prior.data["hash"] != digest:
                raise ContractError("IDEMPOTENCY_CONFLICT", "Batch sequence reused with different content", 409)
            if prior.data["state"] == "done":
                return copy.deepcopy(prior.data["ack"])
            self.sessions.interrupt(device_id, session_id, "runtime_state_unknown")
            raise ContractError("SESSION_LOST", "The earlier attempt died mid-step; open a new session", 409)

        if session["state"] != "running":
            raise ContractError("SESSION_STOPPED", "Session is stopped", 409)
        runtime = ctx.runtimes.get(session_id)
        if runtime is None:  # this process restarted: the live state is gone
            self.sessions.interrupt(device_id, session_id, "backend_restarted")
            raise ContractError("SESSION_LOST", "The backend lost this session's state; open a new session", 409)

        expected = 0 if session["received_seq"] is None else session["received_seq"] + 1
        if seq < expected or body["source_start_us"] < session["source_time_us"]:
            raise ContractError("OUT_OF_ORDER", "Out-of-order or overlapping batch", 409)
        gaps = self._gaps(session, body, expected)

        now = utc_z(ctx.now())
        processing = tables.insert(T_BATCHES, session_id, seq_rk(seq), {"hash": digest, "state": "processing", "received_at": now})
        try:
            decision = self._decision(session, body, runtime.step(copy.deepcopy(body)), gaps)
        except Exception as exc:
            self.sessions.interrupt(device_id, session_id, "runtime_error")
            log.error("runtime failed on session %s batch %d: %s", session_id, seq, type(exc).__name__)
            raise ContractError("RUNTIME_ERROR", "The runtime failed; the session was ended", 503) from None

        commands: list[dict] = []
        if decision["trigger"] and not gaps:
            event = self._open_event(session, decision)
            if event is not None:
                decision["event_id"] = event["event_id"]
                commands = event["commands"]
            else:  # the cooldown suppressed it; the contract says a trigger always carries an event
                decision["trigger"] = False
        ack = self._ack(session, body, decision, gaps, commands)
        tables.replace(T_BATCHES, session_id, seq_rk(seq), {"hash": digest, "state": "done", "ack": ack, "received_at": now}, processing.etag)
        self._advance(found, seq, body["source_end_us"], now)
        return copy.deepcopy(ack)

    # ---------------------------------------------------------------------------- pieces

    @staticmethod
    def _gaps(session: dict, body: dict, expected: int) -> list[dict]:
        identity = {"schema_version": "1.0", "device_id": session["device_id"], "session_id": session["session_id"], "epoch": session["epoch"]}
        gaps = []
        if body["batch_seq"] != expected or body["source_start_us"] != session["source_time_us"]:
            gaps.append(identity | {"source_start_us": session["source_time_us"], "source_end_us": body["source_start_us"], "reason": "missing_batch"})
        for flag in ("dropped_events", "adc_clipped"):
            if body["quality"][flag]:
                gaps.append(identity | {"source_start_us": body["source_start_us"], "source_end_us": body["source_end_us"], "reason": flag})
        return gaps

    def _decision(self, session: dict, body: dict, raw: object, gaps: list[dict]) -> dict:
        if not isinstance(raw, dict) or not isinstance(raw.get("trigger"), bool) or raw.get("status") not in _RUNTIME_STATUSES:
            raise ValueError("the runtime returned an invalid decision")
        score = raw.get("score")
        if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float))):
            raise ValueError("the runtime returned an invalid score")
        status = "gap" if gaps else raw["status"]
        return {
            "schema_version": "1.0", "device_id": session["device_id"], "session_id": session["session_id"],
            "epoch": session["epoch"], "source_time_us": body["source_end_us"], "model_hash": session["model_hash"],
            "decision_id": self.ctx.new_id(), "event_id": None, "batch_seq": body["batch_seq"],
            "encoder_hash": session["encoder_hash"],
            "trigger": raw["trigger"] and status == "valid",  # no trigger during warm-up, a gap or an invalid step
            "status": status, "score": score,
            "score_kind": raw.get("score_kind") if raw.get("score_kind") in _SCORE_KINDS else "unavailable",
            "provenance": raw.get("provenance") if raw.get("provenance") in _PROVENANCE else "simulated",
        }  # fmt: skip

    def _open_event(self, session: dict, decision: dict) -> dict | None:
        """Record a new event and its capture command in one transaction; None when the cooldown suppresses it."""
        ctx, tables = self.ctx, self.ctx.storage.tables
        now = ctx.now()
        if not self._claim_cooldown(session["device_id"], now):
            return None
        event_id, command_id = ctx.new_id(), ctx.new_id()
        live = session["mode"] == "live"
        command = validate("CaptureCommand", {
            "schema_version": "1.0", "device_id": session["device_id"], "session_id": session["session_id"],
            "epoch": session["epoch"], "command_id": command_id, "event_id": event_id, "type": "capture",
            "mode": "live" if live else "demo", "issued_at": utc_z(now),
            "expires_at": utc_z(now + timedelta(seconds=ctx.settings.capture_ttl_s)),
            "parameters": {"frames": ctx.settings.capture_frames, "max_bytes": ctx.settings.capture_max_bytes},
        })  # fmt: skip
        decision = decision | {"event_id": event_id}
        event = validate("Event", {
            "schema_version": "1.0", "device_id": session["device_id"], "session_id": session["session_id"],
            "epoch": session["epoch"], "event_id": event_id, "status": "photo_requested", "decision": decision,
            "vision": None, "commands": [command], "demo": not live,
        })  # fmt: skip
        meta = {"created_at": utc_z(now), "created_ms": ms(now), "image_ids": [], "notifications": {}, "policy": None, "mode": session["mode"]}
        tables.transaction(T_EVENTS, CORE, [
            Op("insert", event_rk(event_id), {"event": event, "meta": meta}),
            Op("insert", list_rk(ms(now), event_id), {"event_id": event_id}),
            Op("insert", cmd_rk(command_id), {"command": command}),
            outbox_op(ctx, event_id, 0, "index_command", {"device_id": session["device_id"], "command_id": command_id}),
        ])  # fmt: skip
        return event

    def _claim_cooldown(self, device_id: str, now) -> bool:
        """Remember the time of this event on the device record (compare-and-swap); False inside the cooldown."""
        tables = self.ctx.storage.tables
        for _ in range(_CAS_ATTEMPTS):
            device = tables.get(T_DEVICES, DEVICES_PK, device_id)
            last = device.data.get("last_event_ms")
            if last is not None and ms(now) - last < self.ctx.settings.event_cooldown_s * 1000:
                return False
            try:
                tables.replace(T_DEVICES, DEVICES_PK, device_id, device.data | {"last_event_ms": ms(now)}, device.etag)
                return True
            except PreconditionFailed:
                continue
        return False

    def _ack(self, session: dict, body: dict, decision: dict, gaps: list[dict], commands: list[dict]) -> dict:
        status = "gap" if gaps else ("warmup" if decision["status"] == "warmup" else "running")
        ack = {
            "schema_version": "1.0", "device_id": session["device_id"], "session_id": session["session_id"],
            "epoch": session["epoch"], "request_id": body["request_id"], "received_seq": body["batch_seq"],
            "processed_seq": body["batch_seq"], "durable_seq": body["batch_seq"], "status": status,
            "gaps": gaps[:8], "decision": decision, "commands": commands, "demo": session["mode"] != "live",
        }  # fmt: skip
        return validate("BatchAck", ack)

    def _advance(self, found, seq: int, end_us: int, now: str) -> None:
        tables, session_id = self.ctx.storage.tables, found.data["session_id"]
        for _ in range(_CAS_ATTEMPTS):
            current = tables.get(T_SESSIONS, found.data["device_id"], session_id)
            record = current.data | {"received_seq": seq, "processed_seq": seq, "durable_seq": seq, "source_time_us": end_us, "last_seen_at": now}
            try:
                tables.replace(T_SESSIONS, found.data["device_id"], session_id, record, current.etag)
                return
            except PreconditionFailed:
                continue
        raise ContractError("BUSY", "Could not record the batch; retry", 409)
