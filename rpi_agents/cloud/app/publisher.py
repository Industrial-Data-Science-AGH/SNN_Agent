"""The outbox publisher and reconciler.

Why this exists: Azure has no transaction that spans a table and a queue. So a side effect ("tell the worker to
analyse this image", "make this command visible to the device") is first RECORDED as an outbox row in the same
transaction as the state change that caused it, and only then published. Publishing is idempotent and the row is
deleted only after it succeeded, so a crash between "recorded" and "published" leaves a row that the reconciler
finds and publishes later. Publishing can happen twice (two replicas, a crash after the send but before the
delete); consumers therefore de-duplicate.
"""

from __future__ import annotations

import logging

from rpi_agents.cloud.app.context import Context
from rpi_agents.cloud.app.records import (
    CORE,
    Q_NOTIFY,
    Q_VISION,
    T_DEVICE_COMMANDS,
    T_EVENTS,
    cmd_rk,
    ms,
    outbox_rk,
)
from rpi_agents.cloud.app.storage import Conflict, NotFound, Op

log = logging.getLogger("snn_backend.publisher")


def outbox_op(ctx: Context, event_id: str, n: int, kind: str, payload: dict) -> Op:
    """An insert of one outbox row, to be included in the transaction of the change that needs it."""
    created = ms(ctx.now())
    return Op("insert", outbox_rk(created, event_id, n), {"kind": kind, "payload": payload, "created_ms": created})


class Publisher:
    def __init__(self, ctx: Context):
        self.ctx = ctx

    def pending(self, limit: int = 100):
        return self.ctx.storage.tables.query(T_EVENTS, CORE, rk_prefix="outbox:", limit=limit)

    def publish_pending(self, *, min_age_s: float = 0.0, limit: int = 100) -> int:
        """Publish outbox rows at least `min_age_s` old. Returns how many were published."""
        now_ms, done = ms(self.ctx.now()), 0
        for row in self.pending(limit):
            if now_ms - row.data["created_ms"] < min_age_s * 1000:
                continue  # too young: the inline publisher of the request that wrote it is still on it
            try:
                self._publish(row.data["kind"], row.data["payload"])
            except Exception:
                log.exception("could not publish outbox row %s", row.rk)  # the row stays: the next round retries it
                continue
            try:
                self.ctx.storage.tables.delete(T_EVENTS, CORE, row.rk, row.etag)
            except NotFound:
                pass  # another publisher already finished it
            done += 1
        return done

    def _publish(self, kind: str, payload: dict) -> None:
        tables, queues = self.ctx.storage.tables, self.ctx.storage.queues
        if kind == "index_command":
            issued = tables.get(T_EVENTS, CORE, cmd_rk(payload["command_id"]))
            if issued is None:
                raise NotFound(f"command {payload['command_id']}")
            try:
                tables.insert(T_DEVICE_COMMANDS, payload["device_id"], payload["command_id"],
                              {"command": issued.data["command"], "status": "issued", "acks": []})  # fmt: skip
            except Conflict:
                pass  # already indexed
        elif kind == "vision_job":
            queues.send(Q_VISION, {"event_id": payload["event_id"], "image_id": payload["image_id"]})
        elif kind == "notify_job":
            queues.send(Q_NOTIFY, {"event_id": payload["event_id"], "reason": payload.get("reason", "")})
        else:
            raise ValueError(f"unknown outbox kind {kind!r}")

    def drained(self) -> bool:
        """True when nothing is waiting to be published: the condition before scaling to zero."""
        return not self.pending(1)
