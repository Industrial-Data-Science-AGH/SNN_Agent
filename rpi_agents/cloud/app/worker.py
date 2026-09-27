"""The backend worker: publishes the outbox, analyses images, applies the alarm policy, sends notifications.

Guarantees, each covered by tests:
- Queues deliver at least once, so every job is de-duplicated: a run record per (event, image hash) stops a second
  analysis, and a resolved event or a recorded notification stops the rest.
- A job in flight is kept invisible by renewing its queue message while the (slow) model call runs; if the claim
  is lost anyway, the result is thrown away instead of racing the new owner.
- Retries are bounded. When they are exhausted the message goes to a poison queue, and the EVENT is still
  resolved (to human review), so an outage of the vision provider never leaves an event silently pending.
- The alarm is only ever the policy's decision on a vision result, and it is downgraded to review when the session
  is no longer active, the event is too old, or the alarm cool-down applies.
- The e-mail is a separate job with its own status. Its failure never cancels an alarm, an unknown outcome is
  recorded as such and not resent.
"""

from __future__ import annotations

import copy
import logging
import threading
from datetime import timedelta

from contracts.validation import validate
from rpi_agents.cloud.app.context import Context
from rpi_agents.cloud.app.notify import DeliveryUnknown, Notification, NotificationFailed, Notifier
from rpi_agents.cloud.app.policy import PolicyDecision, evaluate
from rpi_agents.cloud.app.publisher import Publisher, outbox_op
from rpi_agents.cloud.app.records import (
    CORE,
    DEVICES_PK,
    IMAGES,
    Q_NOTIFY,
    Q_NOTIFY_POISON,
    Q_VISION,
    Q_VISION_POISON,
    T_DEVICES,
    T_EVENTS,
    T_SESSIONS,
    T_VISION_RUNS,
    cmd_rk,
    event_rk,
    image_name,
    ms,
    parse_utc,
    utc_z,
)
from rpi_agents.cloud.app.storage import Conflict, NotFound, Op, PreconditionFailed, QueueMessage
from rpi_agents.cloud.app.vision import VisionClient, VisionFailed, VisionUnavailable, build_result

log = logging.getLogger("snn_backend.worker")
_OPEN = ("photo_requested", "analyzing")
_FINAL_NOTIFICATION = ("sent", "failed", "delivery_unknown", "not_configured")
_POISON_PILL_DELIVERIES = 10
_CAS_ATTEMPTS = 8
_PROVENANCE = {"demo": "demo", "replay": "synthetic", "live": "real"}


class ClaimLost(Exception):
    """Another worker received the message while this one was busy: this result must not be applied."""


class Worker:
    def __init__(
        self, ctx: Context, publisher: Publisher, vision: VisionClient, notifier: Notifier | None = None, *,
        renew_interval_s: float | None = None,
    ):  # fmt: skip
        self.ctx, self.publisher, self.vision, self.notifier = ctx, publisher, vision, notifier
        self._renew_s = renew_interval_s or max(1.0, ctx.settings.vision_visibility_s / 3)

    # ---------------------------------------------------------------------------------- loop

    def run_once(self, max_messages: int = 20) -> dict:
        """One round: the reconciler, then the vision queue, then the notification queue."""
        published = self.publisher.publish_pending(min_age_s=self.ctx.settings.publish_grace_s)
        return {
            "published": published,
            "vision": self._drain(Q_VISION, self._handle_vision, max_messages),
            "notify": self._drain(Q_NOTIFY, self._handle_notify, max_messages),
        }

    def run_forever(self, stop: threading.Event, poll_s: float = 1.0) -> None:
        while not stop.is_set():
            try:
                busy = self.run_once()
            except Exception:  # the loop must survive anything; the next round retries
                log.exception("worker round failed")
                busy = {}
            if not any(busy.get(k) for k in ("published", "vision", "notify")):
                stop.wait(poll_s)

    def drained(self) -> bool:
        """Nothing left to publish or process: the condition to confirm before scaling to zero."""
        queues = self.ctx.storage.queues
        return self.publisher.drained() and queues.depth(Q_VISION) == 0 and queues.depth(Q_NOTIFY) == 0

    def _drain(self, queue: str, handler, max_messages: int) -> int:
        handled = 0
        while handled < max_messages:
            got = self.ctx.storage.queues.receive(queue, visibility_s=self.ctx.settings.vision_visibility_s)
            if not got:
                break
            (message,) = got
            handled += 1
            try:
                if message.dequeue_count > _POISON_PILL_DELIVERIES:
                    self._poison(queue, message, "TOO_MANY_DELIVERIES")
                    if queue == Q_VISION:
                        self._resolve_after_poison(message)  # an event must never stay silently pending
                    continue
                handler(message)
            except ClaimLost:
                log.warning("lost the claim on %s message %s", queue, message.id)
            except Exception:
                log.exception("handler crashed on %s message %s; it will be delivered again", queue, message.id)
        return handled

    def _poison(self, queue: str, message: QueueMessage, reason: str) -> None:
        target = Q_VISION_POISON if queue == Q_VISION else Q_NOTIFY_POISON
        self.ctx.storage.queues.send(target, message.body | {"reason": reason, "deliveries": message.dequeue_count})
        self._delete(queue, message)

    def _resolve_after_poison(self, message: QueueMessage) -> None:
        found = self.ctx.storage.tables.get(T_EVENTS, CORE, event_rk(message.body["event_id"]))
        if found is None or found.data["event"]["status"] not in _OPEN:
            return
        event = found.data["event"]
        failure = VisionFailed("WORKER_ERROR")
        result = build_result(identity=event, image_id=message.body.get("image_id"), provenance=_PROVENANCE[found.data["meta"]["mode"]],
                              deployment=self.vision.deployment, prompt_version=self.vision.prompt_version, failure=failure)  # fmt: skip
        self._resolve(event["event_id"], result)

    def _delete(self, queue: str, message: QueueMessage) -> None:
        try:
            self.ctx.storage.queues.delete(queue, message)
        except (NotFound, PreconditionFailed):
            pass  # already gone, or received again elsewhere: either way this worker has nothing left to do

    # --------------------------------------------------------------------------------- vision

    def _handle_vision(self, message: QueueMessage) -> str:
        tables, settings = self.ctx.storage.tables, self.ctx.settings
        event_id, image_id = message.body["event_id"], message.body["image_id"]
        found = tables.get(T_EVENTS, CORE, event_rk(event_id))
        image = None if found is None else next((i for i in found.data["meta"]["image_ids"] if i["image_id"] == image_id), None)
        if found is None or image is None:
            self._delete(Q_VISION, message)
            return "orphan"
        if found.data["event"]["status"] not in _OPEN:
            self._delete(Q_VISION, message)
            return "duplicate"
        if not self._claim_run(event_id, image["sha256"]):
            return "in_progress"  # someone else is on it, or died recently: the message reappears and is judged again

        audit: dict = {}
        try:
            try:
                jpeg = self.ctx.storage.blobs.get(IMAGES, image_name(event_id, image["index"]))
            except NotFound:
                raise VisionFailed("IMAGE_MISSING") from None
            observation = self._analyze(message, jpeg)
            result = build_result(
                identity=found.data["event"], image_id=image_id, provenance=_PROVENANCE[found.data["meta"]["mode"]],
                deployment=self.vision.deployment, prompt_version=self.vision.prompt_version, observation=observation,
            )  # fmt: skip
            audit = {"rationale": observation.rationale, "usage": observation.usage}
        except ClaimLost:
            self._finish_run(event_id, image["sha256"], "retrying")  # the new owner must not wait for this run to go stale
            raise
        except VisionUnavailable as exc:
            if message.dequeue_count < settings.vision_max_attempts:
                self._retry_later(message, exc.retry_after_s or settings.vision_retry_delay_s)
                self._finish_run(event_id, image["sha256"], "retrying")
                return "retry"
            self._poison_keep_going(message, exc.code)
            result = build_result(
                identity=found.data["event"], image_id=image_id, provenance=_PROVENANCE[found.data["meta"]["mode"]],
                deployment=self.vision.deployment, prompt_version=self.vision.prompt_version, failure=exc,
            )  # fmt: skip
        except VisionFailed as exc:
            result = build_result(
                identity=found.data["event"], image_id=image_id, provenance=_PROVENANCE[found.data["meta"]["mode"]],
                deployment=self.vision.deployment, prompt_version=self.vision.prompt_version, failure=exc,
            )  # fmt: skip
        self._resolve(event_id, result)
        self._finish_run(event_id, image["sha256"], "done", audit)
        self._delete(Q_VISION, message)
        return "resolved"

    def _claim_run(self, event_id: str, sha256: str) -> bool:
        tables, now = self.ctx.storage.tables, ms(self.ctx.now())
        row = {"state": "running", "started_ms": now}
        try:
            tables.insert(T_VISION_RUNS, event_id, sha256, row)
            return True
        except Conflict:
            existing = tables.get(T_VISION_RUNS, event_id, sha256)
            if existing is None or existing.data["state"] == "done":
                return False
            stale = now - existing.data["started_ms"] >= 2 * self.ctx.settings.vision_visibility_s * 1000
            if existing.data["state"] == "running" and not stale:
                return False
            try:  # a retry, or a dead worker's run: take it over exactly once
                tables.replace(T_VISION_RUNS, event_id, sha256, row, existing.etag)
                return True
            except PreconditionFailed:
                return False

    def _finish_run(self, event_id: str, sha256: str, state: str, audit: dict | None = None) -> None:
        tables = self.ctx.storage.tables
        for _ in range(_CAS_ATTEMPTS):
            row = tables.get(T_VISION_RUNS, event_id, sha256)
            if row is None:
                return
            try:
                tables.replace(T_VISION_RUNS, event_id, sha256, row.data | {"state": state} | (audit or {}), row.etag)
                return
            except PreconditionFailed:
                continue

    def _retry_later(self, message: QueueMessage, delay_s: float) -> None:
        try:
            self.ctx.storage.queues.renew(Q_VISION, message, max(delay_s, 1.0))
        except (NotFound, PreconditionFailed):
            pass

    def _poison_keep_going(self, message: QueueMessage, code: str) -> None:
        self.ctx.storage.queues.send(Q_VISION_POISON, message.body | {"reason": code, "deliveries": message.dequeue_count})

    def _analyze(self, message: QueueMessage, jpeg: bytes):
        """Run the model call while keeping the queue message invisible; abort if the claim is lost."""
        box: dict = {}

        def run() -> None:
            try:
                box["result"] = self.vision.analyze(jpeg)
            except BaseException as exc:  # handed back to the caller's thread, where it is classified
                box["error"] = exc

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        current = message
        while thread.is_alive():
            thread.join(self._renew_s)
            if thread.is_alive():
                try:
                    current = self.ctx.storage.queues.renew(Q_VISION, current, self.ctx.settings.vision_visibility_s)
                except (NotFound, PreconditionFailed):
                    raise ClaimLost from None
        if "error" in box:
            raise box["error"]
        return box["result"]

    # ---------------------------------------------------------------------- policy and alarm

    def _resolve(self, event_id: str, result: dict) -> None:
        ctx, tables = self.ctx, self.ctx.storage.tables
        claimed: list[bool] = []  # the cool-down is claimed once per resolution, never again by our own retry
        for _ in range(_CAS_ATTEMPTS):
            found = tables.get(T_EVENTS, CORE, event_rk(event_id))
            event, meta = copy.deepcopy(found.data["event"]), copy.deepcopy(found.data["meta"])
            if event["status"] not in _OPEN:
                return  # resolved by a concurrent delivery
            now = ctx.now()
            decision = self._decide(event, meta, result, now, claimed)
            event["status"], event["vision"] = decision.status, validate("VisionResult", result)
            meta["policy"] = {"status": decision.status, "reason": decision.reason, "policy_version": decision.policy_version, "decided_at": utc_z(now)}
            ops, n = [], 2
            if decision.alarm is not None:
                command = self._alarm_command(event, meta, decision, now)
                event["commands"] = event["commands"] + [command]
                ops += [Op("insert", cmd_rk(command["command_id"]), {"command": command}),
                        outbox_op(ctx, event_id, n, "index_command", {"device_id": event["device_id"], "command_id": command["command_id"]})]  # fmt: skip
                n += 1
            if decision.notify:
                ops.append(outbox_op(ctx, event_id, n, "notify_job", {"event_id": event_id, "reason": decision.reason}))
            try:
                tables.transaction(T_EVENTS, CORE, [Op("replace", event_rk(event_id), {"event": validate("Event", event), "meta": meta}, found.etag), *ops])
                break
            except PreconditionFailed:
                continue
        self.publisher.publish_pending()

    def _decide(self, event: dict, meta: dict, result: dict, now, claimed: list[bool]) -> PolicyDecision:
        ctx, settings = self.ctx, self.ctx.settings
        device = ctx.storage.tables.get(T_DEVICES, DEVICES_PK, event["device_id"])
        last = None if device is None else device.data.get("last_alarm_ms")
        # once this resolution has claimed the cool-down, a retry must not mistake its own claim for an earlier alarm
        since = None if last is None or claimed else (ms(now) - last) / 1000
        decision = evaluate(settings.policy_version, trigger=event["decision"]["trigger"], vision=result,
                            seconds_since_last_alarm=since, cooldown_s=settings.alarm_cooldown_s, plan=settings.alarm_plan)  # fmt: skip
        if decision.alarm is None:
            return decision
        reason = None
        if (now - parse_utc(meta["created_at"])).total_seconds() > settings.alarm_max_event_age_s:
            reason = "EVENT_TOO_OLD"
        elif not self._session_active(event):
            reason = "SESSION_NOT_ACTIVE"
        else:
            if not claimed:
                claimed.append(self._claim_alarm(event["device_id"], ms(now)))
            if not claimed[0]:
                reason = "ALARM_COOLDOWN"
        if reason is not None:
            return PolicyDecision("review_required", reason, decision.policy_version, None, True)
        return decision

    def _session_active(self, event: dict) -> bool:
        session = self.ctx.storage.tables.get(T_SESSIONS, event["device_id"], event["session_id"])
        return session is not None and session.data["state"] == "running" and session.data["epoch"] == event["epoch"]

    def _claim_alarm(self, device_id: str, now_ms: int) -> bool:
        """Record this alarm on the device (compare-and-swap) so two events cannot both alarm inside the cool-down."""
        tables = self.ctx.storage.tables
        for _ in range(_CAS_ATTEMPTS):
            device = tables.get(T_DEVICES, DEVICES_PK, device_id)
            last = device.data.get("last_alarm_ms")
            if last is not None and now_ms - last < self.ctx.settings.alarm_cooldown_s * 1000:
                return False
            try:
                tables.replace(T_DEVICES, DEVICES_PK, device_id, device.data | {"last_alarm_ms": now_ms}, device.etag)
                return True
            except PreconditionFailed:
                continue
        return False

    def _alarm_command(self, event: dict, meta: dict, decision: PolicyDecision, now) -> dict:
        plan = decision.alarm
        return validate("AlarmCommand", {
            "schema_version": "1.0", "device_id": event["device_id"], "session_id": event["session_id"],
            "epoch": event["epoch"], "command_id": self.ctx.new_id(), "event_id": event["event_id"], "type": "alarm",
            "mode": "live" if meta["mode"] == "live" else "demo", "issued_at": utc_z(now),
            "expires_at": utc_z(now + timedelta(seconds=self.ctx.settings.alarm_ttl_s)),
            "policy_version": decision.policy_version,
            "parameters": {"duration_ms": plan.duration_ms, "led": plan.led, "buzzer": plan.buzzer},
        })  # fmt: skip

    # ------------------------------------------------------------------------- notification

    def _handle_notify(self, message: QueueMessage) -> str:
        tables, settings = self.ctx.storage.tables, self.ctx.settings
        event_id = message.body["event_id"]
        found = tables.get(T_EVENTS, CORE, event_rk(event_id))
        if found is None:
            self._delete(Q_NOTIFY, message)
            return "orphan"
        email = found.data["meta"]["notifications"].get("email")
        if email is not None and email["status"] in _FINAL_NOTIFICATION:
            self._delete(Q_NOTIFY, message)
            return "duplicate"
        if not settings.recipients or self.notifier is None:
            self._record_notification(event_id, {"status": "not_configured"})
            self._delete(Q_NOTIFY, message)
            return "not_configured"

        event, meta = found.data["event"], found.data["meta"]
        vision = event.get("vision") or {}
        jpeg = self._first_image(event_id, meta)
        notification = Notification(
            event_id=event_id, status=event["status"], reason=(meta.get("policy") or {}).get("reason", message.body.get("reason") or "UNKNOWN"),
            observation=vision.get("observation") if vision.get("status") == "ok" else None, jpeg=jpeg, recipients=settings.recipients,
        )  # fmt: skip
        attempts = message.dequeue_count
        try:
            message_id = self.notifier.send(notification)
        except DeliveryUnknown:
            self._record_notification(event_id, {"status": "delivery_unknown", "attempts": attempts})
            self._delete(Q_NOTIFY, message)
            return "delivery_unknown"  # it may have arrived: never resend on a guess
        except NotificationFailed as exc:
            if exc.retryable and attempts < settings.notify_max_attempts:
                self._record_notification(event_id, {"status": "retrying", "attempts": attempts, "code": exc.code})
                try:
                    self.ctx.storage.queues.renew(Q_NOTIFY, message, max(settings.notify_retry_delay_s, 1.0))
                except (NotFound, PreconditionFailed):
                    pass
                return "retry"
            if exc.retryable:
                self._poison(Q_NOTIFY, message, exc.code)
            else:
                self._delete(Q_NOTIFY, message)
            self._record_notification(event_id, {"status": "failed", "attempts": attempts, "code": exc.code})
            return "failed"
        self._record_notification(event_id, {"status": "sent", "attempts": attempts, "message_id": message_id})
        self._delete(Q_NOTIFY, message)
        return "sent"

    def _first_image(self, event_id: str, meta: dict) -> bytes | None:
        if not meta["image_ids"]:
            return None
        try:
            return self.ctx.storage.blobs.get(IMAGES, image_name(event_id, 0))
        except NotFound:
            return None

    def _record_notification(self, event_id: str, entry: dict) -> None:
        tables = self.ctx.storage.tables
        for _ in range(_CAS_ATTEMPTS):
            found = tables.get(T_EVENTS, CORE, event_rk(event_id))
            meta = copy.deepcopy(found.data["meta"])
            meta["notifications"]["email"] = entry | {"at": utc_z(self.ctx.now())}
            try:
                tables.replace(T_EVENTS, CORE, event_rk(event_id), found.data | {"meta": meta}, found.etag)
                return
            except PreconditionFailed:
                continue
