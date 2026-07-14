"""State machine: WAKE → CAPTURE → PREFILTER → VISION → DECIDE → RESLEEP (P1–P4).

Orchestrates the full cycle: wake context → capture → prefilter → vision → decision → notifications.
Wires together camera, prefilter, vision, actuators, notifier, and cloud_sync (F03).
"""

import json
import logging
import time
from typing import Optional

import numpy as np

from agent import actuators, camera, cloud_sync, config, notifier, prefilter, sync_queue, vision
from agent.types import Decision, VisionVerdict, WakeContext

logger = logging.getLogger(__name__)


def run_cycle(wake_context: WakeContext) -> Decision:
    """Run one full alarm-decision cycle from wake to decision + side-effects.

    State flow:
      WAKE → CAPTURE (BGR) → PREFILTER → [VISION] → DECIDE →
      ALARM: alarm_on + notify + save_clip  |  FALSE: save_clip only →
      log locally → push to cloud (best-effort) → flush any backlog

    Args:
        wake_context: Captured at wake time; includes trigger status and timestamps.

    Returns:
        Final Decision (alarm: bool, reason: str).
    """
    t_wake = wake_context.ts_monotonic

    # Generated once, here, so the same id is reused on every retry of this
    # event -- immediate or from the sync queue later (ADR-0014).
    event_id = cloud_sync.generate_event_id()

    # CAPTURE — returns BGR (N, H, W, 3)
    frames = camera.capture()

    # PREFILTER — consumes BGR
    pf = prefilter.run(frames)

    # Always save evidence clip (BGR in)
    clip_path = notifier.save_clip(frames)

    verdict: Optional[VisionVerdict] = None
    snapshot: Optional[np.ndarray] = None
    email_sent = False

    if not pf.escalate:
        decision = Decision(
            alarm=False,
            reason=f"prefilter: static scene (score={pf.score:.3f})",
        )
    else:
        # BGR → RGB once; np.ascontiguousarray so cv2 gets a contiguous buffer
        snapshot = np.ascontiguousarray(frames[-1][..., ::-1])

        try:
            verdict = vision.verdict(snapshot)
        except Exception as exc:
            logger.warning("vision.verdict raised (%s); machine-level failsafe → ALARM", exc)
            verdict = VisionVerdict(
                is_intrusion=True,
                confidence=1.0,
                reason=f"machine-failsafe: {type(exc).__name__}",
                source="failsafe",
            )

        if not verdict.is_intrusion:
            decision = Decision(alarm=False, reason=f"vision: {verdict.reason}")
        else:
            # ALARM path — local siren is independent of notify success
            actuators.alarm_on()
            try:
                notifier.notify(verdict.reason, snapshot)
                email_sent = True
            except Exception as exc:
                logger.error("notify failed: %s; local alarm + clip stand", exc)
            decision = Decision(alarm=True, reason=verdict.reason)

    record = _build_record(
        wake_context,
        pf,
        verdict,
        decision,
        clip_path,
        event_id=event_id,
        email_sent=email_sent,
        latency_s=time.monotonic() - t_wake,
    )
    _log_event(record)
    _sync_to_cloud(record, snapshot)
    return decision


def _build_record(
    wake, pf, verdict, decision, clip_path, *, event_id, email_sent, latency_s
) -> dict:
    """Assemble the one event record shared by the local EVENT_LOG line and
    the cloud push payload (cloud_sync.build_payload() picks the subset of
    this dict it needs; "clip" stays local-only, never pushed)."""
    return {
        "event_id": event_id,
        "ts_wall": wake.ts_wall,
        "woken_by_trigger": wake.woken_by_trigger,
        "escalate": pf.escalate,
        "motion": pf.motion,
        "person": pf.person,
        "score": pf.score,
        "vision_source": verdict.source if verdict is not None else None,
        "is_intrusion": verdict.is_intrusion if verdict is not None else None,
        "alarm": decision.alarm,
        "reason": decision.reason,
        "email_sent": email_sent,
        "clip": str(clip_path),
        "latency_s": latency_s,
    }


def _log_event(record: dict) -> None:
    """Append one JSONL line per wake to EVENT_LOG (best-effort; never aborts a cycle)."""
    try:
        config.EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with config.EVENT_LOG.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception as exc:
        logger.warning("event-log write failed: %s", exc)


def _sync_to_cloud(record: dict, snapshot: Optional[np.ndarray]) -> None:
    """Best-effort cloud push + bounded backlog flush (F03; ADR-0014, ADR-0015).

    Never raises into run_cycle() — a failure here must not affect
    power.resleep() (Tenet 1). No-ops entirely when cloud sync is disabled
    or unconfigured (no URL yet) rather than queuing pushes that could
    never succeed.
    """
    if not cloud_sync.is_configured():
        return
    try:
        payload = cloud_sync.build_payload(record, snapshot)
        if not cloud_sync.push(payload):
            sync_queue.enqueue(record["event_id"], payload)
        cloud_sync.flush_queue()
    except Exception as exc:
        logger.warning("cloud sync step failed: %s", exc)
