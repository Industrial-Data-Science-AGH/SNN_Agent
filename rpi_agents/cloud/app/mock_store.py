"""Bounded in-memory scripted demo. Not an SNN or a production repository."""

from __future__ import annotations

import copy
import uuid
from datetime import datetime, timedelta, timezone
from threading import RLock

from contracts.validation import content_hash, fixture, require, timestamp, validate

SCENARIOS = ("silence", "spike", "trigger", "vision_unavailable")


def identity(s):
    return {k: s[k] for k in ("device_id", "session_id", "epoch")}


def utc(value):
    return value.isoformat().replace("+00:00", "Z")


class DemoStore:
    """One lock per process, bounded state, exact retries, no durable ACKs."""

    def __init__(self):
        self.lock = RLock()
        self.sessions = {}
        self.events = {}
        self.commands = {}
        self.acks = {}
        self.retries = {}
        self.manifest = validate("ModelManifest", fixture("model-manifest"))

    def mutation(self, scope, body, operation):
        key = (scope, body["device_id"], body["request_id"])
        digest = content_hash(body)
        with self.lock:
            if key in self.retries:
                previous, result = self.retries[key]
                require(
                    previous == digest,
                    "Idempotency key reused with different content",
                    "IDEMPOTENCY_CONFLICT",
                    409,
                )
                return copy.deepcopy(result)
            require(
                len(self.retries) < 1024, "Demo request limit reached; restart mock", "DEMO_CAPACITY", 429
            )
            result = operation()
            self.retries[key] = (digest, copy.deepcopy(result))
            return copy.deepcopy(result)

    def get_session(self, session_id):
        require(session_id in self.sessions, "Session not found", "NOT_FOUND", 404)
        return self.sessions[session_id]

    def match(self, s, body):
        require(
            all(s[k] == body[k] for k in identity(s)),
            "Session, device or epoch mismatch",
            "SESSION_MISMATCH",
            409,
        )
        if "boot_id" in body:
            require(s["boot_id"] == body["boot_id"], "Restart requires a new session", "BOOT_MISMATCH", 409)

    def state(self, s):
        result = {
            k: s[k]
            for k in (
                "schema_version",
                "device_id",
                "session_id",
                "epoch",
                "boot_id",
                "mode",
                "model_hash",
                "encoder_hash",
                "state",
                "source_time_us",
                "received_seq",
                "processed_seq",
                "durable_seq",
                "limits",
                "demo",
            )
        }
        return validate("SessionState", result)

    def create(self, body, scenario):
        require(scenario in SCENARIOS, "Unknown demo scenario", "UNKNOWN_SCENARIO", 422)
        require(body["mode"] in ("demo", "replay"), "Mock refuses live sessions", "DEMO_ONLY", 409)
        require(
            body["model_hash"] == content_hash(self.manifest), "Unknown demo model", "MODEL_MISMATCH", 409
        )
        require(
            body["encoder_hash"] == self.manifest["encoder_hash"], "Encoder mismatch", "ENCODER_MISMATCH", 409
        )

        def operation():
            require(len(self.sessions) < 16, "Demo session limit reached; restart mock", "DEMO_CAPACITY", 429)
            require(
                not any(
                    s["device_id"] == body["device_id"] and s["state"] == "running"
                    for s in self.sessions.values()
                ),
                "Device already has an active session",
                "SESSION_ACTIVE",
                409,
            )
            s = {
                **body,
                "session_id": str(uuid.uuid4()),
                "epoch": 1,
                "state": "running",
                "source_time_us": body["source_start_us"],
                "received_seq": None,
                "processed_seq": None,
                "durable_seq": None,
                "demo": True,
                "scenario": scenario,
                "count": 0,
                "limits": {"batch_bytes": 65536, "image_bytes": 1048576, "max_frames": 3},
            }
            s["frame"] = self.frame(s, False, "warmup")
            self.sessions[s["session_id"]] = s
            return self.state(s)

        # Scenario is part of idempotency content, not a different key namespace.
        return self.mutation("create", body | {"demo_scenario": scenario}, operation)

    def frame(self, s, spiked, status):
        return validate(
            "NeuronFrame",
            {
                "schema_version": "1.0",
                **identity(s),
                "source_time_us": s["source_time_us"],
                "model_hash": s["model_hash"],
                "frame_seq": s["count"],
                "topology_version": self.manifest["topology"]["topology_version"],
                "status": status,
                "provenance": "demo",
                "potential_unit": "a.u.",
                "neurons": [
                    {
                        "neuron_id": n["neuron_id"],
                        "v_mem": 0.8 if spiked else 0,
                        "v_threshold": n["v_threshold"],
                        "v_reset": n["v_reset"],
                        "spiked": spiked,
                    }
                    for n in self.manifest["topology"]["neurons"]
                ],
            },
            manifest=self.manifest,
        )

    def ingest(self, session_id, body):
        with self.lock:
            s = self.get_session(session_id)
            self.match(s, body)
            validate("SpikeBatch", body, manifest=self.manifest)

            def operation():
                require(s["state"] == "running", "Session is stopped", "SESSION_STOPPED", 409)
                require(
                    s["count"] < 256, "Demo batch limit reached; start another session", "DEMO_CAPACITY", 429
                )
                previous_seq = s["received_seq"]
                expected = 0 if previous_seq is None else previous_seq + 1
                require(
                    body["batch_seq"] >= expected and body["source_start_us"] >= s["source_time_us"],
                    "Out-of-order or overlapping batch",
                    "OUT_OF_ORDER",
                    409,
                )
                gaps = []
                if body["batch_seq"] != expected or body["source_start_us"] != s["source_time_us"]:
                    gaps.append(
                        {
                            "schema_version": "1.0",
                            **identity(s),
                            "source_start_us": s["source_time_us"],
                            "source_end_us": body["source_start_us"],
                            "reason": "missing_batch",
                        }
                    )
                for flag, reason in [("dropped_events", "dropped_events"), ("adc_clipped", "adc_clipped")]:
                    if body["quality"][flag]:
                        gaps.append(
                            {
                                "schema_version": "1.0",
                                **identity(s),
                                "source_start_us": body["source_start_us"],
                                "source_end_us": body["source_end_us"],
                                "reason": reason,
                            }
                        )
                trigger = (
                    bool(body["spikes"]) and not gaps and s["scenario"] in ("trigger", "vision_unavailable")
                )
                if trigger:
                    require(
                        len(self.events) < 256, "Demo event limit reached; restart mock", "DEMO_CAPACITY", 429
                    )
                decision = {
                    "schema_version": "1.0",
                    **identity(s),
                    "source_time_us": body["source_end_us"],
                    "model_hash": s["model_hash"],
                    "encoder_hash": s["encoder_hash"],
                    "decision_id": str(uuid.uuid4()),
                    "event_id": str(uuid.uuid4()) if trigger else None,
                    "batch_seq": body["batch_seq"],
                    "trigger": trigger,
                    "status": "gap" if gaps else "valid",
                    "score": None,
                    "score_kind": "unavailable",
                    "provenance": "demo",
                }
                commands = []
                if trigger:
                    event_id = decision["event_id"]
                    if s["mode"] == "demo":
                        now = datetime.now(timezone.utc)
                        command = {
                            "schema_version": "1.0",
                            **identity(s),
                            "event_id": event_id,
                            "command_id": str(uuid.uuid4()),
                            "type": "capture",
                            "mode": "demo",
                            "issued_at": utc(now),
                            "expires_at": utc(now + timedelta(seconds=10)),
                            "parameters": {"frames": 1, "max_bytes": 1048576},
                        }
                        validate("CaptureCommand", command)
                        commands.append(command)
                    vision = None
                    if s["scenario"] == "vision_unavailable":
                        vision = fixture("vision-unavailable") | identity(s) | {"event_id": event_id}
                    event = {
                        "schema_version": "1.0",
                        **identity(s),
                        "event_id": event_id,
                        "status": "review_required" if vision else "photo_requested",
                        "decision": decision,
                        "vision": vision,
                        "commands": commands,
                        "demo": True,
                    }
                    validate("Event", event)
                    self.events[event_id] = event
                    for c in commands:
                        self.commands[c["command_id"]] = c
                s.update(
                    source_time_us=body["source_end_us"],
                    received_seq=body["batch_seq"],
                    processed_seq=body["batch_seq"],
                    count=s["count"] + 1,
                )
                s["frame"] = self.frame(s, bool(body["spikes"]) and not gaps, "gap" if gaps else "running")
                return validate(
                    "BatchAck",
                    {
                        "schema_version": "1.0",
                        **identity(s),
                        "request_id": body["request_id"],
                        "received_seq": body["batch_seq"],
                        "processed_seq": body["batch_seq"],
                        "durable_seq": None,
                        "status": "gap" if gaps else "running",
                        "gaps": gaps,
                        "decision": decision,
                        "commands": commands,
                        "demo": True,
                    },
                )

            return self.mutation(f"batch/{session_id}", body, operation)

    def pending(self, device_id):
        with self.lock:
            now = datetime.now(timezone.utc)
            return copy.deepcopy(
                [
                    c
                    for k, c in self.commands.items()
                    if c["device_id"] == device_id
                    and timestamp(c["expires_at"]) > now
                    and self.sessions[c["session_id"]]["state"] == "running"
                    and self.acks.get(k, {}).get("status") not in ("completed", "failed", "expired")
                ]
            )

    def acknowledge(self, command_id, body):
        with self.lock:
            require(command_id == body["command_id"], "Path and command ID differ", "COMMAND_MISMATCH", 409)
            require(command_id in self.commands, "Command not found", "NOT_FOUND", 404)
            c = self.commands[command_id]
            self.match(c, body)

            def operation():
                previous = self.acks.get(command_id)
                require(
                    not previous or previous["status"] == "accepted",
                    "Command is terminal",
                    "COMMAND_TERMINAL",
                    409,
                )
                if body["status"] == "accepted":
                    require(
                        timestamp(c["expires_at"]) > datetime.now(timezone.utc),
                        "Command expired",
                        "COMMAND_EXPIRED",
                        409,
                    )
                    require(
                        self.get_session(c["session_id"])["state"] == "running",
                        "Session is stopped",
                        "SESSION_STOPPED",
                        409,
                    )
                self.acks[command_id] = copy.deepcopy(body)
                return body

            return self.mutation(f"ack/{command_id}", body, operation)

    def stop(self, session_id, body):
        with self.lock:
            s = self.get_session(session_id)
            self.match(s, body)

            def operation():
                s["state"] = "stopped"
                s["frame"] = self.frame(s, False, "stopped")
                return self.state(s)

            return self.mutation(f"stop/{session_id}", body, operation)
