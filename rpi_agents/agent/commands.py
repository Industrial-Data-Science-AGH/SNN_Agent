"""Command handling for the edge (capture and alarm). Standard library only; hardware is reached only through
the camera and alarm adapters.

Rules this module enforces:
- Never photograph unless the image can be stored: without an ImageSink or camera the command fails first.
- Never actuate unless it is safe: an alarm needs a configured adapter, valid parameters, a matching session
  and a TTL that has not run out at the moment of the apply. Every other command type is answered `failed /
  UNSUPPORTED_COMMAND`. The adapter itself enforces the local time limit, independent of the cloud.
- A command_id is handled at most once, durably (a restart cannot repeat it): duplicates do nothing.
- Expiry is decided on a local monotonic clock fixed when the command is received, so a wall-clock jump
  cannot extend it.
- Every handled command ends in exactly one terminal ack (completed, failed or expired); a command that a
  restart interrupted is failed with AGENT_RESTARTED.
Acks go through the durable outbox, so a network outage never loses one.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Protocol

from rpi_agents.agent.alarm import AlarmUnavailable
from rpi_agents.agent.camera import CameraDisconnected, CameraError, CameraOversize
from rpi_agents.agent.outbox import Outbox
from rpi_agents.agent.ports import AlarmAdapter, CameraAdapter

log = logging.getLogger("snn_edge.commands")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
SCHEMA_VERSION = "1.0"
MAX_TTL_S = 30  # contract: a command lives at most 30 seconds


class SinkUnavailable(RuntimeError):
    """The image could not be stored (no endpoint yet, network, quota)."""


class ImageSink(Protocol):
    def store(self, *, event_id: str, command_id: str, index: int, jpeg: bytes, captured_at: str) -> str:
        """Persist one image and return its image_id. Raise SinkUnavailable when it cannot be stored."""
        ...


@dataclass(frozen=True)
class SessionRef:
    session_id: str
    epoch: int
    mode: str  # session mode: demo, replay or live
    image_bytes: int  # per-image limit granted by the backend
    max_frames: int


def _utc_z(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _parse_utc(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.endswith("Z"):
        return None
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return None


def ack_request_id(command_id: str, status: str) -> str:
    rid = f"ack-{command_id}-{status}"
    return rid if len(rid) <= 64 else f"ack-{hashlib.sha256(command_id.encode()).hexdigest()[:40]}-{status}"


class CommandHandler:
    def __init__(
        self,
        *,
        device_id: str,
        state: Outbox,
        camera: CameraAdapter | None,
        sink: ImageSink | None,
        session: Callable[[], SessionRef | None],
        enqueue_ack: Callable[[dict], None],
        alarm: AlarmAdapter | None = None,
        utc_now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self._device_id, self._state, self._camera, self._sink = device_id, state, camera, sink
        self._alarm_adapter = alarm
        self._session, self._enqueue, self._utc_now, self._mono = session, enqueue_ack, utc_now, monotonic

    def _ack(self, command: dict, status: str, *, error: str | None = None, image_id: str | None = None) -> None:
        self._enqueue(
            {
                "schema_version": SCHEMA_VERSION,
                "request_id": ack_request_id(command["command_id"], status),
                "device_id": command["device_id"],
                "session_id": command["session_id"],
                "epoch": command["epoch"],
                "command_id": command["command_id"],
                "status": status,
                "completed_at": _utc_z(self._utc_now()) if status == "completed" else None,
                "error_code": error,
                "image_id": image_id,
            }
        )

    def _terminal(self, command: dict, status: str, error: str | None, image_id: str | None = None) -> str:
        self._state.set_command_status(command["command_id"], status)
        self._state.kv_delete(f"cmd:{command['command_id']}")
        self._ack(command, status, error=error, image_id=image_id)
        log.info("command %s -> %s%s", command["command_id"], status, f" ({error})" if error else "")
        return status if error is None else f"{status}:{error}"

    def recover(self) -> int:
        """After a restart: fail every command that was accepted but never finished."""
        count = 0
        for command_id in self._state.commands_with_status("accepted"):
            raw = self._state.kv_get(f"cmd:{command_id}")
            if raw is not None:
                self._terminal(json.loads(raw), "failed", "AGENT_RESTARTED")
                count += 1
            else:
                self._state.set_command_status(command_id, "failed")
        return count

    def handle(self, command: object) -> str:
        """Process one command from the backend. Returns a short outcome label (for logs and tests)."""
        if not isinstance(command, dict):
            return "invalid"
        ids = {k: command.get(k) for k in ("command_id", "device_id", "session_id", "epoch")}
        if not all(isinstance(ids[k], str) and _ID.fullmatch(ids[k]) for k in ("command_id", "device_id", "session_id")):
            return "invalid"
        if isinstance(ids["epoch"], bool) or not isinstance(ids["epoch"], int) or ids["epoch"] < 1:
            return "invalid"
        if ids["device_id"] != self._device_id:
            return "ignored"
        command_id = ids["command_id"]
        if self._state.command_status(command_id) is not None:
            return "duplicate"

        now_utc, received = self._utc_now(), self._mono()  # sampled together: the deadline is monotonic
        problem = self._problem(command)
        if problem is not None:
            if not self._state.claim_command(command_id, "failed"):
                return "duplicate"
            return self._terminal(command, "failed", problem)
        expires = _parse_utc(command["expires_at"])
        remaining = (expires - now_utc).total_seconds()
        if remaining <= 0:
            if not self._state.claim_command(command_id, "expired"):
                return "duplicate"
            return self._terminal(command, "expired", "COMMAND_EXPIRED")

        if not self._state.claim_command(command_id, "accepted"):
            return "duplicate"
        self._state.kv_set(f"cmd:{command_id}", json.dumps({k: command[k] for k in ids}))
        self._ack(command, "accepted")
        if command["type"] == "alarm":
            return self._alarm(command, deadline=received + remaining)
        return self._capture(command, deadline=received + remaining)

    def _problem(self, command: dict) -> str | None:
        """An error code when the command must not be executed, else None."""
        kind = command.get("type")
        if kind not in ("capture", "alarm"):
            return "UNSUPPORTED_COMMAND"
        if command.get("schema_version") != SCHEMA_VERSION or not isinstance(command.get("event_id"), str):
            return "INVALID_COMMAND"
        if not _ID.fullmatch(command["event_id"]):
            return "INVALID_COMMAND"
        issued, expires = _parse_utc(command.get("issued_at")), _parse_utc(command.get("expires_at"))
        if issued is None or expires is None or expires <= issued:
            return "INVALID_COMMAND"
        if (expires - issued).total_seconds() > MAX_TTL_S:
            return "INVALID_COMMAND"
        problem = self._capture_parameters(command) if kind == "capture" else self._alarm_parameters(command)
        if problem is not None:
            return problem
        session = self._session()
        if session is None or (session.session_id, session.epoch) != (command["session_id"], command["epoch"]):
            return "SESSION_MISMATCH"
        if command.get("mode") != ("live" if session.mode == "live" else "demo"):
            return "MODE_MISMATCH"
        return self._capture_resources() if kind == "capture" else self._alarm_resources(command["parameters"])

    @staticmethod
    def _capture_parameters(command: dict) -> str | None:
        params = command.get("parameters")
        if not isinstance(params, dict) or set(params) != {"frames", "max_bytes"}:
            return "INVALID_COMMAND"
        frames, size = params["frames"], params["max_bytes"]
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (frames, size)):
            return "INVALID_COMMAND"
        return None if 1 <= frames <= 3 and 1 <= size <= 1_048_576 else "INVALID_COMMAND"

    @staticmethod
    def _alarm_parameters(command: dict) -> str | None:
        params = command.get("parameters")
        policy = command.get("policy_version")
        if not isinstance(policy, str) or not _ID.fullmatch(policy):
            return "INVALID_COMMAND"
        if not isinstance(params, dict) or set(params) != {"duration_ms", "led", "buzzer"}:
            return "INVALID_COMMAND"
        duration = params["duration_ms"]
        if isinstance(duration, bool) or not isinstance(duration, int) or not 1 <= duration <= 30_000:
            return "INVALID_COMMAND"
        if not isinstance(params["led"], bool) or not isinstance(params["buzzer"], bool):
            return "INVALID_COMMAND"
        return None if params["led"] or params["buzzer"] else "INVALID_COMMAND"

    def _capture_resources(self) -> str | None:
        if self._camera is None:
            return "CAMERA_NOT_CONFIGURED"
        if self._sink is None:
            return "IMAGE_SINK_UNAVAILABLE"  # nothing could store the image, so do not take it
        return None

    def _alarm_resources(self, params: dict) -> str | None:
        if self._alarm_adapter is None:
            return "ALARM_NOT_CONFIGURED"
        for name in ("led", "buzzer"):
            if params[name] and not getattr(self._alarm_adapter, f"{name}_available", True):
                return "OUTPUT_NOT_CONFIGURED"  # refuse before energising the outputs that are wired
        return None

    def _alarm(self, command: dict, *, deadline: float) -> str:
        params = command["parameters"]
        if self._mono() >= deadline:  # never switch an alarm on after its command has expired
            return self._terminal(command, "expired", "COMMAND_EXPIRED")
        try:
            self._alarm_adapter.apply(duration_ms=params["duration_ms"], led=params["led"], buzzer=params["buzzer"])
        except AlarmUnavailable:
            return self._terminal(command, "failed", "ALARM_UNAVAILABLE")
        except Exception as exc:  # whatever went wrong, leave the outputs off
            log.exception("alarm command %s crashed: %s", command["command_id"], type(exc).__name__)
            try:
                self._alarm_adapter.off()
            except Exception:
                log.error("alarm off after a failed apply also failed")
            return self._terminal(command, "failed", "ALARM_FAILED")
        return self._terminal(command, "completed", None)

    def _capture(self, command: dict, *, deadline: float) -> str:
        session = self._session()
        params = command["parameters"]
        limit = min(params["max_bytes"], session.image_bytes) if session else params["max_bytes"]
        frames = min(params["frames"], session.max_frames) if session else params["frames"]
        first_image: str | None = None
        try:
            for index in range(frames):
                if self._mono() >= deadline:
                    return self._terminal(command, "expired", "COMMAND_EXPIRED", first_image)
                image = self._camera.capture(max_bytes=limit)
                if self._mono() >= deadline:  # the photo arrived too late to be useful
                    return self._terminal(command, "expired", "COMMAND_EXPIRED", first_image)
                image_id = self._sink.store(
                    event_id=command["event_id"], command_id=command["command_id"], index=index,
                    jpeg=image.jpeg, captured_at=image.captured_at,
                )  # fmt: skip
                first_image = first_image or image_id
        except CameraDisconnected:
            return self._terminal(command, "failed", "CAMERA_DISCONNECTED", first_image)
        except CameraOversize:
            return self._terminal(command, "failed", "IMAGE_TOO_LARGE", first_image)
        except CameraError:
            return self._terminal(command, "failed", "CAMERA_ERROR", first_image)
        except SinkUnavailable:
            return self._terminal(command, "failed", "IMAGE_UPLOAD_FAILED", first_image)
        except Exception as exc:  # a bug must not kill the worker; the command still ends in a terminal ack
            log.exception("command %s crashed: %s", command["command_id"], type(exc).__name__)
            return self._terminal(command, "failed", "INTERNAL_ERROR", first_image)
        return self._terminal(command, "completed", None, first_image)
