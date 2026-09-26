"""Table layout, keys and small helpers shared by the backend services. No I/O.

Everything about an event lives in ONE partition ("core") of the `events` table: the event, its list pointer,
the command issuance records and the outbox rows. Azure Table transactions are atomic only within a partition,
and this is what lets "the event, its command and the work to publish them are recorded together or not at
all" be true. Row keys sort so that a prefix query finds what is needed:

    event:<event_id>                     the event (contract Event) plus internal bookkeeping
    list:<reversed ms>:<event_id>        newest-first listing pointer
    cmd:<command_id>                     the command as issued (immutable)
    outbox:<created ms>:<event_id>:<n>   a side effect still to be published; deleted once published
"""

from __future__ import annotations

from datetime import datetime, timezone

T_DEVICES, T_SESSIONS, T_BATCHES = "devices", "sessions", "batches"
T_EVENTS, T_DEVICE_COMMANDS, T_REQUESTS, T_VISION_RUNS = "events", "devicecommands", "requests", "visionruns"
CORE = "core"
DEVICES_PK = "device"
IMAGES = "images"  # blob container
Q_VISION, Q_NOTIFY = "vision-jobs", "notify-jobs"
Q_VISION_POISON, Q_NOTIFY_POISON = "vision-jobs-poison", "notify-jobs-poison"
_MAX_MS = 9_999_999_999_999


def ms(moment: datetime) -> int:
    return int(moment.timestamp() * 1000)


def utc_z(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def event_rk(event_id: str) -> str:
    return f"event:{event_id}"


def list_rk(created_ms: int, event_id: str) -> str:
    return f"list:{_MAX_MS - created_ms:016d}:{event_id}"


def cmd_rk(command_id: str) -> str:
    return f"cmd:{command_id}"


def outbox_rk(created_ms: int, event_id: str, n: int) -> str:
    return f"outbox:{created_ms:016d}:{event_id}:{n}"


def seq_rk(seq: int) -> str:
    return f"{seq:012d}"


def image_name(event_id: str, index: int) -> str:
    return f"{event_id}/{index}.jpg"
