"""JSON Schema + cross-field validation shared by tests and the demo server."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import rfc8785
from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = "1.0"
MAX_BATCH_BYTES = 65536


class ContractError(ValueError):
    def __init__(self, code: str, message: str, status: int = 422):
        super().__init__(message)
        self.code, self.status = code, status


def content_hash(value: object) -> str:
    """RFC 8785 (JCS), SHA-256. Reject NaN, infinity and unsafe integers."""
    try:
        return "sha256:" + hashlib.sha256(rfc8785.dumps(value)).hexdigest()
    except (ValueError, TypeError, RecursionError) as exc:
        raise ContractError("INVALID_JSON", "Payload is not canonical JSON") from exc


def timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


FORMATS = FormatChecker()


@FORMATS.checks("date-time", raises=(ValueError, TypeError))
def utc_datetime(value: object) -> bool:
    return isinstance(value, str) and value.endswith("Z") and timestamp(value).tzinfo is not None


@lru_cache(maxsize=32)
def schema(name: str) -> dict:
    paths = {p.stem.removesuffix(".schema"): p for p in (ROOT / "v1").glob("*.schema.json")}
    if name not in paths:
        raise ContractError("UNKNOWN_CONTRACT", "Unknown contract name")
    return json.loads(paths[name].read_text())


def fixture(name: str) -> dict:
    index = json.loads((ROOT / "fixtures/index.json").read_text())
    if name not in index:
        raise ContractError("UNKNOWN_FIXTURE", "Unknown fixture name", 404)
    return json.loads((ROOT / "fixtures" / f"{name}.json").read_text())


def require(condition: bool, message: str, code: str = "INVALID_CONTRACT", status: int = 422):
    if not condition:
        raise ContractError(code, message, status)


def validate(name: str, value: dict, *, manifest: dict | None = None) -> dict:
    if isinstance(value, dict) and value.get("schema_version") != SCHEMA_VERSION:
        raise ContractError("VERSION_MISMATCH", "Expected schema_version 1.0", 409)
    content_hash(value)
    errors = list(Draft202012Validator(schema(name), format_checker=FORMATS).iter_errors(value))
    if errors:
        # Never echo submitted values (they could accidentally contain secrets).
        path = "/".join(str(x) for x in errors[0].absolute_path) or "$"
        raise ContractError("INVALID_SCHEMA", f"{name}: invalid field at {path}")
    if name == "EncoderProfile":
        channels = value["channel_map"]
        require(len({x["channel"] for x in channels}) == len(channels), "Duplicate channel name")
        require(
            sorted(x["index"] for x in channels) == list(range(len(channels))),
            "Channel indices must be contiguous and unique from zero",
        )
    elif name == "ModelManifest":
        profile = value["encoder_profile"]
        validate("EncoderProfile", profile)
        require(value["encoder_hash"] == content_hash(profile), "Encoder profile hash mismatch")
        topology = value["topology"]
        neurons = topology["neurons"]
        ids = {n["neuron_id"] for n in neurons}
        require(len(ids) == len(neurons), "Duplicate neuron ID")
        channels = {c["channel"] for c in profile["channel_map"]}
        ports = set()
        for n in neurons:
            require(n["v_reset"] < n["v_threshold"], "Reset must be below threshold")
            require(n["potential_unit"] == value["runtime"]["potential_unit"], "Potential unit mismatch")
        for edge in topology["connections"]:
            require(edge["target_id"] in ids, "Unknown target neuron")
            sources = channels if edge["source_kind"] == "channel" else ids
            require(edge["source_id"] in sources, "Unknown connection source")
            port = (edge["target_id"], edge["target_port"])
            require(port not in ports, "A physical synapse port has one incoming connection")
            ports.add(port)
        paths = [a["path"] for a in value["artifacts"]]
        require(len(set(paths)) == len(paths), "Duplicate artifact path")
        require(
            value["provenance"]["checkpoint_hash"] in [a["sha256"] for a in value["artifacts"]],
            "Checkpoint must be included in artifact manifest",
        )
        if value["status"] == "evaluated_champion":
            require(
                profile["provenance"] != "demo" and bool(neurons), "Demo or empty network is not a champion"
            )
            for key in ("training_run_id", "dataset_manifest_hash", "evaluation_hash", "seed"):
                require(value["provenance"][key] is not None, f"Champion requires {key}")
    elif name == "SpikeBatch":
        start, end = value["source_start_us"], value["source_end_us"]
        require(0 < end - start <= 1000000, "Batch duration must be 1..1000000 us")
        times = [s["dt_us"] for s in value["spikes"]]
        require(times == sorted(times), "Spikes must be time ordered")
        require(all(t < end - start for t in times), "Spike outside half-open batch interval")
        if manifest is not None:
            require(
                value["encoder_hash"] == manifest["encoder_hash"],
                "Encoder hash mismatch",
                "ENCODER_MISMATCH",
                409,
            )
            channels = {c["channel"] for c in manifest["encoder_profile"]["channel_map"]}
            require(
                all(s["channel"] in channels for s in value["spikes"]),
                "Unknown encoder channel",
                "UNKNOWN_CHANNEL",
            )
    elif name in ("CaptureCommand", "AlarmCommand"):
        require(
            timestamp(value["expires_at"]) > timestamp(value["issued_at"]),
            "Command expiry must follow issue time",
        )
        require(
            (timestamp(value["expires_at"]) - timestamp(value["issued_at"])).total_seconds() <= 30,
            "Command TTL exceeds 30 seconds",
        )
        if name == "AlarmCommand":
            require(value["parameters"]["led"] or value["parameters"]["buzzer"], "Alarm has no active output")
    elif name == "VisionResult":
        if value["status"] != "ok":
            require(
                value["glass_visible"] == value["person_visible"] == "unknown",
                "Unavailable vision must remain unknown",
            )
            require(value["error_code"] is not None, "Unavailable vision requires error code")
        else:
            require(
                value["image_id"] is not None and value["model_deployment"] is not None,
                "Successful vision requires image and model provenance",
            )
            require(value["error_code"] is None, "Successful vision must not include an error")
    elif name == "SNNDecision":
        require(
            not value["trigger"] or (value["event_id"] is not None and value["status"] == "valid"),
            "Trigger requires a valid decision and event ID",
        )
    elif name == "NeuronFrame":
        ids = [n["neuron_id"] for n in value["neurons"]]
        require(len(set(ids)) == len(ids), "Duplicate neuron ID")
        if manifest is not None:
            topology = manifest["topology"]
            require(
                value["potential_unit"] == manifest["runtime"]["potential_unit"], "Potential unit mismatch"
            )
            require(value["model_hash"] == content_hash(manifest), "Model hash mismatch")
            require(value["topology_version"] == topology["topology_version"], "Topology version mismatch")
            require(
                set(ids) == {n["neuron_id"] for n in topology["neurons"]},
                "Snapshot must include every neuron",
            )
    elif name == "StreamGap":
        require(value["source_end_us"] >= value["source_start_us"], "Gap end precedes start")
    elif name == "CommandAck":
        if value["status"] == "completed":
            require(value["completed_at"] is not None, "Completed ACK requires UTC timestamp")
        if value["status"] in ("failed", "expired"):
            require(value["error_code"] is not None, "Failed ACK requires error code")
    elif name in ("BatchAck", "Event"):
        validate("SNNDecision", value["decision"])
        for cmd in value["commands"]:
            validate("CaptureCommand" if cmd["type"] == "capture" else "AlarmCommand", cmd)
        if name == "BatchAck":
            for gap in value["gaps"]:
                validate("StreamGap", gap)
        elif value["vision"] is not None:
            validate("VisionResult", value["vision"])
    return value
