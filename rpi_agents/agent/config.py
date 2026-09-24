"""Edge configuration: a TOML file parsed into frozen dataclasses. Standard library only (tomllib).

Strict on purpose: unknown keys and out-of-range values are errors, so a typo cannot silently fall back to
a default on a device that runs unattended. The device credential lives in a separate file that must not
be readable by group or others; it never appears in the config, in logs, or in an error message.
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from rpi_agents.agent.api import LOOPBACK_HOSTS, validate_base_url

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_HASH = re.compile(r"sha256:[0-9a-f]{64}")
_BUILD = re.compile(r"[0-9A-F]{8}")
INPUT_KINDS = ("uno", "stand_in", "replay")
SESSION_MODES = ("demo", "replay", "live")


class ConfigError(ValueError):
    """The configuration is unusable; the message names the offending key."""


@dataclass(frozen=True)
class DeviceConfig:
    id: str
    input_kind: str


@dataclass(frozen=True)
class BackendConfig:
    url: str
    credential_file: str | None = None
    ca_file: str | None = None
    demo_scenario: str | None = None
    timeout_s: float = 5.0


@dataclass(frozen=True)
class SessionConfig:
    mode: str
    model_hash: str
    encoder_hash: str


@dataclass(frozen=True)
class SerialConfig:
    channels: tuple[str, ...]
    path: str | None = None
    replay_file: str | None = None
    baud: int = 115200
    expected_build_id: str | None = None
    stall_s: float = 2.0


@dataclass(frozen=True)
class CameraConfig:
    serial: str | None = None


@dataclass(frozen=True)
class ImagesConfig:
    local_dir: str | None = None  # demo sink only: images stay on the device
    keep: int = 20


@dataclass(frozen=True)
class LimitsConfig:
    max_pending: int = 5000
    pre_session_batches: int = 400
    batch_ms: int = 250
    heartbeat_s: float = 5.0
    command_poll_s: float = 1.0
    drain_s: float = 5.0


@dataclass(frozen=True)
class Config:
    device: DeviceConfig
    backend: BackendConfig
    session: SessionConfig
    serial: SerialConfig
    camera: CameraConfig
    state_dir: str
    limits: LimitsConfig
    images: ImagesConfig = ImagesConfig()


def _section(data: dict, name: str, allowed: set[str], *, required: bool = True) -> dict:
    value = data.get(name)
    if value is None:
        if required:
            raise ConfigError(f"missing section [{name}]")
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"[{name}] must be a table")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigError(f"[{name}] unknown key(s): {', '.join(unknown)}")
    return value


def _get(section: dict, where: str, key: str, kind: type | tuple, *, default: Any = ...) -> Any:
    if key not in section:
        if default is ...:
            raise ConfigError(f"missing {where}.{key}")
        return default
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, kind):
        raise ConfigError(f"{where}.{key} has the wrong type")
    return value


def _optional_text(section: dict, where: str, key: str) -> str | None:
    value = _get(section, where, key, str, default="")
    return value or None


def _bounded(value: float, where: str, low: float, high: float) -> float:
    if not low <= value <= high:
        raise ConfigError(f"{where} must be between {low} and {high}")
    return value


def parse_config(data: dict) -> Config:
    unknown = sorted(set(data) - {"device", "backend", "session", "serial", "camera", "state", "limits", "images"})
    if unknown:
        raise ConfigError(f"unknown section(s): {', '.join(unknown)}")

    dev = _section(data, "device", {"id", "input_kind"})
    device_id = _get(dev, "device", "id", str)
    if not _ID.fullmatch(device_id):
        raise ConfigError("device.id must match [A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
    kind = _get(dev, "device", "input_kind", str)
    if kind not in INPUT_KINDS:
        raise ConfigError(f"device.input_kind must be one of {', '.join(INPUT_KINDS)}")

    be = _section(data, "backend", {"url", "credential_file", "ca_file", "demo_scenario", "timeout_s"})
    try:
        url = validate_base_url(_get(be, "backend", "url", str))
    except ValueError as exc:
        raise ConfigError(f"backend.url: {exc}") from None
    credential = _optional_text(be, "backend", "credential_file")
    if urlsplit(url).hostname not in LOOPBACK_HOSTS and credential is None:
        raise ConfigError("backend.credential_file is required for a backend that is not on loopback")
    backend = BackendConfig(
        url=url,
        credential_file=credential,
        ca_file=_optional_text(be, "backend", "ca_file"),
        demo_scenario=_optional_text(be, "backend", "demo_scenario"),
        timeout_s=_bounded(float(_get(be, "backend", "timeout_s", (int, float), default=5.0)), "backend.timeout_s", 0.5, 60),
    )

    se = _section(data, "session", {"mode", "model_hash", "encoder_hash"})
    mode = _get(se, "session", "mode", str)
    if mode not in SESSION_MODES:
        raise ConfigError(f"session.mode must be one of {', '.join(SESSION_MODES)}")
    hashes = {k: _get(se, "session", k, str) for k in ("model_hash", "encoder_hash")}
    for key, value in hashes.items():
        if not _HASH.fullmatch(value):
            raise ConfigError(f"session.{key} must look like sha256:<64 lowercase hex>")
    session = SessionConfig(mode, hashes["model_hash"], hashes["encoder_hash"])

    sr = _section(
        data, "serial", {"path", "replay_file", "baud", "channels", "expected_build_id", "stall_s"}
    )
    channels = _get(sr, "serial", "channels", list)
    if not channels or not all(isinstance(c, str) and _ID.fullmatch(c) for c in channels):
        raise ConfigError("serial.channels must be a non-empty list of channel names")
    if len(set(channels)) != len(channels):
        raise ConfigError("serial.channels must not contain duplicates")
    path, replay = _optional_text(sr, "serial", "path"), _optional_text(sr, "serial", "replay_file")
    if kind == "replay":
        if replay is None or path is not None:
            raise ConfigError("input_kind 'replay' needs serial.replay_file and no serial.path")
    elif path is None or replay is not None:
        raise ConfigError(f"input_kind '{kind}' needs serial.path and no serial.replay_file")
    if path is not None and not path.startswith("/dev/serial/by-id/"):
        raise ConfigError("serial.path must be a persistent /dev/serial/by-id/... path, not /dev/ttyACM0")
    build = _optional_text(sr, "serial", "expected_build_id")
    if build is not None and not _BUILD.fullmatch(build):
        raise ConfigError("serial.expected_build_id must be 8 uppercase hex characters")
    serial = SerialConfig(
        channels=tuple(channels),
        path=path,
        replay_file=replay,
        baud=int(_bounded(_get(sr, "serial", "baud", int, default=115200), "serial.baud", 1200, 4_000_000)),
        expected_build_id=build,
        stall_s=_bounded(float(_get(sr, "serial", "stall_s", (int, float), default=2.0)), "serial.stall_s", 0.2, 60),
    )

    cam = _section(data, "camera", {"serial"}, required=False)
    camera = CameraConfig(_optional_text(cam, "camera", "serial"))

    st = _section(data, "state", {"dir"})
    state_dir = _get(st, "state", "dir", str)
    if not state_dir:
        raise ConfigError("state.dir must not be empty")

    li = _section(data, "limits", {"max_pending", "pre_session_batches", "batch_ms", "heartbeat_s",
                                    "command_poll_s", "drain_s"}, required=False)  # fmt: skip
    limits = LimitsConfig(
        max_pending=int(_bounded(_get(li, "limits", "max_pending", int, default=5000), "limits.max_pending", 10, 1_000_000)),
        pre_session_batches=int(_bounded(_get(li, "limits", "pre_session_batches", int, default=400), "limits.pre_session_batches", 1, 100_000)),
        batch_ms=int(_bounded(_get(li, "limits", "batch_ms", int, default=250), "limits.batch_ms", 10, 1000)),
        heartbeat_s=_bounded(float(_get(li, "limits", "heartbeat_s", (int, float), default=5.0)), "limits.heartbeat_s", 0.05, 3600),
        command_poll_s=_bounded(float(_get(li, "limits", "command_poll_s", (int, float), default=1.0)), "limits.command_poll_s", 0.05, 3600),
        drain_s=_bounded(float(_get(li, "limits", "drain_s", (int, float), default=5.0)), "limits.drain_s", 0, 300),
    )  # fmt: skip
    im = _section(data, "images", {"local_dir", "keep"}, required=False)
    images = ImagesConfig(
        local_dir=_optional_text(im, "images", "local_dir"),
        keep=int(_bounded(_get(im, "images", "keep", int, default=20), "images.keep", 1, 10_000)),
    )
    return Config(DeviceConfig(device_id, kind), backend, session, serial, camera, state_dir, limits, images)


def load_config(path: str | os.PathLike) -> Config:
    try:
        with open(path, "rb") as handle:
            data = tomllib.load(handle)
    except OSError as exc:
        raise ConfigError(f"cannot read config {path}: {exc.strerror}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: {exc}") from None
    return parse_config(data)


def read_token(config: Config) -> str | None:
    """The bearer token, or None when no credential file is configured. Never logged."""
    if config.backend.credential_file is None:
        return None
    path = Path(config.backend.credential_file)
    try:
        mode = path.stat().st_mode
        if mode & 0o077:
            raise ConfigError(f"credential file {path} must not be readable by group or others (chmod 600)")
        token = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ConfigError(f"cannot read credential file {path}: {exc.strerror}") from None
    if not token or any(c.isspace() for c in token):
        raise ConfigError(f"credential file {path} must contain a single token")
    return token
