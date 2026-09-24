"""Backend configuration from environment variables, and the wiring that turns it into running services.

Strict by design, because this runs unattended:
- Nothing has a default that would weaken security: the operator credentials, the model manifest and the runtime must
  be given, secrets are never printed (`repr` hides them), and a bad value stops the process at start-up.
- Anything that is unsafe outside a demo needs its own explicit flag: in-memory storage (SNN_ALLOW_MEMORY=1, data is
  lost on restart), the stand-in runtime (SNN_ALLOW_DEMO_RUNTIME=1), an emulator connection string (SNN_ALLOW_DEV=1),
  live sessions (SNN_ALLOW_LIVE=1), plain-http cookies (SNN_INSECURE_DEV=1).
- Without a vision provider the backend still runs and says so: every analysis is honestly "unavailable" and every
  event goes to human review.
"""

from __future__ import annotations

import importlib
import json
import re
from dataclasses import dataclass, field
from typing import Mapping

from contracts.validation import ContractError, validate
from rpi_agents.cloud.app.api import ApiSettings, Services
from rpi_agents.cloud.app.auth import OperatorAuth
from rpi_agents.cloud.app.context import Context
from rpi_agents.cloud.app.device_commands import DeviceCommandService
from rpi_agents.cloud.app.events import EventReader
from rpi_agents.cloud.app.images import ImageService
from rpi_agents.cloud.app.ingest import IngestService
from rpi_agents.cloud.app.notify import SmtpNotifier, recipients_from
from rpi_agents.cloud.app.policy import POLICIES, AlarmPlan
from rpi_agents.cloud.app.publisher import Publisher
from rpi_agents.cloud.app.sessions import SessionService
from rpi_agents.cloud.app.settings import Settings
from rpi_agents.cloud.app.status import StatusService
from rpi_agents.cloud.app.storage import Storage, memory_storage
from rpi_agents.cloud.app.vision import (
    FAMILIES,
    FoundryVisionClient,
    ManagedIdentityAuth,
    UnavailableVision,
    api_key_auth,
)
from rpi_agents.cloud.app.worker import Worker

_TRUE = ("1", "true", "yes")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


class BackendConfigError(ValueError):
    """The configuration is unusable; the message names the variable, never a secret."""


@dataclass(frozen=True)
class BackendConfig:
    storage: str
    manifest_path: str
    runtime: str
    operator_username: str
    settings: Settings
    api: ApiSettings
    azure_account: str | None = None
    azure_connection_string: str | None = field(default=None, repr=False)
    operator_password_hash: str = field(default="", repr=False)
    vision_endpoint: str | None = None
    vision_deployment: str | None = None
    vision_api_version: str | None = None
    vision_family: str = "chat"
    vision_api_key: str | None = field(default=None, repr=False)
    demo_vision: str | None = None
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_sender: str | None = None
    smtp_username: str | None = None
    smtp_password: str | None = field(default=None, repr=False)
    smtp_starttls: bool = True
    host: str = "0.0.0.0"
    port: int = 8000


def _flag(env: Mapping[str, str], name: str) -> bool:
    return env.get(name, "").strip().lower() in _TRUE


def _need(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise BackendConfigError(f"{name} must be set")
    return value


def _number(env: Mapping[str, str], name: str, default: float, low: float, high: float, kind=float):
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = kind(raw)
    except ValueError:
        raise BackendConfigError(f"{name} must be a number") from None
    if not low <= value <= high:
        raise BackendConfigError(f"{name} must be between {low} and {high}")
    return value


def from_env(env: Mapping[str, str]) -> BackendConfig:
    storage = _need(env, "SNN_STORAGE")
    if storage not in ("azure", "memory"):
        raise BackendConfigError("SNN_STORAGE must be 'azure' or 'memory'")
    if storage == "memory" and not _flag(env, "SNN_ALLOW_MEMORY"):
        raise BackendConfigError("SNN_STORAGE=memory loses all data on restart; set SNN_ALLOW_MEMORY=1 to accept that")
    account, connection = env.get("SNN_AZURE_ACCOUNT", "").strip() or None, env.get("SNN_AZURE_CONNECTION_STRING", "").strip() or None
    if storage == "azure":
        if (account is None) == (connection is None):
            raise BackendConfigError("set exactly one of SNN_AZURE_ACCOUNT (managed identity) and SNN_AZURE_CONNECTION_STRING (emulator)")
        if connection is not None and not _flag(env, "SNN_ALLOW_DEV"):
            raise BackendConfigError("a connection string is for the local emulator only; set SNN_ALLOW_DEV=1 to use it")

    runtime = _need(env, "SNN_RUNTIME")
    if runtime == "demo" and not _flag(env, "SNN_ALLOW_DEMO_RUNTIME"):
        raise BackendConfigError("SNN_RUNTIME=demo is a stand-in, not an SNN; set SNN_ALLOW_DEMO_RUNTIME=1 to accept that")
    if runtime != "demo" and not re.fullmatch(r"[A-Za-z_][\w.]*:[A-Za-z_]\w*", runtime):
        raise BackendConfigError("SNN_RUNTIME must be 'demo' or 'package.module:factory'")

    policy = env.get("SNN_POLICY", "manual-review-only-v1").strip()
    if policy not in POLICIES:
        raise BackendConfigError(f"SNN_POLICY must be one of {', '.join(POLICIES)}")
    try:
        recipients = recipients_from({"recipients": [r.strip() for r in env.get("SNN_ALERT_RECIPIENTS", "").split(",") if r.strip()]})
        plan = AlarmPlan(
            duration_ms=_number(env, "SNN_ALARM_DURATION_MS", 10_000, 1, 30_000, int),
            led=not _flag(env, "SNN_ALARM_NO_LED"), buzzer=not _flag(env, "SNN_ALARM_NO_BUZZER"),
        )
        settings = Settings(
            policy_version=policy, alarm_plan=plan, recipients=recipients, allow_live=_flag(env, "SNN_ALLOW_LIVE"),
            alarm_cooldown_s=_number(env, "SNN_ALARM_COOLDOWN_S", 60.0, 0, 86_400),
            alarm_ttl_s=_number(env, "SNN_ALARM_TTL_S", 15.0, 1, 30), capture_ttl_s=_number(env, "SNN_CAPTURE_TTL_S", 10.0, 1, 30),
            capture_frames=_number(env, "SNN_CAPTURE_FRAMES", 1, 1, 3, int), event_cooldown_s=_number(env, "SNN_EVENT_COOLDOWN_S", 20.0, 0, 86_400),
            session_lease_s=_number(env, "SNN_SESSION_LEASE_S", 30.0, 1, 3600),
        )  # fmt: skip
    except ValueError as exc:
        raise BackendConfigError(str(exc)) from None

    vision_endpoint, vision_deployment = env.get("SNN_VISION_ENDPOINT", "").strip() or None, env.get("SNN_VISION_DEPLOYMENT", "").strip() or None
    if (vision_endpoint is None) != (vision_deployment is None):
        raise BackendConfigError("SNN_VISION_ENDPOINT and SNN_VISION_DEPLOYMENT must be set together")
    family = env.get("SNN_VISION_FAMILY", "chat").strip() or "chat"
    if family not in FAMILIES:
        raise BackendConfigError(f"SNN_VISION_FAMILY must be one of {', '.join(FAMILIES)}")
    demo_vision = env.get("SNN_DEMO_VISION", "").strip() or None
    if demo_vision is not None:
        if not _flag(env, "SNN_ALLOW_DEMO_VISION"):
            raise BackendConfigError("SNN_DEMO_VISION is a scripted stand-in, not a model; set SNN_ALLOW_DEMO_VISION=1 to accept that")
        if vision_endpoint is not None:
            raise BackendConfigError("SNN_DEMO_VISION and SNN_VISION_ENDPOINT are alternatives: choose one")
        from rpi_agents.cloud.app.demo_runtime import DemoVision

        if demo_vision not in [*DemoVision.ANSWERS, "unavailable"]:
            raise BackendConfigError(f"SNN_DEMO_VISION must be one of {', '.join([*DemoVision.ANSWERS, 'unavailable'])}")
    smtp_host, sender = env.get("SNN_SMTP_HOST", "").strip() or None, env.get("SNN_SMTP_FROM", "").strip() or None
    if smtp_host and not sender:
        raise BackendConfigError("SNN_SMTP_FROM must be set with SNN_SMTP_HOST")

    hosts = tuple(h.strip().lower() for h in env.get("SNN_ALLOWED_HOSTS", "").split(",") if h.strip())
    insecure = _flag(env, "SNN_INSECURE_DEV")
    if not hosts and not insecure:
        raise BackendConfigError("SNN_ALLOWED_HOSTS must list the public host name(s) of the API")
    return BackendConfig(
        storage=storage, manifest_path=_need(env, "SNN_MANIFEST_PATH"), runtime=runtime,
        operator_username=_need(env, "SNN_OPERATOR_USERNAME"), operator_password_hash=_need(env, "SNN_OPERATOR_PASSWORD_HASH"),
        settings=settings,
        api=ApiSettings(trusted_proxies=_number(env, "SNN_TRUSTED_PROXIES", 1, 0, 5, int), allowed_hosts=hosts, insecure_dev=insecure),
        azure_account=account, azure_connection_string=connection, vision_endpoint=vision_endpoint, vision_deployment=vision_deployment,
        vision_api_version=env.get("SNN_VISION_API_VERSION", "").strip() or None, vision_family=family,
        vision_api_key=env.get("SNN_VISION_API_KEY", "").strip() or None, demo_vision=demo_vision, smtp_host=smtp_host,
        smtp_port=_number(env, "SNN_SMTP_PORT", 587, 1, 65535, int), smtp_sender=sender,
        smtp_username=env.get("SNN_SMTP_USER", "").strip() or None, smtp_password=env.get("SNN_SMTP_PASSWORD", "").strip() or None,
        smtp_starttls=not _flag(env, "SNN_SMTP_NO_STARTTLS"), host=env.get("SNN_HOST", "0.0.0.0").strip(),
        port=_number(env, "SNN_PORT", 8000, 1, 65535, int),
    )  # fmt: skip


@dataclass
class Backend:
    ctx: Context
    services: Services
    operator: OperatorAuth
    worker: Worker
    publisher: Publisher
    api_settings: ApiSettings
    config: BackendConfig


def _load_manifest(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as handle:
            return validate("ModelManifest", json.load(handle))
    except OSError as exc:
        raise BackendConfigError(f"cannot read the model manifest: {exc.strerror}") from None
    except (ValueError, ContractError) as exc:
        raise BackendConfigError(f"the model manifest is invalid: {exc}") from None


def _runtime_factory(spec: str):
    if spec == "demo":
        from rpi_agents.cloud.app.demo_runtime import DemoRuntime

        return DemoRuntime
    module_name, _, attr = spec.partition(":")
    try:
        factory = getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as exc:
        raise BackendConfigError(f"cannot import the runtime {spec!r}: {type(exc).__name__}") from None
    if not callable(factory):
        raise BackendConfigError(f"the runtime {spec!r} is not callable")
    return factory


def build_storage(config: BackendConfig) -> Storage:
    if config.storage == "memory":
        return memory_storage()
    from rpi_agents.cloud.app.storage_azure import azure_storage

    return azure_storage(account=config.azure_account, connection_string=config.azure_connection_string)


def build(config: BackendConfig, *, storage: Storage | None = None) -> Backend:
    storage = storage if storage is not None else build_storage(config)
    manifest = _load_manifest(config.manifest_path)
    ctx = Context(storage, config.settings, manifest, _runtime_factory(config.runtime))
    sessions, publisher = SessionService(ctx), Publisher(ctx)
    services = Services(sessions, IngestService(ctx, sessions, publisher), DeviceCommandService(ctx, sessions), ImageService(ctx, publisher),
                        EventReader(ctx), StatusService(ctx), ctx)  # fmt: skip
    try:
        operator = OperatorAuth(ctx, username=config.operator_username, password_hash=config.operator_password_hash)
    except ValueError as exc:
        raise BackendConfigError(f"the operator credentials are unusable: {exc}") from None
    if config.demo_vision:
        from rpi_agents.cloud.app.demo_runtime import DemoVision

        vision = DemoVision(config.demo_vision)
    elif config.vision_endpoint:
        auth = api_key_auth(config.vision_api_key) if config.vision_api_key else ManagedIdentityAuth()
        vision = FoundryVisionClient(config.vision_endpoint, config.vision_deployment, auth=auth, api_version=config.vision_api_version, family=config.vision_family)
    else:
        vision = UnavailableVision()
    notifier = None
    if config.smtp_host:
        notifier = SmtpNotifier(config.smtp_host, config.smtp_port, config.smtp_sender, username=config.smtp_username,
                                password=config.smtp_password, starttls=config.smtp_starttls)  # fmt: skip
    return Backend(ctx, services, operator, Worker(ctx, publisher, vision, notifier), publisher, config.api, config)


__all__ = ["Backend", "BackendConfig", "BackendConfigError", "build", "build_storage", "from_env"]
