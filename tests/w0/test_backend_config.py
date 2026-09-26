import io
import json
import types

import pytest
from fastapi.testclient import TestClient

from contracts.validation import fixture
from rpi_agents.cloud.app import admin, backend_config, server
from rpi_agents.cloud.app.api import create_app
from rpi_agents.cloud.app.auth import authenticate_device, hash_password, verify_password
from rpi_agents.cloud.app.backend_config import BackendConfigError, build, from_env
from rpi_agents.cloud.app.notify import SmtpNotifier
from rpi_agents.cloud.app.storage import memory_storage
from rpi_agents.cloud.app.vision import FoundryVisionClient, ManagedIdentityAuth, UnavailableVision

PASSWORD = "correct horse battery staple"
VERIFIER = hash_password(PASSWORD, log2_n=14)


def make_env(tmp_path, **over):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(fixture("model-manifest")))
    env = {
        "SNN_STORAGE": "memory", "SNN_ALLOW_MEMORY": "1", "SNN_MANIFEST_PATH": str(manifest), "SNN_RUNTIME": "demo",
        "SNN_ALLOW_DEMO_RUNTIME": "1", "SNN_OPERATOR_USERNAME": "operator", "SNN_OPERATOR_PASSWORD_HASH": VERIFIER,
        "SNN_ALLOWED_HOSTS": "api.example.com",
    }  # fmt: skip
    for key, value in over.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return env


def test_a_minimal_environment_gives_the_safe_defaults(tmp_path):
    c = from_env(make_env(tmp_path))
    assert (c.storage, c.runtime, c.settings.policy_version, c.settings.recipients) == ("memory", "demo", "manual-review-only-v1", ())
    assert (c.settings.allow_live, c.api.allowed_hosts, c.api.insecure_dev, c.api.trusted_proxies) == (False, ("api.example.com",), False, 1)
    assert c.vision_endpoint is None and c.smtp_host is None and c.port == 8000


def test_secrets_never_appear_in_the_printed_configuration(tmp_path):
    env = make_env(tmp_path, SNN_VISION_ENDPOINT="https://v.example.com", SNN_VISION_DEPLOYMENT="gpt", SNN_VISION_API_KEY="KEY-SECRET-123",
                   SNN_SMTP_HOST="smtp.example.com", SNN_SMTP_FROM="agent@example.com", SNN_SMTP_PASSWORD="SMTP-SECRET-456")  # fmt: skip
    text = repr(from_env(env))
    for secret in (VERIFIER, "KEY-SECRET-123", "SMTP-SECRET-456"):
        assert secret not in text


@pytest.mark.parametrize("name", ["SNN_STORAGE", "SNN_MANIFEST_PATH", "SNN_RUNTIME", "SNN_OPERATOR_USERNAME", "SNN_OPERATOR_PASSWORD_HASH", "SNN_ALLOWED_HOSTS"])
def test_every_required_setting_is_required_and_named_in_the_error(tmp_path, name):
    with pytest.raises(BackendConfigError, match=name):
        from_env(make_env(tmp_path, **{name: None}))
    with pytest.raises(BackendConfigError, match=name):
        from_env(make_env(tmp_path, **{name: "   "}))


@pytest.mark.parametrize(
    "changes,needle",
    [({"SNN_ALLOW_MEMORY": None}, "SNN_ALLOW_MEMORY"), ({"SNN_ALLOW_DEMO_RUNTIME": None}, "SNN_ALLOW_DEMO_RUNTIME"),
     ({"SNN_STORAGE": "s3"}, "SNN_STORAGE"), ({"SNN_STORAGE": "azure"}, "exactly one"),
     ({"SNN_STORAGE": "azure", "SNN_AZURE_ACCOUNT": "acct", "SNN_AZURE_CONNECTION_STRING": "UseDevelopmentStorage=true"}, "exactly one"),
     ({"SNN_STORAGE": "azure", "SNN_AZURE_CONNECTION_STRING": "UseDevelopmentStorage=true"}, "SNN_ALLOW_DEV"),
     ({"SNN_RUNTIME": "not a module"}, "SNN_RUNTIME"), ({"SNN_POLICY": "armed-v9"}, "SNN_POLICY"),
     ({"SNN_ALARM_TTL_S": "99"}, "SNN_ALARM_TTL_S"), ({"SNN_CAPTURE_TTL_S": "soon"}, "number"), ({"SNN_CAPTURE_FRAMES": "7"}, "SNN_CAPTURE_FRAMES"),
     ({"SNN_ALARM_DURATION_MS": "999999"}, "SNN_ALARM_DURATION_MS"), ({"SNN_ALERT_RECIPIENTS": "not-an-address"}, "address"),
     ({"SNN_VISION_ENDPOINT": "https://v.example.com"}, "together"), ({"SNN_VISION_DEPLOYMENT": "gpt"}, "together"),
     ({"SNN_SMTP_HOST": "smtp.example.com"}, "SNN_SMTP_FROM"), ({"SNN_PORT": "0"}, "SNN_PORT"), ({"SNN_TRUSTED_PROXIES": "50"}, "SNN_TRUSTED_PROXIES")],
)  # fmt: skip
def test_unsafe_or_invalid_settings_stop_the_process_at_start_up(tmp_path, changes, needle):
    with pytest.raises(BackendConfigError, match=needle):
        from_env(make_env(tmp_path, **changes))


def test_unsafe_options_are_available_only_through_their_own_explicit_flag(tmp_path):
    assert from_env(make_env(tmp_path, SNN_ALLOW_LIVE="1")).settings.allow_live is True
    dev = from_env(make_env(tmp_path, SNN_STORAGE="azure", SNN_AZURE_CONNECTION_STRING="UseDevelopmentStorage=true", SNN_ALLOW_DEV="1", SNN_ALLOW_MEMORY=None))
    assert dev.azure_connection_string == "UseDevelopmentStorage=true"
    assert from_env(make_env(tmp_path, SNN_STORAGE="azure", SNN_AZURE_ACCOUNT="acct", SNN_ALLOW_MEMORY=None)).azure_account == "acct"
    assert from_env(make_env(tmp_path, SNN_INSECURE_DEV="1", SNN_ALLOWED_HOSTS=None)).api.insecure_dev is True  # plain http, no host list


def test_recipients_and_the_alarm_plan_come_from_configuration(tmp_path):
    c = from_env(make_env(tmp_path, SNN_ALERT_RECIPIENTS="a@example.com, b@example.org", SNN_ALARM_DURATION_MS="3000", SNN_ALARM_NO_BUZZER="1", SNN_POLICY="armed-glass-and-person-v1"))
    assert c.settings.recipients == ("a@example.com", "b@example.org")
    assert (c.settings.alarm_plan.duration_ms, c.settings.alarm_plan.led, c.settings.alarm_plan.buzzer) == (3000, True, False)
    with pytest.raises(BackendConfigError):
        from_env(make_env(tmp_path, SNN_ALARM_NO_LED="1", SNN_ALARM_NO_BUZZER="1"))  # an alarm with no output is not an alarm


# ---------------------------------------------------------------------------------------------- building


def test_building_wires_honest_defaults_without_a_vision_provider_or_a_mailbox(tmp_path):
    backend = build(from_env(make_env(tmp_path)))
    assert isinstance(backend.worker.vision, UnavailableVision) and backend.worker.notifier is None
    assert backend.ctx.settings.policy_version == "manual-review-only-v1"


def test_the_scripted_demo_vision_needs_its_own_flag_and_excludes_a_real_provider(tmp_path):
    with pytest.raises(BackendConfigError, match="SNN_ALLOW_DEMO_VISION"):
        from_env(make_env(tmp_path, SNN_DEMO_VISION="glass_person"))
    with pytest.raises(BackendConfigError, match="alternatives"):
        from_env(make_env(tmp_path, SNN_DEMO_VISION="glass_person", SNN_ALLOW_DEMO_VISION="1", SNN_VISION_ENDPOINT="https://v.example.com", SNN_VISION_DEPLOYMENT="gpt"))
    with pytest.raises(BackendConfigError, match="SNN_DEMO_VISION must be one of"):
        from_env(make_env(tmp_path, SNN_DEMO_VISION="everything", SNN_ALLOW_DEMO_VISION="1"))
    backend = build(from_env(make_env(tmp_path, SNN_DEMO_VISION="glass_person", SNN_ALLOW_DEMO_VISION="1")))
    assert backend.worker.vision.deployment == "demo-vision"
    result = backend.worker.vision.analyze(b"any bytes")
    assert (result.glass_visible, result.person_visible) == (True, True)


def test_a_configured_vision_provider_and_mailbox_are_wired(tmp_path):
    env = make_env(tmp_path, SNN_VISION_ENDPOINT="https://v.example.com", SNN_VISION_DEPLOYMENT="gpt-vision-1", SNN_SMTP_HOST="smtp.example.com",
                   SNN_SMTP_FROM="agent@example.com", SNN_SMTP_USER="agent", SNN_SMTP_PASSWORD="pw")  # fmt: skip
    backend = build(from_env(env))
    assert isinstance(backend.worker.vision, FoundryVisionClient) and backend.worker.vision.deployment == "gpt-vision-1"
    assert isinstance(backend.worker.vision._auth, ManagedIdentityAuth)  # no key configured: the managed identity
    assert isinstance(backend.worker.notifier, SmtpNotifier)
    keyed = build(from_env(env | {"SNN_VISION_API_KEY": "k"}))
    assert not isinstance(keyed.worker.vision._auth, ManagedIdentityAuth)


@pytest.mark.parametrize("bad", ["missing", "broken", "wrong-schema"])
def test_a_missing_or_invalid_model_manifest_stops_the_start(tmp_path, bad):
    env = make_env(tmp_path)
    path = tmp_path / "bad.json"
    if bad == "broken":
        path.write_text("{not json")
    elif bad == "wrong-schema":
        path.write_text(json.dumps({"schema_version": "1.0"}))
    env["SNN_MANIFEST_PATH"] = str(path)
    with pytest.raises(BackendConfigError, match="manifest"):
        build(from_env(env))


@pytest.mark.parametrize("spec,needle", [("no.such.module:factory", "cannot import"), ("os.path:no_such_attr", "cannot import"), ("os:name", "not callable")])
def test_a_runtime_that_cannot_be_imported_or_called_stops_the_start(tmp_path, spec, needle):
    with pytest.raises(BackendConfigError, match=needle):
        build(from_env(make_env(tmp_path, SNN_RUNTIME=spec)))


def test_a_real_runtime_is_loaded_from_its_dotted_path(tmp_path):
    backend = build(from_env(make_env(tmp_path, SNN_RUNTIME="rpi_agents.cloud.app.demo_runtime:DemoRuntime", SNN_ALLOW_DEMO_RUNTIME=None)))
    assert type(backend.ctx.runtime_factory()).__name__ == "DemoRuntime"


@pytest.mark.parametrize("verifier", ["plaintext", "scrypt$10$8$1$AAAAAAAAAAAAAAAA$" + "A" * 43 + "="])
def test_a_weak_or_malformed_operator_verifier_stops_the_start_without_echoing_it(tmp_path, verifier):
    with pytest.raises(BackendConfigError, match="operator credentials") as error:
        build(from_env(make_env(tmp_path, SNN_OPERATOR_PASSWORD_HASH=verifier)))
    assert verifier not in str(error.value)


def test_environment_variables_boot_a_working_application(tmp_path):
    backend = build(from_env(make_env(tmp_path)))
    client = TestClient(create_app(backend.services, backend.operator, backend.api_settings), base_url="https://api.example.com")
    assert client.get("/healthz").status_code == 200
    assert client.post("/auth/login", json={"username": "operator", "password": PASSWORD}).status_code == 200
    assert client.get("/auth/session").json()["authenticated"] is True
    assert TestClient(create_app(backend.services, backend.operator, backend.api_settings), base_url="https://evil.example").get("/auth/session").status_code == 400


def test_vision_family_defaults_to_chat_accepts_reasoning_and_rejects_anything_else(tmp_path):
    base = {"SNN_VISION_ENDPOINT": "https://v.example.com", "SNN_VISION_DEPLOYMENT": "gpt-5-mini"}
    assert from_env(make_env(tmp_path, **base)).vision_family == "chat"
    cfg = from_env(make_env(tmp_path, SNN_VISION_FAMILY="reasoning", **base))
    assert cfg.vision_family == "reasoning" and cfg.vision_api_version is None  # the client picks the family's default
    with pytest.raises(BackendConfigError, match="SNN_VISION_FAMILY"):
        from_env(make_env(tmp_path, SNN_VISION_FAMILY="turbo", **base))
    assert from_env(make_env(tmp_path, SNN_VISION_API_VERSION="2025-06-01", **base)).vision_api_version == "2025-06-01"


def test_the_configured_family_reaches_the_vision_client(tmp_path):
    cfg = from_env(make_env(tmp_path, SNN_VISION_ENDPOINT="https://v.example.com", SNN_VISION_DEPLOYMENT="gpt-5-mini",
                            SNN_VISION_FAMILY="reasoning", SNN_VISION_API_KEY="k"))  # fmt: skip
    vision = build(cfg).worker.vision
    assert vision._family == "reasoning" and vision._path.endswith("api-version=2025-04-01-preview")


# ------------------------------------------------------------------------------------------- entry points


def test_check_config_validates_and_exits_without_serving(tmp_path, capsys):
    assert server.main(["api", "--check-config"], environ=make_env(tmp_path)) == 0
    assert "configuration ok" in capsys.readouterr().out


def test_the_server_keeps_the_azure_sdk_request_log_out_of_the_service_log(tmp_path):
    import logging

    root = logging.getLogger()
    before = root.level
    root.setLevel(logging.INFO)  # as the server itself sets it; without the cap the SDK would inherit INFO
    try:
        for name in ("azure", "urllib3"):
            logging.getLogger(name).setLevel(logging.NOTSET)
        server.main(["api", "--check-config"], environ=make_env(tmp_path))
        assert logging.getLogger("azure.core.pipeline.policies.http_logging_policy").getEffectiveLevel() == logging.WARNING
        assert logging.getLogger("urllib3.connectionpool").getEffectiveLevel() == logging.WARNING
    finally:
        root.setLevel(before)


def test_a_bad_configuration_exits_with_code_two_naming_the_variable_and_no_secret(tmp_path, capsys):
    code = server.main(["both", "--check-config"], environ=make_env(tmp_path, SNN_OPERATOR_PASSWORD_HASH=None, SNN_STORAGE="memory"))
    err = capsys.readouterr().err
    assert code == 2 and "SNN_OPERATOR_PASSWORD_HASH" in err and VERIFIER not in err


def test_the_server_role_is_validated_by_the_command_line():
    with pytest.raises(SystemExit):
        server.main(["everything"], environ={})


def test_hash_password_reads_stdin_and_prints_a_verifier_never_the_password(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(PASSWORD + "\n"))
    assert admin.main(["hash-password"]) == 0
    out = capsys.readouterr().out.strip()
    assert PASSWORD not in out and verify_password(PASSWORD, out)


def test_a_short_password_is_refused(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("short\n"))
    assert admin.main(["hash-password"]) == 1 and "12 characters" in capsys.readouterr().err


def test_a_device_token_is_printed_once_and_authenticates_the_device(tmp_path, capsys):
    storage = memory_storage()
    env = make_env(tmp_path, SNN_STORAGE="azure", SNN_AZURE_ACCOUNT="acct", SNN_ALLOW_MEMORY=None)
    assert admin.main(["issue-device-token", "demo-pi"], environ=env, storage=storage) == 0
    token = capsys.readouterr().out.strip()
    assert authenticate_device(types.SimpleNamespace(storage=storage), token) == "demo-pi"
    assert admin.main(["issue-device-token", "demo-pi"], environ=env, storage=storage) == 0
    new = capsys.readouterr().out.strip()
    assert new != token and authenticate_device(types.SimpleNamespace(storage=storage), token) is None  # rotated


def test_a_device_credential_is_refused_when_it_would_live_only_in_memory(tmp_path, capsys):
    assert admin.main(["issue-device-token", "demo-pi"], environ=make_env(tmp_path)) == 1
    assert "memory" in capsys.readouterr().err
    assert admin.main(["issue-device-token", "bad id"], environ=make_env(tmp_path)) == 1


def test_a_configuration_error_in_admin_exits_with_two(tmp_path, capsys):
    assert admin.main(["issue-device-token", "demo-pi"], environ=make_env(tmp_path, SNN_RUNTIME=None)) == 2


def test_the_module_exports_are_the_documented_ones():
    assert set(backend_config.__all__) == {"Backend", "BackendConfig", "BackendConfigError", "build", "build_storage", "from_env"}


def test_admin_events_shows_the_reasoning_and_usage_without_secrets_or_the_raw_event(tmp_path, capsys):
    from rpi_agents.cloud.app.policy import ARMED
    from rpi_agents.cloud.app.vision import Observation
    from tests.w0.backend_env import Env
    from tests.w0.fakes import FakeNotifier, FakeVision

    env = Env(policy_version=ARMED, recipients=("owner@example.com",), event_cooldown_s=0)
    _, event_id, _ = env.pending_event()
    answer = Observation(True, True, "good", "Glass and a person.", "Shards by a window; a figure stands there.", {"prompt_tokens": 9, "reasoning_tokens": 4})
    env.worker(FakeVision([answer]), FakeNotifier()).run_once()

    assert admin.main(["events", "--limit", "3"], environ=make_env(tmp_path), storage=env.storage) == 0
    (line,) = capsys.readouterr().out.strip().splitlines()
    shown = json.loads(line)
    assert shown["event_id"] == event_id and shown["status"] == "alarm_confirmed" and shown["commands"] == ["capture", "alarm"]
    assert shown["vision"]["glass_visible"] is True and shown["vision"]["provenance"] == "demo"
    assert shown["vision_runs"] == [{"state": "done", "rationale": answer.rationale, "usage": answer.usage}]
    assert shown["notifications"]["email"]["status"] == "sent" and shown["images"] == 1
    assert "owner@example.com" not in line  # recipients are configuration, not event output


def test_admin_events_on_an_empty_store_prints_nothing_and_accepts_no_device_id(tmp_path, capsys):
    assert admin.main(["events"], environ=make_env(tmp_path), storage=memory_storage()) == 0
    assert capsys.readouterr().out == ""
