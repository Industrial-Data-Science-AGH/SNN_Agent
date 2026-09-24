import pytest

from rpi_agents.agent.api import UrllibTransport
from rpi_agents.cloud.app import admin, server
from rpi_agents.cloud.app.keyvault import KeyVault, KeyVaultError, resolve
from tests.w0.fakes import FakeHttp
from tests.w0.test_backend_config import make_env


@pytest.fixture
def http():
    server_ = FakeHttp()
    yield server_
    server_.close()


def vault(http, **kw):
    return KeyVault(http.url, token=lambda: "tok-1", transport=UrllibTransport(http.url), sleep=lambda s: None, **kw)


def test_a_secret_is_read_with_the_bearer_token_and_cached(http):
    http.script.append((200, {"value": "s3cret-value", "id": "x"}, {}))
    v = vault(http)
    assert v.get("smtp-password") == "s3cret-value" and v.get("smtp-password") == "s3cret-value"
    (method, path, headers, _), = http.requests  # the second read came from the cache
    assert (method, path) == ("GET", "/secrets/smtp-password?api-version=7.4") and headers["authorization"] == "Bearer tok-1"


@pytest.mark.parametrize("bad", ["../x", "a b", "", "x/y", "a" * 200, "smtp_password"])
def test_a_secret_name_that_could_change_the_request_is_refused(http, bad):
    with pytest.raises(KeyVaultError, match="letters, digits and hyphens"):
        vault(http).get(bad)
    assert http.requests == []


@pytest.mark.parametrize("status,needle", [(404, "does not exist"), (403, "Key Vault Secrets User"), (401, "Key Vault Secrets User")])
def test_missing_and_forbidden_secrets_fail_at_once_with_a_useful_message(http, status, needle):
    http.script.append((status, {"error": {"message": "the-real-value-must-not-be-echoed"}}, {}))
    with pytest.raises(KeyVaultError, match=needle) as error:
        vault(http).get("smtp-password")
    assert len(http.requests) == 1 and "the-real-value" not in str(error.value)


def test_a_server_error_is_retried_then_reported_without_any_body(http):
    http.script += [(503, {}, {}), (500, {}, {}), (200, {"value": "ok"}, {})]
    assert vault(http).get("a") == "ok" and len(http.requests) == 3
    http.script += [(503, {}, {})] * 3
    with pytest.raises(KeyVaultError, match="http 503"):
        vault(http).get("b")


@pytest.mark.parametrize("payload", [{}, {"value": 5}, [], "text"])
def test_a_malformed_answer_is_not_a_secret(http, payload):
    http.script += [(200, payload, {})] * 3
    with pytest.raises(KeyVaultError):
        vault(http).get("a")


def test_the_vault_url_must_be_a_real_vault_address():
    for bad in ("http://x.vault.azure.net", "https://evil.example.com", "https://x.vault.azure.net.evil.com", ""):
        with pytest.raises(KeyVaultError, match="SNN_KEYVAULT_URL"):
            KeyVault(bad, token=lambda: "t")
    KeyVault("https://snnbe-kv-abc.vault.azure.net/", token=lambda: "t")


def test_resolve_replaces_only_keyvault_values_and_leaves_the_input_alone(http):
    http.script += [(200, {"value": "pw"}, {}), (200, {"value": "hash"}, {})]
    env = {"SNN_SMTP_PASSWORD": "keyvault:smtp-password", "SNN_OPERATOR_PASSWORD_HASH": "keyvault: operator-hash ", "SNN_POLICY": "x", "SNN_KEYVAULT_URL": "u"}
    out = resolve(env, vault(http))
    assert out["SNN_SMTP_PASSWORD"] == "pw" and out["SNN_OPERATOR_PASSWORD_HASH"] == "hash" and out["SNN_POLICY"] == "x"
    assert env["SNN_SMTP_PASSWORD"] == "keyvault:smtp-password"  # not mutated


def test_resolve_without_references_is_a_plain_copy_and_needs_no_vault():
    env = {"A": "1"}
    out = resolve(env)
    assert out == env and out is not env


def test_a_reference_without_a_vault_url_is_a_configuration_error():
    with pytest.raises(KeyVaultError, match="SNN_KEYVAULT_URL is not set"):
        resolve({"SNN_SMTP_PASSWORD": "keyvault:smtp-password"})


def test_the_error_names_the_variable_and_the_secret_but_never_a_value(http):
    http.script.append((404, {}, {}))
    with pytest.raises(KeyVaultError, match="SNN_SMTP_PASSWORD: secret smtp-password does not exist"):
        resolve({"SNN_SMTP_PASSWORD": "keyvault:smtp-password"}, vault(http))


def test_the_managed_identity_is_used_when_the_platform_provides_one(http):
    tokens = FakeHttp()
    try:
        tokens.script.append((200, {"access_token": "msi-token", "expires_on": "9999999999"}, {}))
        http.script.append((200, {"value": "v"}, {}))
        env = {"IDENTITY_ENDPOINT": tokens.url + "/msi/token", "IDENTITY_HEADER": "h", "AZURE_CLIENT_ID": "cid"}
        assert KeyVault(http.url, env=env, transport=UrllibTransport(http.url)).get("x") == "v"
        assert http.requests[0][2]["authorization"] == "Bearer msi-token"
        assert "resource=https://vault.azure.net" in tokens.requests[0][1] and "client_id=cid" in tokens.requests[0][1]
    finally:
        tokens.close()


def test_server_and_admin_stop_with_exit_code_two_when_a_secret_cannot_be_resolved(tmp_path, capsys):
    env = make_env(tmp_path, SNN_OPERATOR_PASSWORD_HASH="keyvault:operator-hash")  # no SNN_KEYVAULT_URL
    assert server.main(["api", "--check-config"], environ=env) == 2
    assert "SNN_KEYVAULT_URL is not set" in capsys.readouterr().err
    assert admin.main(["events"], environ=env) == 2


def test_a_token_failure_names_our_own_error_but_never_an_unknown_exceptions_text(http):
    def broken_identity():
        raise ValueError("plain http is only allowed for loopback")

    with pytest.raises(KeyVaultError, match=r"token \(ValueError: plain http is only allowed"):
        KeyVault(http.url, token=broken_identity, transport=UrllibTransport(http.url), attempts=1).get("a")

    def other():
        raise RuntimeError("Bearer eyJhbGciOi-a-real-token")

    with pytest.raises(KeyVaultError) as error:
        KeyVault(http.url, token=other, transport=UrllibTransport(http.url), attempts=1).get("a")
    assert "RuntimeError" in str(error.value) and "eyJ" not in str(error.value)
