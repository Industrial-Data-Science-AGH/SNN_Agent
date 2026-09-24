"""Secrets from Azure Key Vault, resolved by the process itself at start-up.

Why not the platform's own Key Vault references: Container Apps "express" environments (what a new students
subscription gets) refuse them. Resolving in the process works on every host, keeps the app definition free of secret
values AND of secret names' contents, and picks up a rotated secret on the next restart.

An environment value of the form `keyvault:<secret-name>` is replaced by the secret's value; SNN_KEYVAULT_URL names
the vault. The token comes from the managed identity when running on Container Apps, else from the developer's
`az login` (through azure-identity, imported only then). Values are never logged; an error names the variable and the
secret, never a value.
"""

from __future__ import annotations

import os
import re
import time
from typing import Callable, Mapping

from rpi_agents.agent.api import Transport, TransportError, UrllibTransport
from rpi_agents.cloud.app.vision import ManagedIdentityAuth, VisionUnavailable

PREFIX = "keyvault:"
VAULT_RESOURCE = "https://vault.azure.net"
API_VERSION = "7.4"
_SECRET_NAME = re.compile(r"[A-Za-z0-9-]{1,127}")
_VAULT_URL = re.compile(r"https://[A-Za-z0-9-]{3,24}\.vault\.azure\.net/?")


class KeyVaultError(Exception):
    """A secret could not be read. The message never contains a secret value."""


def _developer_token() -> str:
    try:
        from azure.identity import DefaultAzureCredential
    except ImportError:
        raise KeyVaultError("no managed identity here and azure-identity is not installed") from None
    try:
        return DefaultAzureCredential().get_token(f"{VAULT_RESOURCE}/.default").token
    except Exception as exc:  # noqa: BLE001 - the SDK raises many types; none may leak into a log with details
        raise KeyVaultError(f"cannot get a Key Vault token: {type(exc).__name__}") from None


class KeyVault:
    def __init__(
        self, url: str, *, token: Callable[[], str] | None = None, transport: Transport | None = None,
        env: Mapping[str, str] | None = None, attempts: int = 3, sleep: Callable[[float], None] = time.sleep,
    ):  # fmt: skip
        loopback = transport is not None and url.startswith("http://127.0.0.1")  # tests only: the fake server
        if not loopback and not _VAULT_URL.fullmatch(url):
            raise KeyVaultError("SNN_KEYVAULT_URL must look like https://<name>.vault.azure.net")
        self._url = url.rstrip("/")
        env = os.environ if env is None else env
        if token is None:
            if env.get("IDENTITY_ENDPOINT") and env.get("IDENTITY_HEADER"):
                identity = ManagedIdentityAuth(VAULT_RESOURCE, env=env)
                token = lambda: identity()["Authorization"].removeprefix("Bearer ")  # noqa: E731
            else:
                token = _developer_token
        self._token, self._attempts, self._sleep = token, attempts, sleep
        self._transport = transport or UrllibTransport(self._url)
        self._cache: dict[str, str] = {}

    def get(self, name: str) -> str:
        if not _SECRET_NAME.fullmatch(name):
            raise KeyVaultError("a secret name may contain only letters, digits and hyphens")
        if name in self._cache:
            return self._cache[name]
        last = "unknown"
        for attempt in range(self._attempts):
            try:
                bearer = self._token()
                response = self._transport.request(
                    "GET", f"/secrets/{name}?api-version={API_VERSION}", headers={"Authorization": f"Bearer {bearer}"}, timeout_s=10.0,
                )  # fmt: skip
            except TransportError as exc:
                last = f"network ({exc})"
            except KeyVaultError:
                raise
            except Exception as exc:  # noqa: BLE001 - a token failure, e.g. from the identity endpoint
                # The message of a VisionUnavailable/ValueError comes from our own code and holds no secret; others only get a type.
                detail = str(exc)[:80] if isinstance(exc, (ValueError, VisionUnavailable)) else ""
                last = f"token ({type(exc).__name__}{': ' + detail if detail else ''})"
            else:
                if response.status == 200 and isinstance(response.body, dict) and isinstance(response.body.get("value"), str):
                    self._cache[name] = response.body["value"]
                    return self._cache[name]
                if response.status in (401, 403):
                    raise KeyVaultError(f"access denied to secret {name}: the identity needs Key Vault Secrets User")
                if response.status == 404:
                    raise KeyVaultError(f"secret {name} does not exist")
                last = f"http {response.status}"
            if attempt + 1 < self._attempts:
                self._sleep(2.0 * (attempt + 1))  # a fresh identity or role can take a moment to be accepted
        raise KeyVaultError(f"could not read secret {name}: {last}")


def resolve(env: Mapping[str, str], vault: KeyVault | None = None) -> dict[str, str]:
    """A copy of `env` with every `keyvault:<name>` value replaced by the secret. Nothing to resolve = a plain copy."""
    resolved = dict(env)
    wanted = {key: value[len(PREFIX):].strip() for key, value in env.items() if isinstance(value, str) and value.startswith(PREFIX)}
    if not wanted:
        return resolved
    if vault is None:
        url = env.get("SNN_KEYVAULT_URL", "").strip()
        if not url:
            raise KeyVaultError(f"{sorted(wanted)[0]} refers to Key Vault but SNN_KEYVAULT_URL is not set")
        vault = KeyVault(url, env=env)
    for key, name in wanted.items():
        try:
            resolved[key] = vault.get(name)
        except KeyVaultError as exc:
            raise KeyVaultError(f"{key}: {exc}") from None
    return resolved
