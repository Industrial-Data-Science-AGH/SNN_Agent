"""Access control: the shared operator login, device credentials, CSRF and log redaction.

Decisions (from the plan and the architecture document):
- ONE shared username and password for the whole team, verified on the server. There are no accounts, roles or
  registration. The audit `actor` is `shared_operator`: it identifies the account, never a person.
- Nothing has a default. A missing username or password hash is a startup error, not a fallback.
- The password is stored only as a scrypt verifier (in Key Vault or an environment secret). Comparison is
  constant time, and a wrong username costs the same work as a wrong password.
- Session tokens are random, only their SHA-256 is stored, and a session has a hard lifetime plus a sliding idle
  timeout. Logout deletes the session on the server, so a stolen cookie stops working.
- Login attempts are limited per client and globally, with Retry-After. State is in memory: it resets when the
  process restarts, which is acceptable for one replica and is stated here, not hidden.
- Every browser request that changes state must carry the session's CSRF token.
- A device authenticates with its own bearer token `<device_id>~<secret>`. Only a hash is stored. The token is
  bound to that device: it opens that device's data and nothing else.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import re
import secrets
import threading
from collections import deque
from dataclasses import dataclass
from datetime import timedelta

from contracts.validation import ContractError
from rpi_agents.cloud.app.context import Context
from rpi_agents.cloud.app.records import DEVICES_PK, T_DEVICES, parse_utc, utc_z
from rpi_agents.cloud.app.storage import Conflict, PreconditionFailed

T_WEB_SESSIONS = "websessions"
WEB_PK = "sess"
ACTOR = "shared_operator"
MIN_SCRYPT_LOG2_N = 14
_SCRYPT_MAXMEM = 256 * 1024 * 1024
_DEVICE_TOKEN = re.compile(r"([A-Za-z0-9][A-Za-z0-9_.-]{0,63})~([A-Za-z0-9_-]{32,128})")


# ------------------------------------------------------------------------------------------ passwords


def _scrypt(password: str, salt: bytes, log2_n: int, r: int, p: int) -> bytes:
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=1 << log2_n, r=r, p=p, dklen=32, maxmem=_SCRYPT_MAXMEM)


def hash_password(password: str, *, log2_n: int = 15, r: int = 8, p: int = 1, salt: bytes | None = None) -> str:
    """A verifier to store instead of the password: scrypt$<log2 N>$<r>$<p>$<salt>$<hash> (base64)."""
    if not password:
        raise ValueError("an empty password is not a password")
    if log2_n < MIN_SCRYPT_LOG2_N:
        raise ValueError(f"scrypt cost below 2^{MIN_SCRYPT_LOG2_N} is refused")
    salt = salt if salt is not None else secrets.token_bytes(16)
    digest = _scrypt(password, salt, log2_n, r, p)
    b64 = lambda raw: base64.b64encode(raw).decode("ascii")  # noqa: E731
    return f"scrypt${log2_n}${r}${p}${b64(salt)}${b64(digest)}"


def parse_verifier(stored: str) -> tuple[int, int, int, bytes, bytes]:
    parts = stored.split("$") if isinstance(stored, str) else []
    if len(parts) != 6 or parts[0] != "scrypt":
        raise ValueError("not a scrypt verifier")
    try:
        log2_n, r, p = int(parts[1]), int(parts[2]), int(parts[3])
        salt, digest = base64.b64decode(parts[4], validate=True), base64.b64decode(parts[5], validate=True)
    except ValueError:
        raise ValueError("malformed verifier") from None
    if log2_n < MIN_SCRYPT_LOG2_N or log2_n > 22 or not 1 <= r <= 32 or not 1 <= p <= 8 or len(salt) < 8 or len(digest) != 32:
        raise ValueError("verifier parameters are outside the accepted range")
    return log2_n, r, p, salt, digest


def verify_password(password: str, stored: str) -> bool:
    try:
        log2_n, r, p, salt, digest = parse_verifier(stored)
    except ValueError:
        return False
    return hmac.compare_digest(_scrypt(password, salt, log2_n, r, p), digest)


# ------------------------------------------------------------------------------------ device credentials


def _token_hash(secret: str) -> str:
    return hashlib.sha256(secret.encode("ascii")).hexdigest()


def issue_device_token(ctx: Context, device_id: str) -> str:
    """Create or rotate a device's credential and return the token ONCE. Only its hash is stored."""
    tables = ctx.storage.tables
    secret = secrets.token_urlsafe(32)
    for _ in range(8):
        row = tables.get(T_DEVICES, DEVICES_PK, device_id)
        data = {"device_id": device_id, "epoch": 0, "active_session_id": None, "active": True} if row is None else row.data
        data = data | {"token_hash": _token_hash(secret), "active": True}
        try:
            if row is None:
                tables.insert(T_DEVICES, DEVICES_PK, device_id, data)
            else:
                tables.replace(T_DEVICES, DEVICES_PK, device_id, data, row.etag)
            return f"{device_id}~{secret}"
        except (Conflict, PreconditionFailed):
            continue
    raise RuntimeError("could not store the device credential")


def authenticate_device(ctx: Context, token: str) -> str | None:
    """The device id a bearer token belongs to, or None. The comparison is constant time."""
    match = _DEVICE_TOKEN.fullmatch(token or "")
    if match is None:
        return None
    device_id, secret = match.groups()
    row = ctx.storage.tables.get(T_DEVICES, DEVICES_PK, device_id)
    stored = row.data.get("token_hash", "") if row is not None else ""
    ok = hmac.compare_digest(_token_hash(secret), stored or "0" * 64)  # always compares, so unknown devices cost the same
    return device_id if ok and row is not None and row.data.get("active") and stored else None


# ---------------------------------------------------------------------------------------- operator login


@dataclass(frozen=True)
class WebSession:
    csrf_token: str
    expires_at: str
    actor: str = ACTOR


class OperatorAuth:
    def __init__(
        self, ctx: Context, *, username: str, password_hash: str, ttl_s: float = 8 * 3600, idle_ttl_s: float = 30 * 60,
        max_failures: int = 5, window_s: float = 300.0, lockout_s: float = 300.0, global_max_failures: int = 30,
    ):  # fmt: skip
        if not username or not password_hash:
            raise ValueError("the operator username and password verifier must be configured: there are no defaults")
        parse_verifier(password_hash)  # refuse a malformed or weak verifier at startup
        if ttl_s <= 0 or idle_ttl_s <= 0 or max_failures < 1 or global_max_failures < max_failures:
            raise ValueError("invalid session or rate limit settings")
        self.ctx, self._username, self._verifier = ctx, username, password_hash
        self._ttl, self._idle = ttl_s, idle_ttl_s
        self._max, self._window, self._lockout, self._global_max = max_failures, window_s, lockout_s, global_max_failures
        self._failures: dict[str, deque] = {}
        self._global: deque = deque()
        self._locked_until: dict[str, float] = {}
        self._lock = threading.Lock()
        self._dummy = hash_password("not-the-password", log2_n=MIN_SCRYPT_LOG2_N)

    # ------------------------------------------------------------------------------------- rate limits

    def _now(self) -> float:
        return self.ctx.now().timestamp()

    def _check_limits(self, client: str, now: float) -> None:
        with self._lock:
            until = self._locked_until.get(client, 0.0)
            if until > now:
                raise ContractError("TOO_MANY_ATTEMPTS", f"Try again in {int(until - now) + 1} seconds", 429)
            while self._global and now - self._global[0] > self._window:
                self._global.popleft()
            if len(self._global) >= self._global_max:
                raise ContractError("TOO_MANY_ATTEMPTS", "Too many failed sign-in attempts; try again later", 429)

    def retry_after_s(self, client: str) -> int:
        with self._lock:
            return max(1, int(self._locked_until.get(client, 0.0) - self._now()) + 1)

    def _record_failure(self, client: str, now: float) -> None:
        with self._lock:
            recent = self._failures.setdefault(client, deque())
            recent.append(now)
            while recent and now - recent[0] > self._window:
                recent.popleft()
            self._global.append(now)
            if len(recent) >= self._max:
                self._locked_until[client] = now + self._lockout
                recent.clear()
            if len(self._failures) > 10_000:  # bound the memory an attacker can make us hold
                self._failures.clear()

    # ------------------------------------------------------------------------------------------ login

    def login(self, username: str, password: str, client: str) -> tuple[str, WebSession]:
        now = self._now()
        self._check_limits(client, now)
        user_ok = hmac.compare_digest(str(username).encode("utf-8"), self._username.encode("utf-8"))
        password_ok = verify_password(str(password), self._verifier if user_ok else self._dummy)  # same work either way
        if not (user_ok and password_ok):
            self._record_failure(client, now)
            raise ContractError("INVALID_CREDENTIALS", "Invalid credentials", 401)
        with self._lock:
            self._failures.pop(client, None)
        token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        expires = utc_z(self.ctx.now() + timedelta(seconds=self._ttl))
        self.ctx.storage.tables.insert(T_WEB_SESSIONS, WEB_PK, _token_hash(token), {
            "csrf": csrf, "created_at": utc_z(self.ctx.now()), "last_seen_at": utc_z(self.ctx.now()), "expires_at": expires,
        })  # fmt: skip
        return token, WebSession(csrf, expires)

    def session(self, token: str | None) -> WebSession | None:
        """The live session for a cookie token, sliding the idle timeout; None when unknown, expired or idle."""
        if not token or len(token) > 200:
            return None
        tables, key = self.ctx.storage.tables, _token_hash(token)
        row = tables.get(T_WEB_SESSIONS, WEB_PK, key)
        if row is None:
            return None
        now = self.ctx.now()
        expired = parse_utc(row.data["expires_at"]) <= now
        idle = (now - parse_utc(row.data["last_seen_at"])).total_seconds() > self._idle
        if expired or idle:
            try:
                tables.delete(T_WEB_SESSIONS, WEB_PK, key)
            except Exception:
                pass
            return None
        try:
            tables.replace(T_WEB_SESSIONS, WEB_PK, key, row.data | {"last_seen_at": utc_z(now)}, row.etag)
        except PreconditionFailed:
            pass  # a concurrent request slid it already
        return WebSession(row.data["csrf"], row.data["expires_at"])

    def logout(self, token: str | None) -> None:
        if token:
            try:
                self.ctx.storage.tables.delete(T_WEB_SESSIONS, WEB_PK, _token_hash(token))
            except Exception:
                pass  # already gone: the result is the same


def check_csrf(session: WebSession, presented: str | None) -> bool:
    return bool(presented) and hmac.compare_digest(presented.encode("utf-8"), session.csrf_token.encode("utf-8"))


def client_ip(forwarded_for: str | None, peer: str, trusted_proxies: int) -> str:
    """The client address, trusting only the last `trusted_proxies` hops of X-Forwarded-For."""
    if trusted_proxies <= 0 or not forwarded_for:
        return peer
    hops = [h.strip() for h in forwarded_for.split(",") if h.strip()]
    return hops[-trusted_proxies] if len(hops) >= trusted_proxies else peer


# ------------------------------------------------------------------------------------------ log redaction

_REDACT = [
    (re.compile(r"(?i)(authorization\s*[:=]\s*)(?:(?:bearer|basic|token|digest|negotiate)\s+)?[^\s,;]+"), r"\1[redacted]"),
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}"), r"\1[redacted]"),
    (re.compile(r"(?i)(cookie\s*[:=]\s*)[^\r\n]+"), r"\1[redacted]"),
    (re.compile(r"(?i)(set-cookie\s*[:=]\s*)[^\r\n]+"), r"\1[redacted]"),
    (re.compile(r"(?i)((?:password|passwd|secret|token|api[-_]?key|x-csrf-token|x-identity-header)[\"']?\s*[:=]\s*[\"']?)[^\s,;\"'&]+"), r"\1[redacted]"),
    (re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}~[A-Za-z0-9_-]{32,128}"), "[redacted-device-token]"),
    (re.compile(r"(?i)(signature=|sig=|sv=)[A-Za-z0-9%+/=_-]{8,}"), r"\1[redacted]"),
]


def redact(text: str) -> str:
    for pattern, replacement in _REDACT:
        text = pattern.sub(replacement, text)
    return text


class RedactingFilter(logging.Filter):
    """Scrubs credentials from every log record (message, arguments and formatted exception text)."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg, record.args = redact(record.getMessage()), None
            if record.exc_info:
                record.exc_text = redact("".join(logging.Formatter().formatException(record.exc_info)))
                record.exc_info = None
        except Exception:  # a logging failure must never break the request
            record.msg, record.args = "[log record could not be redacted]", None
        return True


def install_log_redaction() -> None:
    for handler in logging.getLogger().handlers:
        handler.addFilter(RedactingFilter())
    logging.getLogger().addFilter(RedactingFilter())
