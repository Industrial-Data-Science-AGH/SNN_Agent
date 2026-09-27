"""E-mail notification for events. Standard library only (smtplib, email).

Rules:
- Recipients come only from configuration; nothing that a model or a device says can change them, add an
  attachment, or reach a header. The subject is built from validated fields only.
- Delivery is never over-claimed. A failure BEFORE the message body was sent is definite (NotificationFailed,
  safe to retry). A failure AFTER the body was sent, or a lost final answer, is ambiguous (DeliveryUnknown):
  the mail may have been delivered, so it is recorded as such and not blindly resent as if nothing happened.
  The Message-ID is stable per event and attempt-independent, so a duplicate can at least be recognised.
- STARTTLS is required unless the server is on loopback (tests, a local relay).
"""

from __future__ import annotations

import re
import smtplib
import socket
import ssl
from dataclasses import dataclass, field
from email.message import EmailMessage
from email.utils import formatdate
from typing import Mapping, Protocol

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_ADDRESS = re.compile(r"[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9.-]{1,190}\.[A-Za-z]{2,24}")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
LOOPBACK = {"localhost", "127.0.0.1", "::1"}
MAX_ATTACHMENT_BYTES = 1_048_576


class NotificationFailed(Exception):
    """Definitely not delivered. `retryable` says whether trying again can help."""

    def __init__(self, code: str, retryable: bool):
        super().__init__(code)
        self.code, self.retryable = code, retryable


class DeliveryUnknown(Exception):
    """The message may have been delivered: the connection failed after the body was sent."""


@dataclass(frozen=True)
class Notification:
    event_id: str
    status: str  # the event status, from a fixed set
    reason: str  # the policy reason code
    observation: str | None = None  # model text: data only, sanitised, never a header
    jpeg: bytes | None = None
    recipients: tuple[str, ...] = field(default_factory=tuple)


class Notifier(Protocol):
    def send(self, notification: Notification) -> str:
        """Return the Message-ID. Raise NotificationFailed or DeliveryUnknown."""
        ...


def clean_address(value: str) -> str:
    if not isinstance(value, str) or not _ADDRESS.fullmatch(value):
        raise ValueError("not a plain e-mail address")
    return value


def compose(notification: Notification, sender: str) -> EmailMessage:
    if not _ID.fullmatch(notification.event_id) or not _ID.fullmatch(notification.reason) or not _ID.fullmatch(notification.status):
        raise ValueError("event id, status and reason must be plain identifiers")
    if not notification.recipients:
        raise ValueError("no recipients configured")
    message = EmailMessage()
    message["From"] = clean_address(sender)
    message["To"] = ", ".join(clean_address(r) for r in notification.recipients)
    message["Subject"] = f"[SNN Agent] {notification.status} - event {notification.event_id}"
    message["Message-ID"] = f"<event-{notification.event_id}@snn-agent.invalid>"
    message["Date"] = formatdate(localtime=False)
    note = _CONTROL.sub(" ", notification.observation or "").strip()[:512]
    message.set_content(
        f"Event: {notification.event_id}\nStatus: {notification.status}\nReason: {notification.reason}\n"
        f"Vision observation (untrusted model text): {note or 'none'}\n\n"
        "This message is automatic. The attached image, if any, is the photo taken for this event.\n"
    )
    if notification.jpeg is not None:
        if len(notification.jpeg) > MAX_ATTACHMENT_BYTES:
            raise ValueError("attachment too large")
        message.add_attachment(notification.jpeg, maintype="image", subtype="jpeg", filename=f"event-{notification.event_id}.jpg")
    return message


class SmtpNotifier:
    def __init__(
        self, host: str, port: int, sender: str, *, username: str | None = None, password: str | None = None,
        starttls: bool = True, timeout_s: float = 15.0,
    ):  # fmt: skip
        if starttls is False and host not in LOOPBACK:
            raise ValueError("STARTTLS is required for a server that is not on loopback")
        self._host, self._port, self._sender = host, port, clean_address(sender)
        self._user, self._password, self._starttls, self._timeout = username, password, starttls, timeout_s

    def send(self, notification: Notification) -> str:
        message = compose(notification, self._sender)
        recipients = list(notification.recipients)
        phase = "before"  # flips to "after" once the message body is on its way
        try:
            with smtplib.SMTP(self._host, self._port, timeout=self._timeout) as smtp:
                smtp.ehlo()
                if self._starttls:
                    smtp.starttls(context=ssl.create_default_context())
                    smtp.ehlo()
                if self._user:
                    smtp.login(self._user, self._password or "")
                code, _ = smtp.mail(self._sender)
                self._expect(code)
                for recipient in recipients:
                    self._expect(smtp.rcpt(recipient)[0])
                code, _ = smtp.docmd("DATA")
                self._expect(code, ok=354)  # refused here: nothing was sent, so the failure is definite
                phase = "after"  # from now on a lost connection or a missing answer is ambiguous
                smtp.send(smtplib.quotedata(message.as_string()) + "\r\n.\r\n")
                code, _ = smtp.getreply()
                self._expect(code)  # an explicit answer, even a rejection, is definite
                try:
                    smtp.quit()
                except smtplib.SMTPException:
                    pass  # the message was accepted; a failing QUIT changes nothing
        except NotificationFailed:
            raise
        except smtplib.SMTPAuthenticationError:
            raise NotificationFailed("SMTP_AUTH", retryable=False) from None
        except smtplib.SMTPNotSupportedError:  # STARTTLS or AUTH asked for but not offered: a configuration problem
            raise NotificationFailed("SMTP_NOT_SUPPORTED", retryable=False) from None
        except smtplib.SMTPResponseException as exc:
            raise NotificationFailed(f"SMTP_{exc.smtp_code}", retryable=400 <= exc.smtp_code < 500) from None
        except (smtplib.SMTPException, OSError, socket.timeout, ssl.SSLError) as exc:
            if phase == "after":
                raise DeliveryUnknown(type(exc).__name__) from None
            raise NotificationFailed("SMTP_UNREACHABLE", retryable=True) from None
        return message["Message-ID"]

    @staticmethod
    def _expect(code: int, ok: int | None = None) -> None:
        if code >= 400 or (ok is not None and code != ok):
            raise NotificationFailed(f"SMTP_{code}", retryable=code < 500)


def recipients_from(config: Mapping[str, object]) -> tuple[str, ...]:
    """The recipient list from configuration, each address validated."""
    raw = config.get("recipients", [])
    if not isinstance(raw, (list, tuple)):
        raise ValueError("recipients must be a list")
    return tuple(clean_address(r) for r in raw)
