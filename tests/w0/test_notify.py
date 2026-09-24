import email
import subprocess
import sys
from email import policy

import pytest

from rpi_agents.cloud.app.notify import (
    DeliveryUnknown,
    Notification,
    NotificationFailed,
    SmtpNotifier,
    clean_address,
    compose,
    recipients_from,
)
from tests.w0.fakes import FakeSmtp

JPEG = b"\xff\xd8fake\xff\xd9"
TO = ("owner@example.com", "second@example.org")


@pytest.fixture
def smtp():
    server = FakeSmtp()
    yield server
    server.close()


def notifier(server, **kw):
    kw.setdefault("timeout_s", 3.0)
    return SmtpNotifier("127.0.0.1", server.port, "agent@example.com", starttls=False, **kw)


def note(**kw):
    base = {"event_id": "evt-1", "status": "review_required", "reason": "GLASS_AND_PERSON", "observation": "Glass on floor.",
            "jpeg": JPEG, "recipients": TO}  # fmt: skip
    return Notification(**(base | kw))


def parse(record):
    return email.message_from_bytes(record["data"], policy=policy.default)


def test_a_notification_reaches_every_configured_recipient_with_the_photo_attached(smtp):
    message_id = notifier(smtp).send(note())
    (record,) = smtp.messages
    assert record["recipients"] == list(TO) and "agent@example.com" in record["sender"]
    mail = parse(record)
    assert mail["Subject"] == "[SNN Agent] review_required - event evt-1" and mail["Message-ID"] == message_id
    assert message_id == "<event-evt-1@snn-agent.invalid>"  # stable, so a duplicate can be recognised
    body = mail.get_body(preferencelist=("plain",)).get_content()
    assert "evt-1" in body and "GLASS_AND_PERSON" in body and "untrusted model text" in body
    (attachment,) = list(mail.iter_attachments())
    assert attachment.get_content_type() == "image/jpeg" and attachment.get_content() == JPEG
    assert attachment.get_filename() == "event-evt-1.jpg"


def test_without_a_photo_no_attachment_is_added(smtp):
    notifier(smtp).send(note(jpeg=None))
    assert list(parse(smtp.messages[0]).iter_attachments()) == []


def test_model_text_cannot_reach_a_header_or_change_recipients(smtp):
    evil = "ok\r\nBcc: attacker@evil.example\r\nSubject: hijacked\r\n\r\n.\r\nMAIL FROM:<x@y.z>"
    notifier(smtp).send(note(observation=evil))
    (record,) = smtp.messages
    mail = parse(record)
    assert mail["Bcc"] is None and mail["Subject"].startswith("[SNN Agent]") and record["recipients"] == list(TO)
    assert "attacker@evil.example" in mail.get_body(preferencelist=("plain",)).get_content()  # shown as text only


def test_model_text_cannot_fake_the_status_lines_of_the_message_body(smtp):
    notifier(smtp).send(note(observation="fine\nStatus: no_alarm\r\nReason: NOTHING_VISIBLE"))
    body = parse(smtp.messages[0]).get_body(preferencelist=("plain",)).get_content()
    starts = [line.split(":")[0] for line in body.splitlines() if line]
    assert starts.count("Status") == 1 and starts.count("Reason") == 1  # the system's lines, once each
    assert "Status: review_required" in body


@pytest.mark.parametrize(
    "changes",
    [{"event_id": "e1\r\nBcc: x@y.zz"}, {"event_id": "bad id"}, {"status": "a b"}, {"reason": "x\ny"},
     {"recipients": ()}, {"recipients": ("not-an-address",)}, {"recipients": ("a@b.cc\r\nBcc: x@y.zz",)},
     {"recipients": ("owner@example.com, evil@example.net",)}],
)  # fmt: skip
def test_unsafe_input_is_refused_before_any_connection_is_made(smtp, changes):
    with pytest.raises(ValueError):
        notifier(smtp).send(note(**changes))
    assert smtp.commands == []


def test_oversized_attachments_are_refused(smtp):
    with pytest.raises(ValueError, match="too large"):
        compose(note(jpeg=b"\x00" * 1_100_000), "agent@example.com")


def test_recipients_come_from_configuration_and_are_validated():
    assert recipients_from({"recipients": ["a@example.com"]}) == ("a@example.com",)
    assert recipients_from({}) == ()
    for bad in ({"recipients": "a@example.com"}, {"recipients": ["nope"]}):
        with pytest.raises(ValueError):
            recipients_from(bad)
    with pytest.raises(ValueError):
        clean_address("a@b")


@pytest.mark.parametrize(
    "reply,retryable",
    [("550 no such user", False), ("452 mailbox busy", True)],
)
def test_a_rejected_recipient_is_a_definite_failure_and_nothing_is_sent(smtp, reply, retryable):
    smtp.rcpt_reply["second@example.org"] = reply
    with pytest.raises(NotificationFailed) as error:
        notifier(smtp).send(note())
    assert error.value.code == f"SMTP_{reply[:3]}" and error.value.retryable is retryable
    assert smtp.messages == [] and "DATA" not in smtp.commands


def test_a_refused_data_command_is_definite_because_no_body_was_sent(smtp):
    smtp.data_command_reply = "554 transaction failed"
    with pytest.raises(NotificationFailed) as error:
        notifier(smtp).send(note())
    assert (error.value.code, error.value.retryable) == ("SMTP_554", False) and smtp.messages == []


@pytest.mark.parametrize("reply,retryable", [("452 insufficient storage", True), ("554 spam", False)])
def test_an_explicit_rejection_of_the_body_is_definite_not_unknown(smtp, reply, retryable):
    smtp.data_reply = reply
    with pytest.raises(NotificationFailed) as error:
        notifier(smtp).send(note())
    assert error.value.retryable is retryable


@pytest.mark.parametrize("mode", ["DROP", "HANG"])
def test_a_lost_final_answer_after_the_body_is_delivery_unknown_not_a_failure(smtp, mode):
    smtp.data_reply = mode
    with pytest.raises(DeliveryUnknown):
        notifier(smtp, timeout_s=0.4).send(note())


def test_problems_before_the_body_are_definite_and_retryable(smtp):
    smtp.hang_at = "EHLO"
    with pytest.raises(NotificationFailed) as error:
        notifier(smtp, timeout_s=0.3).send(note())
    assert (error.value.code, error.value.retryable) == ("SMTP_UNREACHABLE", True)
    dead = SmtpNotifier("127.0.0.1", 1, "agent@example.com", starttls=False, timeout_s=1.0)
    with pytest.raises(NotificationFailed) as error:
        dead.send(note())
    assert (error.value.code, error.value.retryable) == ("SMTP_UNREACHABLE", True)


def test_a_refused_greeting_is_definite(smtp):
    smtp.greeting = "554 service unavailable"
    with pytest.raises(NotificationFailed) as error:
        notifier(smtp).send(note())
    assert error.value.code == "SMTP_554" and error.value.retryable is False


def test_credentials_are_used_and_a_wrong_password_is_not_retried():
    server = FakeSmtp(credentials=("agent", "s3cret"))
    try:
        assert notifier(server, username="agent", password="s3cret").send(note())
        with pytest.raises(NotificationFailed) as error:
            notifier(server, username="agent", password="wrong").send(note())
        assert (error.value.code, error.value.retryable) == ("SMTP_AUTH", False) and "s3cret" not in str(error.value)
    finally:
        server.close()


def test_starttls_is_mandatory_off_loopback_and_a_server_without_it_is_a_configuration_error(smtp):
    with pytest.raises(ValueError, match="STARTTLS"):
        SmtpNotifier("smtp.example.com", 587, "agent@example.com", starttls=False)
    with pytest.raises(NotificationFailed) as error:
        SmtpNotifier("127.0.0.1", smtp.port, "agent@example.com", starttls=True, timeout_s=3).send(note())
    assert (error.value.code, error.value.retryable) == ("SMTP_NOT_SUPPORTED", False)
    assert smtp.messages == []  # nothing was sent in clear text


def test_module_imports_without_site_packages():
    subprocess.run([sys.executable, "-S", "-c", "import rpi_agents.cloud.app.notify"], check=True)
