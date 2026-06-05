"""Unit tests for agent.notifier (P3)."""

import smtplib
from email.message import EmailMessage
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from agent import camera, config, notifier

_SNAPSHOT = np.zeros((32, 32, 3), dtype=np.uint8)
_FRAMES = np.zeros((4, 32, 32, 3), dtype=np.uint8)


class FakeSMTP:
    """Context-manager capturing login args and the sent EmailMessage."""

    def __init__(self, *args, **kwargs):
        self.login_args: tuple[str, str] | None = None
        self.sent_message: EmailMessage | None = None

    def __enter__(self) -> "FakeSMTP":
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def login(self, user: str, password: str) -> None:
        self.login_args = (user, password)

    def send_message(self, msg: EmailMessage) -> None:
        self.sent_message = msg


@pytest.fixture()
def fake_settings(monkeypatch: pytest.MonkeyPatch) -> config.Settings:
    settings = config.Settings(
        gemini_api_key="x",
        gmail_user="bot@gmail.com",
        gmail_app_password="pw",
        alert_to="owner@x.com",
    )
    monkeypatch.setattr(config, "load_settings", lambda: settings)
    return settings


@pytest.fixture()
def fake_smtp(monkeypatch: pytest.MonkeyPatch) -> FakeSMTP:
    instance = FakeSMTP()
    monkeypatch.setattr("smtplib.SMTP_SSL", lambda *a, **k: instance)
    return instance


def test_notify_builds_and_sends(
    fake_settings: config.Settings, fake_smtp: FakeSMTP
) -> None:
    notifier.notify("glass break detected", _SNAPSHOT)

    msg = fake_smtp.sent_message
    assert msg is not None
    assert msg["To"] == "owner@x.com"
    assert msg["From"] == "bot@gmail.com"
    assert config.ALERT_SUBJECT_PREFIX in msg["Subject"]
    assert fake_smtp.login_args == ("bot@gmail.com", "pw")

    attachments = [
        part for part in msg.walk()
        if part.get_content_maintype() == "image"
        and part.get_content_subtype() == "jpeg"
    ]
    assert len(attachments) == 1


def test_notify_reraises_on_smtp_failure(
    fake_settings: config.Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailSMTP(FakeSMTP):
        def send_message(self, msg: EmailMessage) -> None:
            raise smtplib.SMTPException("server down")

    monkeypatch.setattr("smtplib.SMTP_SSL", lambda *a, **k: FailSMTP())
    with pytest.raises(smtplib.SMTPException):
        notifier.notify("reason", _SNAPSHOT)


def test_save_clip_delegates_to_camera(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = tmp_path / "clip.mp4"
    spy = MagicMock(return_value=expected)
    monkeypatch.setattr(camera, "save_clip", spy)

    result = notifier.save_clip(_FRAMES)

    spy.assert_called_once_with(_FRAMES)
    assert result == expected
