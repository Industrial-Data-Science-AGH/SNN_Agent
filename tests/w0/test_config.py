import copy
import os
import subprocess
import sys

import pytest

from contracts.validation import fixture
from rpi_agents.agent.config import ConfigError, load_config, parse_config, read_token

CREATE = fixture("session-create")
GOOD = {
    "device": {"id": "snn-pi", "input_kind": "stand_in"},
    "backend": {"url": "http://127.0.0.1:8000"},
    "session": {"mode": "demo", "model_hash": CREATE["model_hash"], "encoder_hash": CREATE["encoder_hash"]},
    "serial": {
        "path": "/dev/serial/by-id/usb-1a86_USB2.0-Serial-if00-port0",
        "channels": ["peak", "peak_cnt", "cv", "zcr", "flux", "hf_lo", "hf_hi"],
    },
    "state": {"dir": "/tmp/snn-edge"},
}


def cfg(**changes):
    data = copy.deepcopy(GOOD)
    for dotted, value in changes.items():
        section, _, key = dotted.partition("__")
        if value is None:
            data.get(section, {}).pop(key, None) if key else data.pop(section, None)
        elif key:
            data.setdefault(section, {})[key] = value
        else:
            data[section] = value
    return data


def test_a_minimal_valid_config_gets_safe_defaults():
    c = parse_config(GOOD)
    assert (c.device.id, c.device.input_kind, c.session.mode) == ("snn-pi", "stand_in", "demo")
    assert (c.serial.baud, c.limits.batch_ms, c.limits.heartbeat_s, c.limits.max_pending) == (115200, 250, 5.0, 5000)
    assert c.camera.serial is None and c.backend.credential_file is None and c.serial.expected_build_id is None
    assert c.serial.channels[0] == "peak" and len(c.serial.channels) == 7


@pytest.mark.parametrize(
    "changes,needle",
    [
        ({"device__id": "bad id"}, "device.id"),
        ({"device__input_kind": "arduino"}, "input_kind"),
        ({"device__id": 5}, "wrong type"),
        ({"device__extra": 1}, "unknown key"),
        ({"surprise": {"a": 1}}, "unknown section"),
        ({"backend": None}, "missing section [backend]"),
        ({"backend__url": "http://api.example.com"}, "loopback"),
        ({"backend__url": "https://api.example.com"}, "credential_file is required"),
        ({"session__mode": "prod"}, "session.mode"),
        ({"session__model_hash": "sha256:abc"}, "model_hash"),
        ({"session__encoder_hash": "SHA256:" + "0" * 64}, "encoder_hash"),
        ({"serial__channels": []}, "channels"),
        ({"serial__channels": ["a", "a"]}, "duplicates"),
        ({"serial__channels": ["bad name"]}, "channels"),
        ({"serial__path": "/dev/ttyACM0"}, "by-id"),
        ({"serial__path": None}, "needs serial.path"),
        ({"serial__replay_file": "/x/replay.bin"}, "needs serial.path and no serial.replay_file"),
        ({"serial__expected_build_id": "abc"}, "expected_build_id"),
        ({"serial__baud": 5}, "serial.baud"),
        ({"backend__timeout_s": 0}, "timeout_s"),
        ({"limits__batch_ms": 5000}, "batch_ms"),
        ({"limits__heartbeat_s": True}, "wrong type"),
        ({"state__dir": ""}, "state.dir"),
    ],
)
def test_invalid_configs_fail_loudly_and_name_the_key(changes, needle):
    with pytest.raises(ConfigError, match=needle.replace("[", r"\[").replace("]", r"\]")):
        parse_config(cfg(**changes))


def test_replay_input_needs_a_replay_file_and_no_serial_path():
    ok = cfg(device__input_kind="replay", serial__path=None, serial__replay_file="/data/replay.bin")
    assert parse_config(ok).serial.replay_file == "/data/replay.bin"
    with pytest.raises(ConfigError, match="replay_file"):
        parse_config(cfg(device__input_kind="replay"))


def test_a_remote_https_backend_with_a_credential_is_accepted():
    c = parse_config(cfg(backend__url="https://api.example.com/", backend__credential_file="/etc/snn/token"))
    assert c.backend.url == "https://api.example.com" and c.backend.credential_file == "/etc/snn/token"


def test_load_config_reads_toml_and_reports_missing_or_broken_files(tmp_path):
    path = tmp_path / "edge.toml"
    path.write_text(
        '[device]\nid = "snn-pi"\ninput_kind = "stand_in"\n[backend]\nurl = "http://127.0.0.1:8000"\n'
        f'[session]\nmode = "demo"\nmodel_hash = "{CREATE["model_hash"]}"\nencoder_hash = "{CREATE["encoder_hash"]}"\n'
        '[serial]\npath = "/dev/serial/by-id/x"\nchannels = ["a", "b"]\n[state]\ndir = "/tmp/x"\n'
    )
    assert load_config(path).serial.channels == ("a", "b")
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "missing.toml")
    path.write_text("[device\n")
    with pytest.raises(ConfigError):
        load_config(path)


def test_token_file_must_be_private_and_hold_one_token(tmp_path):
    c = parse_config(cfg(backend__url="https://api.example.com", backend__credential_file=str(tmp_path / "t")))
    assert read_token(parse_config(GOOD)) is None
    token_file = tmp_path / "t"
    token_file.write_text("s3cret-token\n")
    os.chmod(token_file, 0o644)
    with pytest.raises(ConfigError, match="must not be readable") as exc:
        read_token(c)
    assert "s3cret" not in str(exc.value)  # the secret never appears in an error
    os.chmod(token_file, 0o600)
    assert read_token(c) == "s3cret-token"
    token_file.write_text("two tokens\n")
    with pytest.raises(ConfigError, match="single token"):
        read_token(c)
    token_file.write_text("\n")
    with pytest.raises(ConfigError, match="single token"):
        read_token(c)
    token_file.unlink()
    with pytest.raises(ConfigError, match="cannot read credential"):
        read_token(c)


def test_the_alarm_is_off_by_default_and_pins_are_validated():
    assert parse_config(GOOD).alarm.enabled is False
    c = parse_config(cfg(alarm__enabled=True, alarm__led_pin=17, alarm__buzzer_pin=27, alarm__max_ms=5000))
    assert (c.alarm.enabled, c.alarm.led_pin, c.alarm.buzzer_pin, c.alarm.max_ms) == (True, 17, 27, 5000)
    assert parse_config(cfg(alarm__enabled=True)).alarm.buzzer_pin is None  # the buzzer is an explicit opt-in
    assert parse_config(cfg(alarm__enabled=True, alarm__led_pin=0, alarm__buzzer_pin=27)).alarm.led_pin is None


@pytest.mark.parametrize(
    "changes,needle",
    [
        ({"alarm__led_pin": 1}, "BCM pin"),
        ({"alarm__led_pin": 28}, "BCM pin"),
        ({"alarm__buzzer_pin": 99}, "BCM pin"),
        ({"alarm__led_pin": 17, "alarm__buzzer_pin": 17}, "must differ"),
        ({"alarm__max_ms": 40000}, "max_ms"),
        ({"alarm__max_ms": 20000, "alarm__max_continuous_ms": 10000}, "at least alarm.max_ms"),
        ({"alarm__enabled": True, "alarm__led_pin": 0}, "at least one"),
        ({"alarm__enabled": "yes"}, "wrong type"),
        ({"alarm__enabled": 1}, "wrong type"),
        ({"alarm__led_pin": True}, "wrong type"),
        ({"alarm__pin": 17}, "unknown key"),
    ],
)
def test_invalid_alarm_settings_are_refused(changes, needle):
    with pytest.raises(ConfigError, match=needle):
        parse_config(cfg(**changes))


def test_a_replay_image_needs_a_replay_session_and_excludes_a_real_camera():
    c = parse_config(cfg(session__mode="replay", camera__replay_image="/srv/scene.jpg"))
    assert c.camera.replay_image == "/srv/scene.jpg" and c.camera.serial is None
    with pytest.raises(ConfigError, match="needs session.mode"):
        parse_config(cfg(session__mode="demo", camera__replay_image="/srv/scene.jpg"))  # a file must never stand in for a live camera
    with pytest.raises(ConfigError, match="alternatives"):
        parse_config(cfg(session__mode="replay", camera__replay_image="/srv/scene.jpg", camera__serial="308643024550"))


def test_camera_tuning_is_optional_bounded_and_typed():
    c = parse_config(cfg(camera__serial="308643024550"))
    assert (c.camera.width, c.camera.fps, c.camera.dynamic_framerate, c.camera.enhance) == (None, None, None, None)
    c = parse_config(cfg(camera__serial="308643024550", camera__width=640, camera__height=480, camera__fps=30,
                         camera__dynamic_framerate=False, camera__enhance=False))  # fmt: skip
    assert (c.camera.width, c.camera.height, c.camera.fps, c.camera.dynamic_framerate, c.camera.enhance) == (640, 480, 30, False, False)
    for key, value in (("fps", 0), ("fps", 61), ("width", 100), ("height", 2000), ("enhance", 1), ("dynamic_framerate", "yes"), ("fps", 1.5)):
        with pytest.raises(ConfigError):
            parse_config(cfg(camera__serial="308643024550", **{f"camera__{key}": value}))


def test_the_shipped_example_config_stays_valid():
    from pathlib import Path

    example = Path(__file__).resolve().parents[2] / "rpi_agents/deploy/edge.example.toml"
    c = load_config(example)
    assert c.device.input_kind == "stand_in" and len(c.serial.channels) == 7 and c.camera.serial == "308643024550"
    assert c.serial.path.startswith("/dev/serial/by-id/") and c.images.local_dir is None
    assert c.alarm.enabled is False  # the shipped example never energises a pin by itself


def test_module_imports_without_site_packages():
    subprocess.run([sys.executable, "-S", "-c", "import rpi_agents.agent.config"], check=True)
