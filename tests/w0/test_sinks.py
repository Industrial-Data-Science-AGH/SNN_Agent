import os
import stat
import subprocess
import sys

import pytest

from contracts.validation import fixture
from rpi_agents.agent.commands import SinkUnavailable
from rpi_agents.agent.config import ConfigError, parse_config
from rpi_agents.agent.sinks import LocalDirSink

CREATE = fixture("session-create")
GOOD = {
    "device": {"id": "snn-pi", "input_kind": "stand_in"},
    "backend": {"url": "http://127.0.0.1:8000"},
    "session": {"mode": "demo", "model_hash": CREATE["model_hash"], "encoder_hash": CREATE["encoder_hash"]},
    "serial": {"path": "/dev/serial/by-id/x", "channels": ["a", "b"]},
    "state": {"dir": "/tmp/snn-edge"},
}


def cfg(**sections):
    return GOOD | sections


def store(sink, command_id="c1", index=0, jpeg=b"\xff\xd8x\xff\xd9"):
    return sink.store(event_id="e1", command_id=command_id, index=index, jpeg=jpeg, captured_at="2026-09-24T12:00:00Z")


def test_images_are_written_privately_and_atomically(tmp_path):
    sink = LocalDirSink(str(tmp_path / "images"))
    assert store(sink) == "c1-0"
    path = tmp_path / "images" / "c1-0.jpg"
    assert path.read_bytes() == b"\xff\xd8x\xff\xd9"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / "images").stat().st_mode) == 0o700
    assert [p.name for p in (tmp_path / "images").iterdir()] == ["c1-0.jpg"]  # no temporary file left behind


def test_only_the_newest_images_are_kept(tmp_path):
    sink = LocalDirSink(str(tmp_path), keep=3)
    for i in range(6):
        store(sink, command_id=f"c{i}")
        os.utime(tmp_path / f"c{i}-0.jpg", (1000 + i, 1000 + i))  # make the age order explicit
    store(sink, command_id="c6")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["c4-0.jpg", "c5-0.jpg", "c6-0.jpg"]


def test_unsafe_names_and_io_errors_are_reported_as_an_unavailable_sink(tmp_path):
    sink = LocalDirSink(str(tmp_path))
    with pytest.raises(SinkUnavailable, match="safe file name"):
        store(sink, command_id="../etc/passwd")
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory")
    broken = LocalDirSink(str(tmp_path / "ok"))
    broken._dir = str(blocker)  # writing below a regular file must fail cleanly
    with pytest.raises(SinkUnavailable, match="cannot store"):
        store(broken)
    with pytest.raises(ValueError):
        LocalDirSink(str(tmp_path), keep=0)


def test_the_images_section_is_parsed_and_validated():
    assert parse_config(GOOD).images.local_dir is None
    c = parse_config(cfg(images={"local_dir": "/var/lib/snn-edge/images", "keep": 5}))
    assert (c.images.local_dir, c.images.keep) == ("/var/lib/snn-edge/images", 5)
    with pytest.raises(ConfigError, match="images.keep"):
        parse_config(cfg(images={"keep": 0}))
    with pytest.raises(ConfigError, match="unknown key"):
        parse_config(cfg(images={"dir": "/x"}))


def test_module_imports_without_site_packages():
    subprocess.run([sys.executable, "-S", "-c", "import rpi_agents.agent.sinks"], check=True)
