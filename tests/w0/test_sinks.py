import os
import stat
import subprocess
import sys

import pytest

from contracts.validation import fixture
from rpi_agents.agent.api import Outcome, Result
from rpi_agents.agent.commands import SinkUnavailable
from rpi_agents.agent.config import ConfigError, parse_config
from rpi_agents.agent.sinks import BackendImageSink, LocalDirSink

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


class FakeApi:
    def __init__(self, results):
        self.results, self.calls = list(results), []

    def upload_image(self, event_id, index, jpeg, *, sha256, captured_at):
        self.calls.append((event_id, index, jpeg, sha256, captured_at))
        return self.results.pop(0)


def ok(image_id="evt-0"):
    return Result(Outcome.OK, 200, None, {"image_id": image_id})


def test_the_backend_sink_uploads_and_returns_the_image_id_the_backend_assigned():
    import hashlib

    api = FakeApi([ok("evt-0")])
    sink = BackendImageSink(api)
    assert store(sink, command_id="c1") == "evt-0"
    (event_id, index, jpeg, sha, when) = api.calls[0]
    assert (event_id, index, when) == ("e1", 0, "2026-09-24T12:00:00Z") and sha == hashlib.sha256(jpeg).hexdigest()


def test_a_transient_upload_failure_is_retried_and_then_reported_as_an_unavailable_sink():
    flaky = FakeApi([Result(Outcome.RETRY, 503, None, None), ok()])
    assert store(BackendImageSink(flaky, attempts=2)) == "evt-0" and len(flaky.calls) == 2
    down = FakeApi([Result(Outcome.RETRY, None, "TRANSPORT_ERROR", None)] * 2)
    with pytest.raises(SinkUnavailable, match="TRANSPORT_ERROR"):
        store(BackendImageSink(down, attempts=2))
    assert len(down.calls) == 2


def test_a_permanent_refusal_is_not_retried():
    refused = FakeApi([Result(Outcome.PERMANENT, 422, "NOT_JPEG", {"error": {"code": "NOT_JPEG"}})])
    with pytest.raises(SinkUnavailable, match="NOT_JPEG"):
        store(BackendImageSink(refused, attempts=3))
    assert len(refused.calls) == 1


@pytest.mark.parametrize("body", [None, {}, {"image_id": 5}, {"image_id": None}, {"other": "x"}])
def test_a_success_response_without_an_image_id_is_not_trusted(body):
    with pytest.raises(SinkUnavailable):
        store(BackendImageSink(FakeApi([Result(Outcome.OK, 200, None, body)]), attempts=1))


def test_the_backend_sink_needs_at_least_one_attempt():
    with pytest.raises(ValueError):
        BackendImageSink(FakeApi([]), attempts=0)


def test_upload_and_local_directory_are_alternative_sinks():
    assert parse_config(cfg(images={"upload": True})).images.upload is True
    assert parse_config(GOOD).images.upload is False
    with pytest.raises(ConfigError, match="alternatives"):
        parse_config(cfg(images={"upload": True, "local_dir": "/x"}))
    with pytest.raises(ConfigError, match="wrong type"):
        parse_config(cfg(images={"upload": "yes"}))
