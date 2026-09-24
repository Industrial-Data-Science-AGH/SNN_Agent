import pytest

pytest.importorskip("PIL")

from rpi_agents.agent.camera import ReplayCamera  # noqa: E402
from rpi_agents.cloud.infra.scenarios import synthetic_intruder as scene  # noqa: E402


def draw(tmp_path, *flags):
    out = tmp_path / "scene.jpg"
    assert scene.main([str(out), *flags]) == 0
    return out.read_bytes()


def test_the_scene_is_a_deterministic_valid_jpeg_the_replay_camera_accepts(tmp_path):
    first = draw(tmp_path)
    assert first.startswith(b"\xff\xd8") and first.endswith(b"\xff\xd9") and first == draw(tmp_path)
    assert ReplayCamera(str(tmp_path / "scene.jpg")).capture(max_bytes=len(first)).jpeg == first


def test_the_variants_differ_so_a_missing_person_or_missing_glass_is_a_different_picture(tmp_path):
    full, no_person, no_glass = draw(tmp_path), draw(tmp_path, "--no-person"), draw(tmp_path, "--no-glass")
    assert len({full, no_person, no_glass}) == 3
