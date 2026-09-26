import io
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import pytest

from rpi_agents.agent.camera import (
    CameraConfig,
    CameraDisconnected,
    CameraError,
    CameraOversize,
    ReplayCamera,
    UvcCamera,
    find_v4l2_device,
    grab_v4l2_frame,
    shadow_lift_table,
    tune_v4l2,
    yuyv_to_jpeg,
)

SERIAL = "308643024550"
NODES = ["/dev/video0", "/dev/video2", "/dev/video4", "/dev/video7"]
PROPS = {
    "/dev/video0": SERIAL,
    "/dev/video2": SERIAL,
    "/dev/video4": SERIAL,
    "/dev/video7": "OTHER",
}
FORMATS = {
    "/dev/video0": "\t[0]: 'Z16 ' (16-bit Depth)\n",
    "/dev/video2": "\t[0]: 'GREY' (8-bit Greyscale)\n\t[1]: 'UYVY' (UYVY 4:2:2)\n",
    "/dev/video4": "\t[0]: 'YUYV' (YUYV 4:2:2)\n",
    "/dev/video7": "\t[0]: 'YUYV' (YUYV 4:2:2)\n",
}


def fake_sh(props=PROPS, formats=FORMATS):
    def sh(argv, timeout_s):
        node = argv[argv.index("-n") + 1] if argv[0] == "udevadm" else argv[2]
        if argv[0] == "udevadm":
            return f"ID_V4L_CAPABILITIES=:capture:\nID_SERIAL_SHORT={props[node]}\n" if node in props else ""
        return formats.get(node, "")

    return sh


def find(serial=SERIAL, **kw):
    return find_v4l2_device(serial, "YUYV", sh=kw.pop("sh", fake_sh()), nodes=lambda: NODES)


def test_find_picks_the_colour_node_of_the_right_camera():
    assert find() == "/dev/video4"  # depth (Z16) and infrared (UYVY/GREY) offer no YUYV


def test_find_reports_missing_camera_as_disconnected():
    with pytest.raises(CameraDisconnected, match="OTHERSERIAL"):
        find("OTHERSERIAL")
    with pytest.raises(CameraDisconnected):
        find_v4l2_device(SERIAL, "YUYV", sh=lambda argv, t: "", nodes=lambda: NODES)


def test_find_refuses_ambiguous_match():
    formats = FORMATS | {"/dev/video2": "\t[0]: 'YUYV' (YUYV 4:2:2)\n"}
    with pytest.raises(CameraError, match="several nodes"):
        find(sh=fake_sh(formats=formats))


def python_stream(script: str):
    return (sys.executable, "-c", script)


def test_grab_returns_only_the_last_frame():
    script = "import sys; [sys.stdout.buffer.write(bytes([i]) * 16) for i in (1, 2, 3)]"
    raw = grab_v4l2_frame("/dev/x", 4, 2, "YUYV", 3, 5.0, command=python_stream(script))
    assert raw == bytes([3]) * 16


def test_grab_stream_ending_early_is_a_disconnect():
    script = "import sys; sys.stdout.buffer.write(bytes(24))"  # one and a half frames
    with pytest.raises(CameraDisconnected, match="1 of 3"):
        grab_v4l2_frame("/dev/x", 4, 2, "YUYV", 3, 5.0, command=python_stream(script))


def test_grab_stalled_stream_times_out():
    script = "import time; time.sleep(30)"
    with pytest.raises(CameraDisconnected, match="no frame within"):
        grab_v4l2_frame("/dev/x", 4, 2, "YUYV", 1, 0.3, command=python_stream(script))


def test_grab_missing_tool_is_a_disconnect():
    with pytest.raises(CameraDisconnected, match="cannot start"):
        grab_v4l2_frame("/dev/x", 4, 2, "YUYV", 1, 1.0, command=("/nonexistent/v4l2-ctl",))


@pytest.mark.parametrize(
    "y,cb,cr,expected",
    [
        (16, 128, 128, (0, 0, 0)),  # limited-range black must become full-range black
        (235, 128, 128, (255, 255, 255)),
        (81, 90, 240, (255, 0, 0)),  # a swapped U/V would turn red into blue
        (41, 240, 110, (0, 0, 255)),
    ],
)
def test_yuyv_conversion_range_and_channel_order(y, cb, cr, expected):
    pil = pytest.importorskip("PIL.Image")
    width, height = 16, 16
    raw = bytes([y, cb, y, cr]) * (width // 2 * height)
    jpeg = yuyv_to_jpeg(raw, width, height, 90)
    assert jpeg[:2] == b"\xff\xd8" and jpeg[-2:] == b"\xff\xd9"
    pixel = pil.open(io.BytesIO(jpeg)).convert("RGB").getpixel((8, 8))
    assert all(abs(got - want) <= 14 for got, want in zip(pixel, expected)), pixel


def test_yuyv_conversion_rejects_wrong_frame_size():
    pytest.importorskip("PIL.Image")
    with pytest.raises(CameraError, match="expected 512"):
        yuyv_to_jpeg(bytes(100), 16, 16, 80)


NOW = datetime(2026, 9, 24, 12, 0, 0, 123456, tzinfo=timezone(timedelta(hours=2)))


def camera(sizes, *, find=lambda s, f: "/dev/video4", grab=None, cfg=None):
    seen = iter(sizes)
    calls = {"grab": [], "quality": []}

    def default_grab(*args, fps=None):
        calls["grab"].append(args)
        calls.setdefault("fps", []).append(fps)
        return b"raw"

    def encode(raw, w, h, q):
        calls["quality"].append(q)
        return b"j" * next(seen)

    calls["tune"] = []
    cam = UvcCamera(cfg or CameraConfig(serial=SERIAL), find=find, grab=grab or default_grab, encode=encode,
                    tune=lambda device, controls: calls["tune"].append((device, dict(controls))), clock=lambda: NOW)  # fmt: skip
    return cam, calls


def test_capture_returns_utc_timestamp_and_uses_configured_stream():
    cam, calls = camera([100])
    image = cam.capture(max_bytes=1000)
    assert len(image.jpeg) == 100
    assert image.captured_at == "2026-09-24T10:00:00.123Z"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", image.captured_at)
    assert calls["grab"] == [("/dev/video4", 1280, 720, "YUYV", 21, 12.0)]  # 20 warmup + the kept frame
    assert calls["fps"] == [15]


def test_capture_lowers_quality_until_it_fits():
    cam, calls = camera([900, 700, 400, 100])
    assert len(cam.capture(max_bytes=500).jpeg) == 400
    assert calls["quality"] == [85, 70, 55]


def test_capture_oversize_raises_instead_of_truncating():
    cam, calls = camera([900, 800, 700, 600, 550])
    with pytest.raises(CameraOversize, match="550 bytes, limit 500"):
        cam.capture(max_bytes=500)
    assert calls["quality"] == [85, 70, 55, 40, 30]


def test_capture_propagates_disconnect_and_never_returns_a_stale_image():
    def gone(serial, fmt):
        raise CameraDisconnected("unplugged")

    cam, _ = camera([1], find=gone)
    with pytest.raises(CameraDisconnected):
        cam.capture(max_bytes=1000)


def test_capture_rejects_non_positive_limit():
    cam, _ = camera([1])
    with pytest.raises(ValueError):
        cam.capture(max_bytes=0)


@pytest.mark.parametrize(
    "kwargs", [{"serial": ""}, {"serial": "x", "width": 641}, {"serial": "x", "pixel_format": "MJPG"},
               {"serial": "x", "timeout_s": 0}, {"serial": "x", "jpeg_qualities": ()},
               {"serial": "x", "fps": 0}, {"serial": "x", "fps": 61}],
)  # fmt: skip
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        CameraConfig(**kwargs)


def test_module_imports_without_site_packages():
    subprocess.run([sys.executable, "-S", "-c", "import rpi_agents.agent.camera"], check=True)


# ------------------------------------------------------------------------------------ replay camera

JPEG_BYTES = b"\xff\xd8" + b"scene" * 10 + b"\xff\xd9"


def test_replay_camera_serves_the_file_with_a_utc_capture_time(tmp_path):
    path = tmp_path / "scene.jpg"
    path.write_bytes(JPEG_BYTES)
    now = datetime(2026, 9, 24, 12, 0, 0, 123000, tzinfo=timezone.utc)
    camera = ReplayCamera(str(path), clock=lambda: now)
    path.write_bytes(b"changed after start")  # read once at start-up: a running service does not follow the file
    image = camera.capture(max_bytes=10_000)
    assert image.jpeg == JPEG_BYTES and image.captured_at == "2026-09-24T12:00:00.123Z"


def test_replay_camera_refuses_a_missing_file_a_non_jpeg_and_an_oversize_frame(tmp_path):
    with pytest.raises(CameraError, match="cannot read"):
        ReplayCamera(str(tmp_path / "missing.jpg"))
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"GIF89a-not-a-jpeg")
    with pytest.raises(CameraError, match="not a JPEG"):
        ReplayCamera(str(bad))
    ok = tmp_path / "ok.jpg"
    ok.write_bytes(JPEG_BYTES)
    camera = ReplayCamera(str(ok))
    with pytest.raises(CameraOversize):
        camera.capture(max_bytes=len(JPEG_BYTES) - 1)
    with pytest.raises(ValueError):
        camera.capture(max_bytes=0)
    assert camera.capture(max_bytes=len(JPEG_BYTES)).jpeg == JPEG_BYTES


# ------------------------------------------------------------------------------------ image quality


def test_capture_states_the_dynamic_framerate_control_every_time_in_both_directions():
    cam, calls = camera([100, 100], cfg=CameraConfig(serial=SERIAL, dynamic_framerate=True))
    cam.capture(max_bytes=1000)
    assert calls["tune"] == [("/dev/video4", {"exposure_dynamic_framerate": 1})]
    cam, calls = camera([100], cfg=CameraConfig(serial=SERIAL, dynamic_framerate=False))
    cam.capture(max_bytes=1000)
    assert calls["tune"] == [("/dev/video4", {"exposure_dynamic_framerate": 0})]  # never left over from an earlier setting


def test_grab_passes_the_frame_rate_to_the_stream_command_only_when_given(tmp_path):
    script = tmp_path / "fake-v4l2"
    script.write_text("#!/bin/sh\necho \"$@\" > " + str(tmp_path / "argv") + "\nhead -c 8 /dev/zero\n")
    script.chmod(0o755)
    grab_v4l2_frame("/dev/video4", 2, 2, "YUYV", 1, 5.0, command=(str(script),), fps=15)
    assert "--set-parm=15" in (tmp_path / "argv").read_text()
    grab_v4l2_frame("/dev/video4", 2, 2, "YUYV", 1, 5.0, command=(str(script),))
    assert "--set-parm" not in (tmp_path / "argv").read_text()


def test_a_control_the_camera_refuses_is_logged_and_the_others_still_apply(caplog):
    calls = []

    def run(argv, **kw):
        calls.append(argv[-1])
        code = 255 if argv[-1].startswith("gain") else 0
        return subprocess.CompletedProcess(argv, code, "", "gain: Permission denied")

    with caplog.at_level("WARNING", logger="snn_edge.camera"):
        tune_v4l2("/dev/video4", {"gain": 70, "exposure_dynamic_framerate": 1}, run=run)
    assert calls == ["gain=70", "exposure_dynamic_framerate=1"]
    assert [r.getMessage() for r in caplog.records] == ["camera control gain=70 not applied: gain: Permission denied"]


def test_a_missing_v4l2_tool_is_logged_not_raised(caplog):
    def run(argv, **kw):
        raise FileNotFoundError("v4l2-ctl")

    with caplog.at_level("WARNING", logger="snn_edge.camera"):
        tune_v4l2("/dev/video4", {"exposure_dynamic_framerate": 1}, run=run)
    assert "FileNotFoundError" in caplog.text


def test_the_shadow_lift_only_touches_dark_frames_and_never_over_lifts():
    assert shadow_lift_table(90.0) is None and shadow_lift_table(200.0) is None  # bright enough: untouched
    table = shadow_lift_table(40.0)
    assert table[0] == 0 and table[255] == 255 and all(a <= b for a, b in zip(table, table[1:]))  # monotonic, ends fixed
    assert table[40] > 40 and abs(table[40] - 105) < 25  # the mean moves towards the target
    darkest = shadow_lift_table(0.0)
    assert darkest[20] <= round(255 * (20 / 255) ** 0.5) + 1  # gamma is floored at 0.5: black does not become grey noise


def _yuyv(width, height, y, u=128, v=128, speckles=()):
    raw = bytearray([y, u, y, v] * (width * height // 2))
    for x, row in speckles:  # a coloured speck: one chroma pair far from grey
        pair = (row * (width // 2) + x // 2) * 4
        raw[pair + 1], raw[pair + 3] = 40, 60
    return bytes(raw)


def _decode(jpeg):
    from PIL import Image

    return Image.open(io.BytesIO(jpeg)).convert("RGB")


def test_enhance_removes_isolated_colour_speckle_but_the_plain_conversion_keeps_it():
    pytest.importorskip("PIL")
    raw = _yuyv(64, 64, 100, speckles=[(20, 20), (40, 30), (10, 50)])
    plain, clean = _decode(yuyv_to_jpeg(raw, 64, 64, 95)), _decode(yuyv_to_jpeg(raw, 64, 64, 95, enhance=True))
    spread = lambda im: max(max(p) - min(p) for p in (im.getpixel((20, 20)), im.getpixel((40, 30)), im.getpixel((10, 50))))  # noqa: E731
    assert spread(plain) > 40  # the speck is a visible colour
    assert spread(clean) < 12  # and is gone, leaving grey


def test_enhance_lifts_a_dark_frame_and_leaves_a_bright_one_alone():
    pytest.importorskip("PIL")
    from PIL import ImageStat

    dark, bright = _yuyv(64, 64, 40), _yuyv(64, 64, 180)
    mean = lambda raw, enhance: ImageStat.Stat(_decode(yuyv_to_jpeg(raw, 64, 64, 95, enhance=enhance)).convert("L")).mean[0]  # noqa: E731
    assert mean(dark, True) > mean(dark, False) + 25
    assert abs(mean(bright, True) - mean(bright, False)) < 2


def test_enhance_keeps_the_colour_relationship_of_a_dark_frame():
    pytest.importorskip("PIL")
    raw = _yuyv(64, 64, 50, u=110, v=160)  # reddish
    r, g, b = _decode(yuyv_to_jpeg(raw, 64, 64, 95, enhance=True)).getpixel((32, 32))
    assert r > g > b or r > b  # still red-dominant after the lift: only luma was moved
