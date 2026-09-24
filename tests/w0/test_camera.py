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
    UvcCamera,
    find_v4l2_device,
    grab_v4l2_frame,
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

    def default_grab(*args):
        calls["grab"].append(args)
        return b"raw"

    def encode(raw, w, h, q):
        calls["quality"].append(q)
        return b"j" * next(seen)

    cam = UvcCamera(cfg or CameraConfig(serial=SERIAL), find=find, grab=grab or default_grab, encode=encode,
                    clock=lambda: NOW)  # fmt: skip
    return cam, calls


def test_capture_returns_utc_timestamp_and_uses_configured_stream():
    cam, calls = camera([100])
    image = cam.capture(max_bytes=1000)
    assert len(image.jpeg) == 100
    assert image.captured_at == "2026-09-24T10:00:00.123Z"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", image.captured_at)
    assert calls["grab"] == [("/dev/video4", 640, 480, "YUYV", 15, 8.0)]  # 14 warmup + the kept frame


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
               {"serial": "x", "timeout_s": 0}, {"serial": "x", "jpeg_qualities": ()}],
)  # fmt: skip
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        CameraConfig(**kwargs)


def test_module_imports_without_site_packages():
    subprocess.run([sys.executable, "-S", "-c", "import rpi_agents.agent.camera"], check=True)
