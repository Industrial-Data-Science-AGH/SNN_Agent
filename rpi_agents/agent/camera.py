"""UVC camera adapter: one bounded JPEG from a device picked by USB serial number.

Implements ports.CameraAdapter without optional dependencies at import time: device discovery and
frame capture shell out to udevadm/v4l2-ctl, and Pillow is imported only when a frame is encoded.

Image quality in poor light (why the defaults are what they are): at a fixed 30 fps the sensor cannot expose longer
than about 33 ms, so a dim room comes out dark, noisy and with a strong colour cast, especially when auto exposure and
white balance are read after less than half a second. So the stream runs at a modest frame rate with the camera's
dynamic frame rate on (exposure may lengthen, bounded by 1/fps), settles for about a second and a half, and one frame is
kept. Frames are deliberately NOT averaged: that would smear anything that moves, and a moving person is the point.
Colour speckle (chroma noise) is removed per frame and a dark frame gets a gentle shadow lift; both are off in
`enhance=False`.
Tested with an Intel RealSense D415 colour stream (YUYV, limited-range BT.601); depth and infrared
nodes are never selected because they do not offer the requested pixel format.

Failure is explicit: a missing or stalled camera raises CameraDisconnected, a frame that cannot be
compressed under the limit raises CameraOversize. Nothing here returns a stale or partial image.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import re
import select
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from typing import Callable, Mapping, Sequence

from rpi_agents.agent.ports import CameraImage

log = logging.getLogger("snn_edge.camera")
_FOURCC = re.compile(r"^\s*\[\d+\]:\s+'(.{4})'", re.MULTILINE)


class CameraError(RuntimeError):
    """Base class for capture failures."""


class CameraDisconnected(CameraError):
    """Device missing, ambiguous, stalled or the stream ended early."""


class CameraOversize(CameraError):
    """The frame does not fit max_bytes even at the lowest configured JPEG quality."""


@dataclass(frozen=True)
class CameraConfig:
    serial: str
    width: int = 1280
    height: int = 720
    pixel_format: str = "YUYV"
    fps: int = 15  # the D415 colour sensor offers 60/30/15/6; with dynamic_framerate this is the ceiling in good light
    dynamic_framerate: bool = True  # let exposure lengthen (down to a lower fps) when it is dark
    warmup_frames: int = 20  # auto exposure and white balance need about a second and a half; only the last frame is kept
    enhance: bool = True  # chroma denoise and a shadow lift for dark frames
    timeout_s: float = 12.0
    jpeg_qualities: tuple[int, ...] = (85, 70, 55, 40, 30)

    def __post_init__(self) -> None:
        if not self.serial:
            raise ValueError("serial must be set: the camera is selected by USB serial number")
        if self.pixel_format != "YUYV":
            raise ValueError("only YUYV capture is implemented")
        if self.width <= 0 or self.height <= 0 or self.width % 2:
            raise ValueError("width must be even and positive, height positive")
        if self.warmup_frames < 0 or self.timeout_s <= 0 or not self.jpeg_qualities:
            raise ValueError("invalid warmup, timeout or quality ladder")
        if not 1 <= self.fps <= 60:
            raise ValueError("fps must be 1..60")


def _sh(argv: Sequence[str], timeout_s: float) -> str:
    """Run a helper; any failure yields empty output so discovery can skip that node."""
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return done.stdout if done.returncode == 0 else ""


def _video_nodes() -> list[str]:
    return sorted(glob.glob("/dev/video*"), key=lambda p: int(re.sub(r"\D", "", p) or 0))


def find_v4l2_device(
    serial: str,
    pixel_format: str,
    *,
    sh: Callable[[Sequence[str], float], str] = _sh,
    nodes: Callable[[], list[str]] = _video_nodes,
) -> str:
    """The one /dev/videoN owned by this USB serial that offers `pixel_format`."""
    matches = []
    for node in nodes():
        props = sh(["udevadm", "info", "-q", "property", "-n", node], 5.0).splitlines()
        if f"ID_SERIAL_SHORT={serial}" not in props:
            continue
        formats = [f.strip() for f in _FOURCC.findall(sh(["v4l2-ctl", "-d", node, "--list-formats"], 5.0))]
        if pixel_format in formats:
            matches.append(node)
    if not matches:
        raise CameraDisconnected(f"no {pixel_format} video node for camera serial {serial}")
    if len(matches) > 1:
        raise CameraError(f"camera serial {serial} matches several nodes: {', '.join(matches)}")
    return matches[0]


def grab_v4l2_frame(
    device: str,
    width: int,
    height: int,
    pixel_format: str,
    frames: int,
    timeout_s: float,
    *,
    command: Sequence[str] = ("v4l2-ctl",),
    fps: int | None = None,
) -> bytes:
    """Stream `frames` frames and return only the last one; memory stays under two frames."""
    frame_bytes = width * height * 2
    argv = [
        *command, "-d", device,
        f"--set-fmt-video=width={width},height={height},pixelformat={pixel_format}",
        *([f"--set-parm={fps}"] if fps else []),
        "--stream-mmap", f"--stream-count={frames}", "--stream-to=-",
    ]  # fmt: skip
    try:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except OSError as exc:
        raise CameraDisconnected(f"cannot start {argv[0]}: {exc}") from exc
    deadline = time.monotonic() + timeout_s
    buf, last, got = bytearray(), b"", 0
    try:
        fd = proc.stdout.fileno()
        while got < frames:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
                raise CameraDisconnected(f"no frame within {timeout_s} s ({got} of {frames} received)")
            chunk = os.read(fd, 1 << 16)
            if not chunk:
                raise CameraDisconnected(f"stream ended after {got} of {frames} frames (rc={proc.poll()})")
            buf += chunk
            while len(buf) >= frame_bytes and got < frames:
                last = bytes(buf[:frame_bytes])
                del buf[:frame_bytes]
                got += 1
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        proc.stdout.close()
    return last


def tune_v4l2(
    device: str, controls: Mapping[str, int], *, run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> None:
    """Set V4L2 controls one by one. Best effort: a control this camera lacks or refuses must not stop a capture."""
    for name, value in controls.items():
        try:
            done = run(["v4l2-ctl", "-d", device, "--set-ctrl", f"{name}={value}"], capture_output=True, text=True, timeout=5.0, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            log.warning("camera control %s=%s not applied: %s", name, value, type(exc).__name__)
            continue
        if done.returncode != 0:
            log.warning("camera control %s=%s not applied: %s", name, value, (done.stderr or "").strip()[:80] or f"rc={done.returncode}")


def _full_range(offset: int, span: int, centre: int) -> list[int]:
    return [min(255, max(0, round((i - offset) * 255 / span + centre))) for i in range(256)]


DARK_LUMA = 90.0  # mean luma (0..255) under which a frame is lifted
LIFT_TARGET = 105.0
MIN_GAMMA = 0.5  # the lift never exceeds this, so a black frame does not become a noisy grey one


def shadow_lift_table(mean_luma: float) -> list[int] | None:
    """A gamma table that moves a dark frame's mean towards LIFT_TARGET, or None when the frame is bright enough."""
    if mean_luma >= DARK_LUMA:
        return None
    gamma = min(1.0, max(MIN_GAMMA, math.log(LIFT_TARGET / 255) / math.log(max(mean_luma, 1.0) / 255)))
    return [round(255 * (i / 255) ** gamma) for i in range(256)]


def yuyv_to_jpeg(raw: bytes, width: int, height: int, quality: int, *, enhance: bool = False) -> bytes:
    """YUYV 4:2:2 limited-range BT.601 -> full-range JPEG. Needs Pillow (present on Raspberry Pi OS).

    `enhance`: a median + blur on the two chroma planes removes coloured speckle without touching detail (luma is
    left alone), and a dark frame gets a gentle gamma lift on luma only, so colours keep their relation."""
    try:
        from PIL import Image, ImageFilter, ImageStat
    except ImportError as exc:  # pragma: no cover - environment specific
        raise CameraError("Pillow is required to encode JPEG (apt install python3-pil)") from exc
    if len(raw) != width * height * 2:
        raise CameraError(f"frame is {len(raw)} bytes, expected {width * height * 2}")
    luma = Image.frombytes("L", (width, height), raw[0::2]).point(_full_range(16, 219, 0))
    chroma = [
        Image.frombytes("L", (width // 2, height), raw[offset::4])
        .resize((width, height), Image.BILINEAR)
        .point(_full_range(128, 224, 128))
        for offset in (1, 3)
    ]
    if enhance:
        chroma = [plane.filter(ImageFilter.MedianFilter(5)).filter(ImageFilter.GaussianBlur(1.0)) for plane in chroma]
        table = shadow_lift_table(ImageStat.Stat(luma).mean[0])
        if table is not None:
            luma = luma.point(table)
    out = BytesIO()
    Image.merge("YCbCr", (luma, *chroma)).save(out, "JPEG", quality=quality)
    return out.getvalue()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ReplayCamera:
    """ports.CameraAdapter that returns one fixed JPEG file: a synthetic scene for replay sessions.

    The bridge only builds it for session.mode = "replay", whose images the backend labels `synthetic`, so a file can
    never pass as a live camera. The file is read once, at start-up, so a bad path or a non-JPEG stops the service early."""

    def __init__(self, path: str, *, clock: Callable[[], datetime] = _utc_now):
        try:
            with open(path, "rb") as handle:
                self._jpeg = handle.read()
        except OSError as exc:
            raise CameraError(f"cannot read the replay image: {exc.strerror}") from None
        if not (self._jpeg.startswith(b"\xff\xd8") and self._jpeg.endswith(b"\xff\xd9")):
            raise CameraError("the replay image is not a JPEG")
        self._clock = clock

    def capture(self, *, max_bytes: int) -> CameraImage:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if len(self._jpeg) > max_bytes:
            raise CameraOversize(f"the replay image is {len(self._jpeg)} bytes, limit {max_bytes}")
        stamp = self._clock().astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        return CameraImage(jpeg=self._jpeg, captured_at=stamp)


class UvcCamera:
    """ports.CameraAdapter for a UVC colour camera; collaborators are injectable for tests."""

    def __init__(
        self,
        config: CameraConfig,
        *,
        find: Callable[[str, str], str] = find_v4l2_device,
        grab: Callable[..., bytes] = grab_v4l2_frame,
        encode: Callable[[bytes, int, int, int], bytes] | None = None,
        tune: Callable[[str, Mapping[str, int]], None] = tune_v4l2,
        clock: Callable[[], datetime] = _utc_now,
    ):
        self._cfg, self._find, self._grab, self._tune, self._clock = config, find, grab, tune, clock
        self._encode = encode or (lambda raw, w, h, q: yuyv_to_jpeg(raw, w, h, q, enhance=config.enhance))

    def capture(self, *, max_bytes: int) -> CameraImage:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        cfg = self._cfg
        device = self._find(cfg.serial, cfg.pixel_format)
        self._tune(device, {"exposure_dynamic_framerate": int(cfg.dynamic_framerate)})  # stated every time, never left over
        raw = self._grab(
            device, cfg.width, cfg.height, cfg.pixel_format, cfg.warmup_frames + 1, cfg.timeout_s, fps=cfg.fps
        )
        captured_at = self._clock().astimezone(timezone.utc).isoformat(timespec="milliseconds")
        captured_at = captured_at.replace("+00:00", "Z")
        smallest = 0
        for quality in cfg.jpeg_qualities:
            jpeg = self._encode(raw, cfg.width, cfg.height, quality)
            if len(jpeg) <= max_bytes:
                return CameraImage(jpeg=jpeg, captured_at=captured_at)
            smallest = len(jpeg)
        raise CameraOversize(f"smallest JPEG is {smallest} bytes, limit {max_bytes}")


def main() -> None:
    """Hardware smoke test: capture one frame to a file and print what was captured."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-bytes", type=int, default=500_000)
    parser.add_argument("--no-enhance", action="store_true", help="skip the chroma denoise and shadow lift, for comparison")
    args = parser.parse_args()
    started = time.monotonic()
    image = UvcCamera(CameraConfig(serial=args.serial, enhance=not args.no_enhance)).capture(max_bytes=args.max_bytes)
    with open(args.out, "wb") as handle:
        handle.write(image.jpeg)
    print(json.dumps({"bytes": len(image.jpeg), "captured_at": image.captured_at,
                      "seconds": round(time.monotonic() - started, 2)}))  # fmt: skip


if __name__ == "__main__":
    main()
