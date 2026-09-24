"""UVC camera adapter: one bounded JPEG from a device picked by USB serial number.

Implements ports.CameraAdapter without optional dependencies at import time: device discovery and
frame capture shell out to udevadm/v4l2-ctl, and Pillow is imported only when a frame is encoded.
Tested with an Intel RealSense D415 colour stream (YUYV, limited-range BT.601); depth and infrared
nodes are never selected because they do not offer the requested pixel format.

Failure is explicit: a missing or stalled camera raises CameraDisconnected, a frame that cannot be
compressed under the limit raises CameraOversize. Nothing here returns a stale or partial image.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import select
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from typing import Callable, Sequence

from rpi_agents.agent.ports import CameraImage

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
    width: int = 640
    height: int = 480
    pixel_format: str = "YUYV"
    warmup_frames: int = 14  # auto exposure needs a few frames; only the last one is kept
    timeout_s: float = 8.0
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
) -> bytes:
    """Stream `frames` frames and return only the last one; memory stays under two frames."""
    frame_bytes = width * height * 2
    argv = [
        *command, "-d", device,
        f"--set-fmt-video=width={width},height={height},pixelformat={pixel_format}",
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


def _full_range(offset: int, span: int, centre: int) -> list[int]:
    return [min(255, max(0, round((i - offset) * 255 / span + centre))) for i in range(256)]


def yuyv_to_jpeg(raw: bytes, width: int, height: int, quality: int) -> bytes:
    """YUYV 4:2:2 limited-range BT.601 -> full-range JPEG. Needs Pillow (present on Raspberry Pi OS)."""
    try:
        from PIL import Image
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
    out = BytesIO()
    Image.merge("YCbCr", (luma, *chroma)).save(out, "JPEG", quality=quality)
    return out.getvalue()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class UvcCamera:
    """ports.CameraAdapter for a UVC colour camera; collaborators are injectable for tests."""

    def __init__(
        self,
        config: CameraConfig,
        *,
        find: Callable[[str, str], str] = find_v4l2_device,
        grab: Callable[..., bytes] = grab_v4l2_frame,
        encode: Callable[[bytes, int, int, int], bytes] = yuyv_to_jpeg,
        clock: Callable[[], datetime] = _utc_now,
    ):
        self._cfg, self._find, self._grab, self._encode, self._clock = config, find, grab, encode, clock

    def capture(self, *, max_bytes: int) -> CameraImage:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        cfg = self._cfg
        device = self._find(cfg.serial, cfg.pixel_format)
        raw = self._grab(
            device, cfg.width, cfg.height, cfg.pixel_format, cfg.warmup_frames + 1, cfg.timeout_s
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
    args = parser.parse_args()
    started = time.monotonic()
    image = UvcCamera(CameraConfig(serial=args.serial)).capture(max_bytes=args.max_bytes)
    with open(args.out, "wb") as handle:
        handle.write(image.jpeg)
    print(json.dumps({"bytes": len(image.jpeg), "captured_at": image.captured_at,
                      "seconds": round(time.monotonic() - started, 2)}))  # fmt: skip


if __name__ == "__main__":
    main()
