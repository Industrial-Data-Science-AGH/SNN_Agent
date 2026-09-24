"""JPEG inspection for uploaded images. Standard library only.

A MIME type says nothing about a file, so the backend reads the JPEG structure itself: start and end
markers, a frame header with sane dimensions and 8-bit precision, well-formed segment lengths, and a size
limit. It does not decode pixels; it decides whether the bytes are plausibly a real JPEG of an allowed size
before they are stored or shown to a model.
"""

from __future__ import annotations

from dataclasses import dataclass

MAX_IMAGE_BYTES = 1_048_576  # architecture: JPEG up to 1 MiB
MIN_SIDE, MAX_SIDE = 16, 8192
_SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
_STANDALONE = {0x01, 0xD8, 0xD9, *range(0xD0, 0xD8)}


class InvalidImage(ValueError):
    """The bytes are not an acceptable JPEG. `code` is stable for API errors."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class JpegInfo:
    width: int
    height: int
    components: int


def inspect_jpeg(data: bytes, *, max_bytes: int = MAX_IMAGE_BYTES) -> JpegInfo:
    if len(data) > max_bytes:
        raise InvalidImage("IMAGE_TOO_LARGE", f"image is {len(data)} bytes, limit {max_bytes}")
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        raise InvalidImage("NOT_JPEG", "missing JPEG start marker")
    if data[-2:] != b"\xff\xd9":
        raise InvalidImage("TRUNCATED_JPEG", "missing JPEG end marker")
    pos, info = 2, None
    while pos < len(data) - 1:
        if data[pos] != 0xFF:
            raise InvalidImage("MALFORMED_JPEG", "expected a marker")
        while pos < len(data) and data[pos] == 0xFF:  # fill bytes
            pos += 1
        if pos >= len(data):
            break
        marker, pos = data[pos], pos + 1
        if marker in _STANDALONE:
            continue
        if pos + 2 > len(data):
            raise InvalidImage("MALFORMED_JPEG", "segment header cut off")
        length = int.from_bytes(data[pos : pos + 2], "big")
        if length < 2 or pos + length > len(data):
            raise InvalidImage("MALFORMED_JPEG", "segment length out of range")
        if marker in _SOF:
            if info is not None:
                raise InvalidImage("MALFORMED_JPEG", "more than one frame header")
            info = _frame(data[pos + 2 : pos + length])
        elif marker == 0xDA:  # start of scan: the entropy-coded data follows, which we do not parse
            if info is None:
                raise InvalidImage("MALFORMED_JPEG", "scan before frame header")
            return info
        pos += length
    raise InvalidImage("MALFORMED_JPEG", "no scan data")


def _frame(body: bytes) -> JpegInfo:
    if len(body) < 6:
        raise InvalidImage("MALFORMED_JPEG", "frame header too short")
    precision, height, width, components = body[0], int.from_bytes(body[1:3], "big"), int.from_bytes(body[3:5], "big"), body[5]
    if precision != 8:
        raise InvalidImage("UNSUPPORTED_JPEG", f"{precision}-bit JPEG is not accepted")
    if components not in (1, 3):
        raise InvalidImage("UNSUPPORTED_JPEG", f"{components} colour components")
    if not (MIN_SIDE <= width <= MAX_SIDE and MIN_SIDE <= height <= MAX_SIDE):
        raise InvalidImage("BAD_DIMENSIONS", f"{width}x{height} is outside {MIN_SIDE}..{MAX_SIDE}")
    return JpegInfo(width, height, components)
