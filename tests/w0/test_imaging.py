import subprocess
import sys

import pytest

from rpi_agents.cloud.app.imaging import MAX_IMAGE_BYTES, InvalidImage, inspect_jpeg
from tests.w0.fakes import make_jpeg


def test_a_well_formed_jpeg_reports_its_dimensions():
    info = inspect_jpeg(make_jpeg(1920, 1080))
    assert (info.width, info.height, info.components) == (1920, 1080, 3)
    assert inspect_jpeg(make_jpeg(64, 64, components=1)).components == 1


def test_a_real_encoder_output_is_accepted_when_pillow_is_available():
    image = pytest.importorskip("PIL.Image")
    import io

    buffer = io.BytesIO()
    image.new("RGB", (320, 240), (200, 30, 30)).save(buffer, "JPEG", quality=80)
    info = inspect_jpeg(buffer.getvalue())
    assert (info.width, info.height) == (320, 240)


@pytest.mark.parametrize(
    "data,code",
    [
        (b"", "NOT_JPEG"),
        (b"\x89PNG\r\n\x1a\n" + b"x" * 100, "NOT_JPEG"),
        (b"<html>not an image</html>", "NOT_JPEG"),
        (make_jpeg(eoi=False), "TRUNCATED_JPEG"),
        (make_jpeg()[:40], "TRUNCATED_JPEG"),
        (make_jpeg(precision=12), "UNSUPPORTED_JPEG"),
        (make_jpeg(components=4), "UNSUPPORTED_JPEG"),
        (make_jpeg(0, 480), "BAD_DIMENSIONS"),
        (make_jpeg(640, 8), "BAD_DIMENSIONS"),
        (make_jpeg(9000, 480), "BAD_DIMENSIONS"),
        (make_jpeg(65535, 65535), "BAD_DIMENSIONS"),
        (b"\xff\xd8\xff\xda\x00\x02\xff\xd9", "MALFORMED_JPEG"),  # scan before any frame header
        (b"\xff\xd8\xff\xe0\xff\xff" + b"x" * 10 + b"\xff\xd9", "MALFORMED_JPEG"),  # segment longer than the file
        (b"\xff\xd8\xff\xe0\x00\x01\xff\xd9", "MALFORMED_JPEG"),  # length below the minimum of 2
        (b"\xff\xd8junk\xff\xd9", "MALFORMED_JPEG"),
    ],
)
def test_bytes_that_are_not_an_acceptable_jpeg_are_refused_with_a_stable_code(data, code):
    with pytest.raises(InvalidImage) as error:
        inspect_jpeg(data)
    assert error.value.code == code


@pytest.mark.parametrize(
    "data",
    [b"\xff\xd8\xff\xe0\xff\xff" + b"x" * 10 + b"\xff\xd9", b"\xff\xd8\xff\xe0\x00\x01\xff\xd9",
     make_jpeg()[:20] + b"\xff\xe1\x7f\xff" + make_jpeg()[20:]],
)  # fmt: skip
def test_a_segment_length_that_lies_is_named_as_such(data):
    with pytest.raises(InvalidImage, match="segment length"):
        inspect_jpeg(data)


def test_a_second_frame_header_is_refused():
    one = make_jpeg()
    sof = one[one.index(b"\xff\xc0") : one.index(b"\xff\xda")]
    with pytest.raises(InvalidImage, match="more than one"):
        inspect_jpeg(make_jpeg(extra=sof))


def test_the_size_limit_is_enforced_before_anything_else():
    big = make_jpeg(scan=b"\x00" * (MAX_IMAGE_BYTES))
    with pytest.raises(InvalidImage) as error:
        inspect_jpeg(big)
    assert error.value.code == "IMAGE_TOO_LARGE"
    assert inspect_jpeg(make_jpeg(scan=b"\x00" * 1000), max_bytes=2000).width == 640
    with pytest.raises(InvalidImage):
        inspect_jpeg(make_jpeg(scan=b"\x00" * 1000), max_bytes=100)


def test_fill_bytes_and_standalone_markers_before_the_frame_are_tolerated():
    one = make_jpeg()
    padded = one[:2] + b"\xff\xff\xff" + one[2:]
    assert inspect_jpeg(padded).width == 640


def test_no_input_can_make_the_parser_loop_or_crash():
    import random

    rng = random.Random(7)
    for _ in range(3000):
        blob = b"\xff\xd8" + bytes(rng.choice([0xFF, 0xC0, 0xDA, 0x00, 0x02, 0x08, 0xD9, rng.randrange(256)]) for _ in range(rng.randrange(0, 60))) + b"\xff\xd9"
        try:
            inspect_jpeg(blob)
        except InvalidImage:
            pass  # any refusal is fine; only an unexpected exception type or a hang would fail this test


def test_module_imports_without_site_packages():
    subprocess.run([sys.executable, "-S", "-c", "import rpi_agents.cloud.app.imaging"], check=True)
