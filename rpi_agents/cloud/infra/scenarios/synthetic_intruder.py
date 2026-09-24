"""Draw a synthetic "intruder at a broken window" scene as a JPEG, for the replay scenario of the acceptance run.

    python -m rpi_agents.cloud.infra.scenarios.synthetic_intruder out.jpg [--no-person] [--no-glass]

The scene is a DRAWING, not a photograph and not a test of the vision model's accuracy: it exists so the whole chain
(capture -> upload -> vision -> policy -> alarm -> e-mail) can be exercised with an image whose content is known, in a
session that the backend labels `synthetic`. It is deterministic: the same arguments give the same bytes.
Needs Pillow (a development dependency, not part of the backend image or the edge service).
"""

from __future__ import annotations

import argparse
import math
import random

from PIL import Image, ImageDraw, ImageFilter

W, H = 640, 480


def _room(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle([0, 0, W, H], fill=(38, 40, 48))  # wall
    draw.polygon([(0, 360), (W, 360), (W, H), (0, H)], fill=(66, 58, 50))  # floor
    draw.line([(0, 360), (W, 360)], fill=(28, 26, 24), width=4)  # skirting


def _window(draw: ImageDraw.ImageDraw, *, broken: bool) -> tuple[int, int, int, int]:
    box = (330, 70, 590, 330)
    draw.rectangle(box, fill=(20, 26, 46), outline=(200, 200, 205), width=10)  # night sky and frame
    draw.line([(460, 70), (460, 330)], fill=(200, 200, 205), width=6)
    draw.line([(330, 200), (590, 200)], fill=(200, 200, 205), width=6)
    draw.ellipse([520, 95, 560, 135], fill=(235, 235, 210))  # moon
    if broken:
        rng = random.Random(7)
        cx, cy = 400, 150  # impact point
        for i in range(14):  # radial cracks
            angle = i * (2 * math.pi / 14) + rng.uniform(-0.15, 0.15)
            length = rng.uniform(70, 150)
            draw.line([(cx, cy), (cx + length * math.cos(angle), cy + length * math.sin(angle))], fill=(230, 235, 245), width=2)
        for ring in (30, 60, 95):  # concentric cracks
            pts = [(cx + ring * math.cos(a / 8 * math.pi) + rng.uniform(-5, 5), cy + ring * math.sin(a / 8 * math.pi) + rng.uniform(-5, 5)) for a in range(17)]
            draw.line(pts, fill=(215, 222, 235), width=1)
        draw.polygon([(cx - 22, cy - 14), (cx + 6, cy - 26), (cx + 20, cy + 8), (cx - 10, cy + 24)], fill=(8, 10, 18))  # the hole
    return box


def _shards(draw: ImageDraw.ImageDraw) -> None:
    rng = random.Random(11)
    for _ in range(60):  # broken glass on the floor under the window
        x, y = rng.randint(300, 640), rng.randint(365, 470)
        size = rng.randint(5, 22)
        pts = [(x + rng.randint(-size, size), y + rng.randint(-size // 2, size // 2)) for _ in range(3)]
        draw.polygon(pts, fill=(205, 225, 240), outline=(245, 250, 255))


def _person(draw: ImageDraw.ImageDraw) -> None:
    dark, mask = (14, 14, 18), (24, 24, 30)
    draw.ellipse([182, 118, 250, 196], fill=mask)  # head in a balaclava
    draw.rectangle([198, 150, 236, 162], fill=(215, 175, 140))  # the eye slit
    draw.ellipse([204, 152, 212, 160], fill=(10, 10, 10))
    draw.ellipse([224, 152, 232, 160], fill=(10, 10, 10))
    draw.polygon([(150, 205), (282, 205), (300, 330), (132, 330)], fill=dark)  # torso in a dark jacket
    draw.polygon([(282, 210), (318, 205), (392, 168), (384, 148), (312, 182), (270, 190)], fill=dark)  # arm reaching to the window
    draw.polygon([(150, 210), (120, 205), (100, 290), (128, 296)], fill=dark)  # other arm
    draw.rectangle([352, 132, 396, 146], fill=(120, 120, 128))  # a crowbar in the hand
    draw.polygon([(160, 328), (232, 328), (228, 470), (168, 470)], fill=dark)  # legs
    draw.polygon([(232, 328), (296, 328), (300, 470), (238, 470)], fill=dark)
    draw.ellipse([160, 462, 236, 480], fill=(8, 8, 10))
    draw.ellipse([236, 462, 312, 480], fill=(8, 8, 10))


def render(*, person: bool = True, glass: bool = True) -> Image.Image:
    image = Image.new("RGB", (W, H))
    draw = ImageDraw.Draw(image)
    _room(draw)
    _window(draw, broken=glass)
    if glass:
        _shards(draw)
    if person:
        _person(draw)
    return image.filter(ImageFilter.GaussianBlur(0.8))  # camera-like softness


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output")
    parser.add_argument("--no-person", action="store_true", help="the broken window only")
    parser.add_argument("--no-glass", action="store_true", help="the person only, at an intact window")
    args = parser.parse_args(argv)
    render(person=not args.no_person, glass=not args.no_glass).save(args.output, "JPEG", quality=85)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
