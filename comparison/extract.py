"""Turn dataset/versions/v2.0.0 into cached per-frame features.

One pass over 20.8 hours of audio, parallel over clips, cached so the fitting
and the metric can be re-run in seconds. Nothing here decides anything; it only
makes the same frames available to both sides of the comparison.

    python -m comparison.extract --variant full --out comparison/cache
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from comparison.features import SETS, frame_features

MANIFEST = Path("dataset/versions/v2.0.0/manifest.csv")


def read_manifest(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def _one(job: tuple[str, str]) -> np.ndarray:
    path, variant = job
    try:
        return frame_features(path, variant)
    except Exception:  # a clip that cannot be read is zero frames, never a guess
        return np.zeros((0, SETS[variant].width), dtype=np.float32)


def extract(rows: list[dict], variant: str, out: Path, workers: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        subset = [r for r in rows if r["split"] == split]
        jobs = [(r["filepath"], variant) for r in subset]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            blocks = list(pool.map(_one, jobs, chunksize=16))

        keep = [(r, b) for r, b in zip(subset, blocks) if len(b)]
        lengths = np.array([len(b) for _, b in keep], dtype=np.int64)
        np.savez(
            out / f"{variant}-{split}.npz",
            features=np.vstack([b for _, b in keep]),
            lengths=lengths,
            label=np.array([r["label"] == "positive" for r, _ in keep]),
            kind=np.array([r["kind"] for r, _ in keep]),
            group=np.array([r["group_id"] for r, _ in keep]),
            clip=np.array([r["id"] for r, _ in keep]),
        )
        dropped = len(subset) - len(keep)
        print(
            f"{variant} {split}: {len(keep)} clips, {lengths.sum()} frames"
            + (f", {dropped} unreadable and skipped" if dropped else ""),
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(SETS), required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--out", type=Path, default=Path("comparison/cache"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="first N clips per split, for a smoke run")
    args = parser.parse_args(argv)

    rows = read_manifest(args.manifest)
    if args.limit:
        rows = [r for split in ("train", "val", "test") for r in [x for x in rows if x["split"] == split][: args.limit]]
    extract(rows, args.variant, args.out, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
