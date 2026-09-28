import copy
import json
from pathlib import Path

import pytest

from contracts.validation import content_hash

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def lui8() -> dict:
    """The network we actually trained: 7 channels, 8 boards, 7-4-3-1, fan-in 3."""
    return json.loads((FIXTURES / "lui8-v2-manifest.json").read_text(encoding="utf-8"))


@pytest.fixture
def mutate():
    """Deep-copy a manifest, apply a change, and re-seal the encoder hash.

    Re-sealing matters: the contract requires encoder_hash == hash(profile), so a
    test that edits the profile would otherwise fail on the hash rather than on
    the thing it means to exercise.
    """

    def _mutate(manifest: dict, change) -> dict:
        copied = copy.deepcopy(manifest)
        change(copied)
        copied["encoder_hash"] = content_hash(copied["encoder_profile"])
        return copied

    return _mutate


@pytest.fixture
def channels(lui8) -> list[str]:
    """Encoder channel names in index order."""
    return [c["channel"] for c in sorted(lui8["encoder_profile"]["channel_map"], key=lambda c: c["index"])]


@pytest.fixture
def make_batch(lui8):
    """Build a SpikeBatch for one window of a frame-indexed stream.

    ``stream`` is a list of per-frame sets of channel names. The batch covers
    frames ``[first, first + count)`` of it and is laid on the same grid the
    device uses, so batches tile the timeline exactly.
    """
    profile = lui8["encoder_profile"]
    fs, hop = profile["sample_rate_hz"], profile["hop_samples"]

    def grid(hop_index: int) -> int:
        """The device's own timeline: rpi_agents/agent/batching.py:_grid.

        Copied deliberately rather than imported, so that a change on either
        side shows up as a failing test instead of tracking silently. Using
        dt_us here instead was the bug that made every real batch `invalid`:
        192/19231 is about 9984 us per hop and dt_us is 10000.
        """
        return (hop_index * hop * 1_000_000 + fs // 2) // fs

    def _make(stream, first: int, count: int, *, seq: int = 0, epoch: int = 1, origin_us: int = 0) -> dict:
        spikes = [
            {"dt_us": grid(frame) - grid(first), "channel": channel}
            for frame in range(first, first + count)
            for channel in sorted(stream[frame])
        ]
        return {
            "schema_version": "1.0",
            "request_id": f"r{seq}",
            "device_id": "dev1",
            "session_id": "sess1",
            "epoch": epoch,
            "boot_id": "boot1",
            "batch_seq": seq,
            "encoder_hash": lui8["encoder_hash"],
            "source_start_us": origin_us + grid(first),
            "source_end_us": origin_us + grid(first + count),
            "spikes": spikes,
            "quality": {"dropped_events": 0, "adc_clipped": False},
        }

    return _make


@pytest.fixture
def stream(channels):
    """A deterministic, reasonably dense pseudo-random channel stream."""
    import random

    rng = random.Random(20260925)
    return [{c for c in channels if rng.random() < 0.25} for _ in range(400)]


@pytest.fixture
def grid(lui8):
    """The device's hop timeline, for tests that need an absolute source time."""
    profile = lui8["encoder_profile"]
    fs, hop = profile["sample_rate_hz"], profile["hop_samples"]
    return lambda hop_index: (hop_index * hop * 1_000_000 + fs // 2) // fs
