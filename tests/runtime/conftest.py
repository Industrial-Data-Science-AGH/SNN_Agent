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
