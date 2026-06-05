"""pytest configuration: make rpi_agents/ importable as 'agent.*'."""

import sys
from pathlib import Path

# Mirror test_forward.py:6 — prepend rpi_agents/ so `from agent import ...` resolves.
sys.path.insert(0, str(Path(__file__).parent.parent))
