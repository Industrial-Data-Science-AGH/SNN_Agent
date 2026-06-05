"""Power management: cooldown + power-mode-aware re-halt (P1).

Top-level imports remain hardware-free; subprocess and time are imported lazily.
"""

import logging

from agent import config

logger = logging.getLogger(__name__)


def cooldown(seconds: float) -> None:
    """Sleep for cooldown period before re-arming."""
    import time
    time.sleep(seconds)


def resleep() -> None:
    """Put RPi 5 back into low-power halt mode, or no-op in warm/dev mode.

    Reads config.POWER_MODE at call time so tests can monkeypatch it.
    In halt mode, a non-zero exit from `sudo halt` is surfaced (logged + raised)
    rather than swallowed — a Pi that fails to halt must not silently re-arm.
    """
    import subprocess
    if config.POWER_MODE == "halt":
        logger.info("Halting Pi (POWER_MODE=halt)")
        result = subprocess.run(["sudo", "halt"], check=False)
        if result.returncode != 0:
            logger.error("sudo halt failed (returncode=%s); Pi did not halt", result.returncode)
            raise RuntimeError(f"sudo halt failed with returncode {result.returncode}")
    else:
        logger.info("POWER_MODE=%s — skipping halt (warm/dev mode)", config.POWER_MODE)
