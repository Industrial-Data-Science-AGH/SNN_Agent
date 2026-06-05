"""Entrypoint for the Wake-Up AI RPi 5 agent (P1–P4 fill implementation).

Usage:
    python -m agent.main          # production: run one wake cycle
    python -m agent.main --test   # smoke-test: import-check + print banner
"""

import argparse
import sys


_BANNER = """
╔══════════════════════════════════════════╗
║   Wake-Up AI — RPi 5 Agent  v0.1.0      ║
║   State machine: WAKE→CAPTURE→DECIDE    ║
║   [--test mode: stubs only, no hw]      ║
╚══════════════════════════════════════════╝
""".strip()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wake-Up AI agent entrypoint.")
    p.add_argument(
        "--test",
        action="store_true",
        help="Smoke-test: verify imports and print banner, then exit 0.",
    )
    return p.parse_args()


def _smoke_test() -> None:
    """Import every agent module and confirm the package is loadable."""
    import agent.actuators  # noqa: F401
    import agent.camera     # noqa: F401
    import agent.config     # noqa: F401
    import agent.machine    # noqa: F401
    import agent.notifier   # noqa: F401
    import agent.power      # noqa: F401
    import agent.prefilter  # noqa: F401
    import agent.trigger    # noqa: F401
    import agent.types      # noqa: F401
    import agent.vision     # noqa: F401
    print(_BANNER)
    print("\n[--test] All agent modules imported successfully.")


def _configure_logging() -> None:
    import logging
    from agent import config
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _run_production() -> None:
    from agent import actuators, config, machine, power, trigger

    wake = trigger.read_wake_context()
    decision = None
    try:
        decision = machine.run_cycle(wake)
    finally:
        if decision is not None and decision.alarm:
            power.cooldown(config.ALARM_HOLD_S)
        actuators.close()
        power.cooldown(config.COOLDOWN_S)
        power.resleep()


def main() -> None:
    args = _parse_args()

    if args.test:
        _smoke_test()
        sys.exit(0)

    _configure_logging()
    _run_production()
    sys.exit(0)


if __name__ == "__main__":
    main()
