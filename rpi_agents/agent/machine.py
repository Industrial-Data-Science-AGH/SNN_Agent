"""State machine: WAKE → CAPTURE → PREFILTER → VISION → DECIDE → RESLEEP (P1–P4).

Orchestrates the full cycle: wake context → capture → prefilter → vision → decision → notifications.
Wires together camera, prefilter, vision, actuators, and notifier.
"""

from agent.types import Decision, WakeContext


def run_cycle(wake_context: WakeContext) -> Decision:
    """Run one full alarm-decision cycle from wake to re-sleep.

    State flow:
      WAKE (wake_context captured) → CAPTURE frames → PREFILTER (motion/person) →
      VISION (Gemini or failsafe) → DECIDE (rule-based) → NOTIFY + ACTUATE →
      RESLEEP/COOLDOWN

    Args:
        wake_context: Captured at wake time; includes trigger status and timestamps.

    Returns:
        Final Decision (alarm: bool, reason: str).

    Raises:
        NotImplementedError: State-machine phases P1–P4 not yet implemented.
    """
    del wake_context
    raise NotImplementedError("P1–P4: machine.run_cycle()")
