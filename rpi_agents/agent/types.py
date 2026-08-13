# -*- coding: utf-8 -*-
"""
Stable typed contracts for all rpi_agents phases.

These dataclasses are the single source of truth for data flowing between
modules. Import only from here; never re-define these shapes elsewhere.
"""

from dataclasses import dataclass
from typing import Literal

# PREFILTER CONTRACT


@dataclass(frozen=True)
class PrefilterResult:
    """Result of the local motion / person prefilter (P2)."""

    motion: bool
    person: bool
    escalate: bool
    score: float


# VISION CONTRACT


@dataclass(frozen=True)
class VisionVerdict:
    """Verdict from the Gemini vision stage or the failsafe fallback (P3).

    `window_broken` (2026-07-15 addition, ADR-0017) is a second, independent
    judgment from the same Gemini call: does the frame show visible evidence
    the window/glass itself is broken, as opposed to `is_intrusion` (is
    there a real break-in/intruder). These can differ -- a window can show
    no visible damage yet with a person intruding through an already-open
    door, or vice versa -- which is exactly why the SNN's acoustic
    glass-break trigger benefits from a visual signal that specifically
    confirms its own hypothesis, not just a generic intrusion call. Defaults
    to `True` only so pre-existing test fixtures that construct a
    VisionVerdict without caring about this field keep compiling; every real
    code path (agent/vision.py's Gemini parse and both failsafe branches)
    sets it explicitly. Does NOT drive the alarm decision -- `is_intrusion`
    remains the sole input to `Decision.alarm` (agent/machine.py); this
    field is informational/dashboard-metric-only.
    """

    is_intrusion: bool
    confidence: float
    reason: str
    source: Literal["gemini", "failsafe"]
    window_broken: bool = True


# DECISION CONTRACT


@dataclass(frozen=True)
class Decision:
    """Final alarm decision produced by the state machine (P1)."""

    alarm: bool
    reason: str


# WAKE CONTEXT CONTRACT


@dataclass(frozen=True)
class WakeContext:
    """Context captured immediately after wake-up from halt/sleep (P1)."""

    woken_by_trigger: bool
    ts_monotonic: float
    ts_wall: float
