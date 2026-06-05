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
    """Verdict from the Gemini vision stage or the failsafe fallback (P3)."""

    is_intrusion: bool
    confidence: float
    reason: str
    source: Literal["gemini", "failsafe"]


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
