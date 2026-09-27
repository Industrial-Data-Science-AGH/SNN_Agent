"""Reviewing a wiring sketch from the editor (task P3, point 3).

A draft is what the dashboard's network editor produces: boards placed on a
canvas and wires drawn between them. This module answers two questions about
one, and refuses to answer a third.

**Is it a wiring a Lu.i rack could actually carry?** That is the contract
``TopologyDraft`` plus its cross-field rules: at most 50 boards, three synapse
ports per board, one wire per port, no wire into a board that is not there.
``review_draft`` turns a failure into a sentence a person can act on instead of
a schema path.

**Could a session run on it?** No, and that is not a bug. Zero boards is an
empty editor; fifty boards wired beautifully is still a picture. A session needs
weights, time constants, a decoder, a checkpoint and the hashes that tie them
together — a ``ModelManifest``, produced by training, not by dragging. So
``review_draft`` always reports ``runnable`` as ``False`` and says what is
missing, and ``LuiRuntime.load`` refuses a draft by name rather than dying on a
missing key. Changing the board count in the editor therefore cannot quietly
rebuild the champion: nothing in this module touches a loaded model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from contracts.validation import ContractError, validate

MAX_BOARDS = 50
PORTS_PER_BOARD = 3

#: What a draft is missing before anything could be started from it. These are
#: the parts that only training and export can supply.
MISSING_FOR_A_SESSION = (
    "weights per synapse",
    "tau_mem_us, tau_syn_us, v_leak, v_threshold, v_reset per board",
    "a decoder rule (k, window, cooldown)",
    "an encoder profile and its hash",
    "a checkpoint artifact and its sha256",
)


@dataclass(frozen=True)
class DraftReview:
    """The verdict on one sketch."""

    draft_id: str
    boards: int
    connections: int
    wiring_ok: bool
    runnable: bool
    reasons: tuple[str, ...]

    @property
    def empty(self) -> bool:
        return self.boards == 0


def review_draft(draft: Mapping[str, Any]) -> DraftReview:
    """Check a draft and explain the verdict. Never raises for a valid sketch.

    A draft that breaks the contract raises ``ContractError`` the same way any
    other payload does, because a malformed sketch is a client bug, not an
    opinion. A well-formed sketch always comes back with ``runnable=False``.
    """
    validate("TopologyDraft", dict(draft))

    boards = draft["boards"]
    wires = draft["connections"]
    reasons: list[str] = []

    if not boards:
        reasons.append("the editor is empty: zero boards is a blank canvas, not a classifier")
    else:
        fan_in: dict[str, int] = {}
        for wire in wires:
            fan_in[wire["target"]] = fan_in.get(wire["target"], 0) + 1
        unwired = [b["id"] for b in boards if fan_in.get(b["id"], 0) == 0]
        if unwired:
            reasons.append(
                "these boards have no input wired: " + ", ".join(sorted(unwired)[:8])
            )
    reasons.append(
        "a draft carries none of: " + "; ".join(MISSING_FOR_A_SESSION)
        + " — a new session needs a ModelManifest, not a sketch"
    )

    return DraftReview(
        draft_id=draft["draft_id"],
        boards=len(boards),
        connections=len(wires),
        wiring_ok=True,
        runnable=False,
        reasons=tuple(reasons),
    )


def looks_like_a_draft(payload: Mapping[str, Any]) -> bool:
    """True when a payload is a wiring sketch rather than a model package."""
    return "boards" in payload and "topology" not in payload


def describe_refusal(payload: Mapping[str, Any]) -> str:
    """The sentence ``LuiRuntime.load`` uses when it is handed a draft."""
    try:
        review = review_draft(payload)
    except ContractError:
        return "this is a topology draft, not a ModelManifest, and it is not even a valid draft"
    return (
        f"draft {review.draft_id} ({review.boards} boards, {review.connections} connections) "
        "is a wiring sketch, not a model package: " + review.reasons[-1]
    )
