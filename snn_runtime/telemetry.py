"""NeuronFrame telemetry: what the boards are doing right now (task P3).

Two things live here, and they are deliberately separate.

``build_frame`` turns the integrator's state into one ``NeuronFrame``. It is a
pure function of state that has already been computed, so asking for telemetry
can never change a decision: the UI reads the simulation, it does not drive it.

``TelemetryFeed`` decides *how often* a frame leaves the runtime. A session runs
at ``dt_us`` (10 ms on our model, 100 frames a second) and no browser needs that,
so the feed thins the stream on **source time**, not on wall clock: the same
stream replayed faster or slower produces the same frames. Two rules keep the
thinning honest, and they are the reason this is a class and not a modulo:

* a frame whose ``status`` differs from the last one that was sent is always
  sent — a gap never disappears because it fell between two sampling points;
* ``frame_seq`` counts frames that were *emitted*, and ``source_time_us`` says
  when each one was taken, so a client can see it is looking at a thinned
  stream instead of assuming it has every frame.

What a frame does not carry is any claim about volts. ``potential_unit`` comes
from the manifest and stays ``a.u.`` until a board calibration (task A2) gives
the numbers a unit, and ``provenance`` says ``simulated`` for exactly as long.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "1.0"

#: Status values the contract allows, in the order the session moves through them.
STATUSES = ("warmup", "running", "gap", "stopped")


def frame_provenance(*, scripted: bool, calibration: str) -> str:
    """What the numbers in a frame are: a script, a simulation or a measurement."""
    if scripted:
        return "demo"
    return "measured" if calibration == "calibrated" else "simulated"


def build_frame(
    *,
    device_id: str,
    session_id: str,
    epoch: int,
    source_time_us: int,
    model_hash: str,
    frame_seq: int,
    topology_version: str,
    status: str,
    provenance: str,
    potential_unit: str,
    neurons: Sequence[Mapping[str, Any]],
    membrane: Sequence[float] | None,
    spiked: Sequence[bool],
) -> dict:
    """One NeuronFrame for the current state of the network.

    ``membrane`` is ``None`` for a package with no physics to report (a scripted
    demo): every ``v_mem`` is then ``null``, which the contract allows and which
    a dashboard must render as "no reading" rather than as a potential of zero.
    """
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}, not {status!r}")

    rows = []
    for index, neuron in enumerate(neurons):
        rows.append(
            {
                "neuron_id": neuron["neuron_id"],
                "v_mem": None if membrane is None else float(membrane[index]),
                "v_threshold": float(neuron["v_threshold"]),
                "v_reset": float(neuron["v_reset"]),
                "spiked": bool(spiked[index]) if index < len(spiked) else False,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "device_id": device_id,
        "session_id": session_id,
        "epoch": epoch,
        "source_time_us": source_time_us,
        "model_hash": model_hash,
        "frame_seq": frame_seq,
        "topology_version": topology_version,
        "status": status,
        "provenance": provenance,
        "potential_unit": potential_unit,
        "neurons": rows,
    }


class TelemetryFeed:
    """Thins a frame stream for a viewer without hiding what happened.

    ``min_interval_us`` is spacing in **source** time. ``0`` means "send every
    frame offered", which is what a test or a short golden replay wants.
    """

    def __init__(self, *, min_interval_us: int = 100_000) -> None:
        if min_interval_us < 0:
            raise ValueError("min_interval_us cannot be negative")
        self.min_interval_us = min_interval_us
        self._last_sent_us: int | None = None
        self._last_status: str | None = None
        self._skipped = 0
        self._sent = 0

    @property
    def skipped(self) -> int:
        """Frames offered and not sent since the feed was created."""
        return self._skipped

    @property
    def sent(self) -> int:
        """Frames actually emitted since the feed was created."""
        return self._sent

    def reset(self) -> None:
        self._last_sent_us = None
        self._last_status = None
        self._skipped = 0
        self._sent = 0

    def offer(self, frame: Mapping[str, Any]) -> dict | None:
        """Return the frame if a viewer should see it, otherwise ``None``."""
        status = frame["status"]
        now = frame["source_time_us"]
        due = (
            self._last_sent_us is None
            or status != self._last_status
            or now - self._last_sent_us >= self.min_interval_us
        )
        if not due:
            self._skipped += 1
            return None
        self._last_sent_us = now
        self._last_status = status
        # `frame_seq` is the number of frames SENT, which only the feed knows.
        # The snapshot counter cannot serve: thinning drops frames, so reusing
        # it would hand the viewer a sequence with holes in it, and a hole in
        # `frame_seq` is exactly how a viewer detects that it lost frames.
        sent = dict(frame) | {"frame_seq": self._sent}
        self._sent += 1
        return sent
