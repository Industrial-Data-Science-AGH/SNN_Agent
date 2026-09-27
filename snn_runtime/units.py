"""Units and the training-to-reference-model conversion (task P1, point 3).

The training stack works in normalised units and in seconds; the wire contract
works in microseconds and names its potential unit explicitly. This module is
the only place that converts between them, so a wrong factor is one grep away.

Potential unit
--------------
``V_TH`` in the training stack is 1.0 by construction: the physical threshold is
VDD/2 and is not learnable, so every potential is expressed as a fraction of it.
That is ``a.u.``, not volts. A manifest may only claim ``potential_unit: "V"``
once a calibration run has tied 1.0 a.u. to a measured voltage on a specific
board set; until then the runtime marks its scores ``uncalibrated`` and the
decisions it emits are comparable across runs but not to a voltmeter. Producing
that calibration is task A2 (Andrzej), not a runtime concern.

Time unit
---------
The training stack integrates with a fixed ``DT = 0.010`` s per frame. The
encoder actually emits a frame every ``HOP_SAMPLES / FS_HZ`` seconds, which on
the ATmega is ``192 / 19231 = 9.98388 ms``. The two disagree by 0.16 %, i.e. the
simulation clock runs about one second fast per ten minutes of audio. Which one
is authoritative is a decision for the manifest, not for this module: the
runtime takes ``runtime.dt_us`` as the integration step and checks it against
the encoder profile, refusing a package where the two drift apart by more than
``DT_TOLERANCE_FRACTION``. See MAPPING.md, row ``runtime.dt_us``.

Membrane equation
-----------------
The docstring of ``snn_hw_pipeline.py`` documents

    V[t] = b*V[t-1] + (1-b)*(V_leak + I[t])

while the code, consistently in both the training and the export path
(``snn_hw_pipeline.py:236`` and ``:524``), computes

    V = b*V + (1-b)*V_leak + I

so the synaptic current is *not* scaled by ``(1-b)``. At ``tau_mem = 150 ms``
and ``dt = 10 ms`` that is a factor of about 15 on every weight. The exported
trimmer settings follow the code, so the code is what the boards were set from;
the docstring is the one that is wrong. This is recorded here rather than fixed
because changing either side silently reinterprets every exported weight, and
the arbitration belongs to P4 (comparison against Andrzej's measurements). The
runtime implements the code form and names it in ``runtime.integrator``.
"""

from __future__ import annotations

from dataclasses import dataclass

DT_TOLERANCE_FRACTION = 0.01

CALIBRATED_POTENTIAL_UNITS = frozenset({"V"})
UNCALIBRATED_POTENTIAL_UNITS = frozenset({"a.u."})

KNOWN_INTEGRATORS = frozenset({"none", "lui-order2-hard-reset"})
SCRIPTED_INTEGRATOR = "none"


def seconds_to_us(seconds: float) -> int:
    """Convert a training-stack duration to the contract's integer micros."""
    return int(round(seconds * 1_000_000))


def us_to_seconds(micros: int) -> float:
    return micros / 1_000_000


def frame_period_us(sample_rate_hz: int, hop_samples: int) -> int:
    """Encoder frame period implied by the profile, in whole microseconds."""
    if sample_rate_hz <= 0 or hop_samples <= 0:
        raise ValueError("sample_rate_hz and hop_samples must be positive")
    return int(round(1_000_000 * hop_samples / sample_rate_hz))


@dataclass(frozen=True)
class FrameGrid:
    """The encoder's timeline, which is not the integrator's step.

    This distinction caused a real bug, so it is worth stating plainly. The
    integrator advances by ``runtime.dt_us`` because that is the step the model
    was trained with. The *timeline* a batch is laid on belongs to the encoder:
    ``rpi_agents/agent/batching.py`` anchors hop ``k`` at
    ``round(k * hop_samples * 1e6 / sample_rate_hz)``, which at 192/19231 is
    about 9984 us, not 10000. Treating ``dt_us`` as the timeline unit made every
    real batch fail an "is this a whole number of steps" test, because it is a
    whole number of *hops* and never a whole number of steps.

    So: frame counts come from here, physics comes from ``dt_us``, and
    ``load_manifest`` keeps the two within ``DT_TOLERANCE_FRACTION`` of each
    other so the drift stays bounded and visible.

    All arithmetic is exact integer arithmetic on the same rational the device
    uses, so nothing accumulates. A span may sit up to one microsecond off the
    exact multiple, because the device rounds each endpoint to whole
    microseconds independently; that is the tolerance and it does not grow with
    the length of the run.
    """

    sample_rate_hz: int
    hop_samples: int

    @property
    def period_us(self) -> int:
        """Nominal frame period, for humans and for coarse comparisons."""
        return frame_period_us(self.sample_rate_hz, self.hop_samples)

    def _exact(self, frames: int) -> int:
        """``frames * hop / fs`` in microseconds, scaled by fs to stay integral."""
        return frames * self.hop_samples * 1_000_000

    def frames_in(self, span_us: int) -> int | None:
        """How many whole encoder frames a span covers, or None if it is not one.

        Rejecting rather than rounding matters: a span that is not a whole
        number of frames did not come off this encoder, and guessing would put
        the spikes of one frame into another.
        """
        if span_us <= 0:
            return None
        scaled = span_us * self.sample_rate_hz
        frames = round(scaled / (self.hop_samples * 1_000_000))
        if frames < 1 or abs(scaled - self._exact(frames)) > self.sample_rate_hz:
            return None
        return frames

    def frame_of(self, offset_us: int) -> int:
        """Which frame of a batch an in-batch spike offset belongs to.

        The device places a spike at the start of the hop it summarises, so the
        offset is a grid point and this is exact rather than a nearest match.
        """
        return round(offset_us * self.sample_rate_hz / (self.hop_samples * 1_000_000))
