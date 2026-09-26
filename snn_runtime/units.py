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
