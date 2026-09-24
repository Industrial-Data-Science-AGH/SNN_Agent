"""Load and reject model packages (task P1).

``contracts.validation`` already owns the wire contract: schema shape, encoder
hash, unique neuron ids, one connection per physical port, artifact bookkeeping,
champion provenance. This module does not repeat any of that. It adds the checks
that only a runtime can make, because they are about whether the package can
actually be *integrated*, not about whether it is well formed:

  * the integration step agrees with the encoder frame period (units.py),
  * the step is small enough for the time constants it is asked to integrate,
  * the decoder rule is expressible on that step grid,
  * exactly one neuron carries the decision,
  * the artifacts named in the manifest exist and hash to what was declared,
  * a package claiming volts also names the calibration that produced them.

Every rejection raises RuntimeLoadError with a stable code, before any state is
allocated, so a bad package can never half-start a session.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from contracts.validation import ContractError, content_hash, validate

from .errors import RuntimeLoadError
from .units import (
    CALIBRATED_POTENTIAL_UNITS,
    DT_TOLERANCE_FRACTION,
    KNOWN_INTEGRATORS,
    SCRIPTED_INTEGRATOR,
    frame_period_us,
)

# Lu.i boards expose three solder-side synapse inputs, J1..J3. The contract
# encodes them as target_port 1..3; this is the physical constant behind it and
# is deliberately unrelated to the number of encoder features (currently 7).
PHYSICAL_PORTS = (1, 2, 3)


@dataclass(frozen=True)
class PortBinding:
    """One wire into one physical synapse input of one board."""

    target_id: str
    target_port: int
    source_kind: str
    source_id: str
    sign: str
    weight: float
    delay_us: int

    @property
    def signed_weight(self) -> float:
        """Weight with its sign reapplied, for the integrator.

        The contract stores magnitude and polarity separately (``weight`` has
        ``minimum: 0``) because a trimmer setting has no sign; the sign is which
        header the wire goes into. The simulator wants one number.
        """
        return -self.weight if self.sign == "inhibitory" else self.weight


@dataclass(frozen=True)
class LoadedModel:
    """An accepted package. Everything the integrator needs, already checked."""

    model_id: str
    status: str
    model_hash: str
    encoder_hash: str
    dt_us: int
    integrator: str
    potential_unit: str
    calibration: str
    decision_neuron: str
    neuron_order: tuple[str, ...]
    bindings: Mapping[str, tuple[PortBinding, ...]]
    channel_index: Mapping[str, int]
    unused_channels: tuple[str, ...]
    decoder: Mapping[str, Any]
    manifest: Mapping[str, Any]

    @property
    def is_scripted(self) -> bool:
        return self.integrator == SCRIPTED_INTEGRATOR

    @property
    def score_kind(self) -> str:
        """What an SNNDecision may claim about its score.

        Uncalibrated potentials still yield a well defined spike count, so the
        contract's ``spike_count`` stays honest; what is not available is any
        claim in volts. A package that is scripted has no score at all.
        """
        if self.is_scripted:
            return "unavailable"
        return "spike_count" if self.calibration == "calibrated" else "uncalibrated"


def _reject(code: str, message: str) -> None:
    raise RuntimeLoadError(code, message)


def _check_clock(manifest: Mapping[str, Any]) -> int:
    dt_us = manifest["runtime"]["dt_us"]
    profile = manifest["encoder_profile"]
    expected = frame_period_us(profile["sample_rate_hz"], profile["hop_samples"])
    drift = abs(dt_us - expected) / expected
    if drift > DT_TOLERANCE_FRACTION:
        _reject(
            "DT_ENCODER_MISMATCH",
            f"runtime.dt_us={dt_us} but the encoder emits a frame every {expected} us "
            f"({profile['hop_samples']}/{profile['sample_rate_hz']} Hz), a drift of "
            f"{drift:.2%}; one of the two is wrong and the integrator would run at "
            "the wrong speed",
        )
    return dt_us


def _check_time_constants(manifest: Mapping[str, Any], dt_us: int) -> None:
    for neuron in manifest["topology"]["neurons"]:
        for field in ("tau_mem_us", "tau_syn_us"):
            tau = neuron[field]
            if tau <= dt_us:
                _reject(
                    "DT_NOT_STABLE",
                    f"neuron {neuron['neuron_id']} has {field}={tau} us at dt_us={dt_us}; "
                    "the explicit integrator needs a step below every time constant",
                )


def _check_decoder(manifest: Mapping[str, Any], dt_us: int) -> None:
    decoder = manifest["decoder"]
    window, cooldown, threshold = decoder["window_us"], decoder["cooldown_us"], decoder["threshold"]
    if window < dt_us or window % dt_us:
        _reject(
            "DECODER_WINDOW",
            f"decoder.window_us={window} is not a whole number of dt_us={dt_us} steps; "
            "the k-of-w rule would straddle frames",
        )
    if cooldown < window:
        _reject(
            "DECODER_COOLDOWN",
            f"decoder.cooldown_us={cooldown} is shorter than the decision window {window}; "
            "one event could be counted twice",
        )
    if threshold <= 0 or threshold != int(threshold):
        _reject(
            "DECODER_THRESHOLD",
            f"decoder.threshold={threshold} must be a positive whole number of spikes; "
            "zero would alarm on silence",
        )


def _find_decision_neuron(manifest: Mapping[str, Any]) -> str:
    neurons = [n["neuron_id"] for n in manifest["topology"]["neurons"]]
    driving = {
        edge["source_id"]
        for edge in manifest["topology"]["connections"]
        if edge["source_kind"] == "neuron"
    }
    terminal = [n for n in neurons if n not in driving]
    if not terminal:
        _reject(
            "NO_DECISION_NEURON",
            "every neuron drives another one, so the network has no output; "
            "the contract has no explicit decision_neuron field, so the runtime "
            "infers it as the neuron with no outgoing connection",
        )
    if len(terminal) > 1:
        _reject(
            "AMBIGUOUS_DECISION_NEURON",
            f"{len(terminal)} neurons have no outgoing connection ({', '.join(sorted(terminal))}); "
            "the runtime cannot tell which one the decoder should read",
        )
    return terminal[0]


def _build_bindings(manifest: Mapping[str, Any]) -> dict[str, tuple[PortBinding, ...]]:
    by_target: dict[str, list[PortBinding]] = {
        n["neuron_id"]: [] for n in manifest["topology"]["neurons"]
    }
    for edge in manifest["topology"]["connections"]:
        if edge["target_port"] not in PHYSICAL_PORTS:
            _reject(
                "PORT_OUT_OF_RANGE",
                f"connection into {edge['target_id']} uses port {edge['target_port']}; "
                f"a Lu.i board has only ports {PHYSICAL_PORTS}",
            )
        by_target[edge["target_id"]].append(
            PortBinding(
                target_id=edge["target_id"],
                target_port=edge["target_port"],
                source_kind=edge["source_kind"],
                source_id=edge["source_id"],
                sign=edge["sign"],
                weight=edge["weight"],
                delay_us=edge["delay_us"],
            )
        )
    for neuron_id, bound in by_target.items():
        if len(bound) > len(PHYSICAL_PORTS):
            _reject(
                "PORT_FANIN",
                f"neuron {neuron_id} has {len(bound)} inputs but only "
                f"{len(PHYSICAL_PORTS)} physical synapse ports",
            )
    return {k: tuple(sorted(v, key=lambda b: b.target_port)) for k, v in by_target.items()}


def _check_calibration(manifest: Mapping[str, Any]) -> str:
    unit = manifest["runtime"]["potential_unit"]
    calibration_id = manifest["provenance"]["calibration_id"]
    if unit in CALIBRATED_POTENTIAL_UNITS:
        if not calibration_id:
            _reject(
                "CALIBRATION_MISSING",
                f"potential_unit={unit!r} claims measured volts but provenance."
                "calibration_id is null; no board measurement backs the numbers",
            )
        return "calibrated"
    return "uncalibrated"


def _check_artifacts(manifest: Mapping[str, Any], artifact_root: Path) -> None:
    for artifact in manifest["artifacts"]:
        path = artifact_root / artifact["path"]
        if not path.is_file():
            _reject("ARTIFACT_MISSING", f"artifact {artifact['path']} is not on disk under {artifact_root}")
        digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != artifact["sha256"]:
            _reject(
                "ARTIFACT_HASH_MISMATCH",
                f"artifact {artifact['path']} hashes to {digest} but the manifest declares "
                f"{artifact['sha256']}",
            )


def load_manifest(manifest: Mapping[str, Any], *, artifact_root: str | Path | None = None) -> LoadedModel:
    """Validate a model package and return what the integrator needs.

    ``artifact_root`` enables the on-disk artifact check. It is optional so that
    contract-level tests and the dashboard can validate a manifest they received
    over the wire without holding the weights.
    """
    try:
        validate("ModelManifest", dict(manifest))
    except ContractError as exc:
        raise RuntimeLoadError(exc.code, str(exc)) from exc

    if not manifest["topology"]["neurons"]:
        _reject(
            "EMPTY_TOPOLOGY",
            "the package declares zero neurons; an empty editor draft is not a classifier",
        )

    integrator = manifest["runtime"]["integrator"]
    if integrator not in KNOWN_INTEGRATORS:
        _reject(
            "UNKNOWN_INTEGRATOR",
            f"runtime.integrator={integrator!r} is not implemented here; known: "
            f"{sorted(KNOWN_INTEGRATORS)}",
        )

    dt_us = manifest["runtime"]["dt_us"]
    if integrator != SCRIPTED_INTEGRATOR:
        # A scripted fixture has no physics, so neither the encoder clock nor the
        # time constants mean anything for it.
        dt_us = _check_clock(manifest)
        _check_time_constants(manifest, dt_us)
    _check_decoder(manifest, dt_us)

    decision_neuron = _find_decision_neuron(manifest)
    bindings = _build_bindings(manifest)
    calibration = _check_calibration(manifest)

    if artifact_root is not None:
        _check_artifacts(manifest, Path(artifact_root))

    channel_map: Sequence[Mapping[str, Any]] = manifest["encoder_profile"]["channel_map"]
    channel_index = {c["channel"]: c["index"] for c in channel_map}
    wired = {
        edge["source_id"]
        for edge in manifest["topology"]["connections"]
        if edge["source_kind"] == "channel"
    }

    return LoadedModel(
        model_id=manifest["model_id"],
        status=manifest["status"],
        model_hash=content_hash(dict(manifest)),
        encoder_hash=manifest["encoder_hash"],
        dt_us=dt_us,
        integrator=integrator,
        potential_unit=manifest["runtime"]["potential_unit"],
        calibration=calibration,
        decision_neuron=decision_neuron,
        neuron_order=tuple(n["neuron_id"] for n in manifest["topology"]["neurons"]),
        bindings=bindings,
        channel_index=channel_index,
        unused_channels=tuple(sorted(set(channel_index) - wired)),
        decoder=dict(manifest["decoder"]),
        manifest=dict(manifest),
    )
