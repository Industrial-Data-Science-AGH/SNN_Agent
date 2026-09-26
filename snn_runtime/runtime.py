"""The session: one model, one epoch, one continuous stream (task P2).

What this deliberately does not do
----------------------------------
The backend (``rpi_agents/cloud/app/ingest.py``) already owns the transport side
of a stream: it rejects a reused ``batch_seq`` with different content, refuses an
out of order or overlapping batch, compares ``epoch`` and ``boot_id``, emits the
``StreamGap`` records and overrides the decision status to ``gap`` when it made
one. T02/P2 says not to build a second API next to it, so this runtime does not
re-implement any of that and does not return those fields. It reads a validated
``SpikeBatch`` and answers the five things ``IngestService._decision`` actually
looks at::

    {"trigger", "status", "score", "score_kind", "provenance"}

Everything else in the emitted ``SNNDecision`` (``decision_id``, ``model_hash``,
``source_time_us``, ``event_id``, ``batch_seq``) is the backend's to fill, and it
fills it from the session record rather than from anything said here.

What it does do is the part only a stateful simulator can: carry the membrane,
the synaptic currents, the refractory counters and the decoder history across
batch boundaries, so that the same stream cut into different batch sizes gives
the same spikes and the same decisions. ``tests/runtime/test_streaming.py``
is that acceptance criterion, executed.

Time, not sequence, is the runtime's clock. It never looks at ``batch_seq``; it
looks at ``source_start_us`` against the end of the last batch it consumed. The
device lays batches on an exact frame grid (``rpi_agents/agent/batching.py``), so
a hole between two batches is a real hole in the audio, and that is the only
thing the integrator needs to know about it.
"""

from __future__ import annotations

import math
import os
from typing import Any, Mapping, Sequence

from .decoder import KOfWDecoder
from .errors import RuntimeStateError
from .integrator import LuiIntegrator
from .manifest import LoadedModel, load_manifest

# How many membrane time constants of unobserved input the runtime treats as
# "the state no longer remembers what it missed". Below this it integrates the
# gap as silence, which is what the boards physically did; at or above it the
# membrane is within e^-3 of rest anyway, so it restarts from rest instead of
# decaying through thousands of empty frames. The same number is how long the
# runtime then reports ``warmup``, because until the missed input has decayed
# out the decision rests on an assumption rather than on data.
GAP_SETTLE_TAUS = 3

_REQUIRED_BATCH_FIELDS = frozenset(
    {"epoch", "encoder_hash", "source_start_us", "source_end_us", "spikes"}
)


class LuiRuntime:
    """Holds one model package and one continuous session on it."""

    def __init__(self, *, artifact_root: str | None = None, allow_unverified_artifacts: bool = False) -> None:
        """A runtime that starts sessions verifies the weights: give it the directory the manifest's artifact
        paths are relative to. ``allow_unverified_artifacts`` is the explicit opt-out for tests and demos."""
        self._artifact_root = artifact_root
        self._allow_unverified = allow_unverified_artifacts
        self._model: LoadedModel | None = None
        self._epoch: int | None = None
        self._source_time_us: int | None = None
        self._integrator: LuiIntegrator | None = None
        self._decoder: KOfWDecoder | None = None
        self._settle_frames = 0
        self._warmup_left = 0

    # ------------------------------------------------------------------ state

    @property
    def model(self) -> LoadedModel:
        if self._model is None:
            raise RuntimeStateError("NOT_LOADED", "no model package has been accepted yet")
        return self._model

    @property
    def started(self) -> bool:
        return self._model is not None and self._epoch is not None

    @property
    def source_time_us(self) -> int | None:
        """End of the last batch consumed; where the next one has to start."""
        return self._source_time_us

    @property
    def warming_up(self) -> bool:
        return self._warmup_left > 0

    # -------------------------------------------------------------- lifecycle

    def load(self, manifest: Mapping[str, Any]) -> None:
        """Accept or reject a package. Rejection leaves the previous state intact."""
        accepted = load_manifest(
            manifest, artifact_root=self._artifact_root, require_artifacts=not self._allow_unverified,
        )
        self._model = accepted
        self._epoch = None
        self._source_time_us = None
        self._integrator = None
        self._decoder = None

    def reset(self, *, epoch: int, source_time_us: int) -> None:
        """Start a session epoch. Membrane, synapses, refractory and decoder all go."""
        model = self.model
        if epoch < 1:
            raise RuntimeStateError("INVALID_EPOCH", "session epoch counts from 1")
        if source_time_us < 0:
            raise RuntimeStateError("INVALID_CLOCK", "source_time_us is a monotonic microsecond count")

        self._epoch = epoch
        self._source_time_us = source_time_us
        self._integrator = LuiIntegrator(model.plan)
        self._decoder = KOfWDecoder.from_manifest(dict(model.decoder), model.dt_us)
        slowest = max(n["tau_mem_us"] for n in model.manifest["topology"]["neurons"])
        self._settle_frames = math.ceil(GAP_SETTLE_TAUS * slowest / model.dt_us)
        # A fresh session starts at rest, which is a real state and not an
        # assumption, so the only thing still unknown is the delay lines.
        self._warmup_left = model.plan.warmup_frames

    # ------------------------------------------------------------------- step

    def step(self, batch: Mapping[str, Any]) -> dict:
        """Consume one SpikeBatch and answer what the decoder now believes."""
        self._require_started()
        model = self.model

        if model.is_scripted:
            # A scripted package has no physics to integrate. Saying so is the
            # only honest answer; inventing one would let a demo fixture look
            # like a detector.
            return _decision(False, "invalid", None, "unavailable", "demo")

        missing = _REQUIRED_BATCH_FIELDS.difference(batch)
        if missing:
            # The backend validates every batch against the contract before we
            # are called, so this only fires for a caller that skipped it. A
            # named refusal beats a KeyError surfacing as an opaque 503.
            raise RuntimeStateError(
                "INVALID_BATCH", f"the batch is missing {', '.join(sorted(missing))}",
            )
        if batch["epoch"] != self._epoch:
            raise RuntimeStateError(
                "EPOCH_MISMATCH",
                f"batch belongs to epoch {batch['epoch']} but this session is epoch {self._epoch}",
            )
        if batch["encoder_hash"] != model.encoder_hash:
            raise RuntimeStateError(
                "ENCODER_MISMATCH",
                "the batch was produced by a different encoder profile than the loaded model",
            )

        dt, consumed = model.dt_us, self._source_time_us
        assert consumed is not None  # _require_started
        start, end = batch["source_start_us"], batch["source_end_us"]
        span = end - start
        if span <= 0 or span % dt or start % dt != consumed % dt:
            return _decision(False, "invalid", None, model.score_kind, self._provenance)
        if start < consumed:
            # The backend rejects this before we are called; if it ever gets
            # here, replaying already integrated time would corrupt the state.
            return _decision(False, "invalid", None, model.score_kind, self._provenance)

        if start > consumed:
            self._absorb_gap((start - consumed) // dt)

        frames = span // dt
        triggered, spikes = self._run(self._frame_inputs(batch, frames), frames)
        self._source_time_us = end

        status = "warmup" if self._warmup_left > 0 else "valid"
        self._warmup_left = max(0, self._warmup_left - frames)
        return _decision(
            triggered and status == "valid", status, float(spikes), model.score_kind, self._provenance,
        )

    # ----------------------------------------------------------------- pieces

    @property
    def _provenance(self) -> str:
        """``measured`` only once a board calibration backs the numbers (task A2)."""
        return "measured" if self.model.calibration == "calibrated" else "simulated"

    def _frame_inputs(self, batch: Mapping[str, Any], frames: int) -> list[list[float]]:
        """Lay the batch's spikes on the frame grid, one binary vector per frame."""
        model = self.model
        grid = [[0.0] * model.plan.n_channels for _ in range(frames)]
        for spike in batch["spikes"]:
            index = model.channel_index.get(spike["channel"])
            if index is None:
                continue  # the contract validated this already; ignoring beats guessing
            frame = spike["dt_us"] // model.dt_us
            if 0 <= frame < frames:
                # A channel that somehow reports twice in one frame is still one
                # spike: the encoder pin is a level, not a counter.
                grid[frame][index] = 1.0
        return grid

    def _run(self, grid: Sequence[Sequence[float]], frames: int) -> tuple[bool, int]:
        """Integrate the batch. Returns (the decoder fired, decision spikes)."""
        integrator, decoder = self._integrator, self._decoder
        assert integrator is not None and decoder is not None
        decision_index = self.model.plan.decision_index
        triggered, spikes = False, 0
        for frame in range(frames):
            fired = integrator.step(grid[frame])[decision_index]
            if fired:
                spikes += 1
            # The decoder runs through warm-up too, so a cooldown opened there
            # still suppresses the duplicate that would follow it. Only the
            # reporting of the trigger is held back, in step().
            if decoder.step(bool(fired)):
                triggered = True
        return triggered, spikes

    def _absorb_gap(self, missing: int) -> None:
        """Account for frames that were produced but never arrived."""
        integrator, decoder = self._integrator, self._decoder
        assert integrator is not None and decoder is not None
        if missing >= self._settle_frames:
            integrator.reset()
        else:
            empty = [0.0] * self.model.plan.n_channels
            for _ in range(missing):
                integrator.step(empty)
        decoder.skip(missing)
        decoder.flush()
        self._warmup_left = max(self._warmup_left, self._settle_frames)

    # ------------------------------------------------------------ later tasks

    def snapshot(self) -> dict:
        self._require_started()
        raise NotImplementedError("NeuronFrame telemetry is task P3")

    def checkpoint(self) -> bytes:
        self._require_started()
        raise NotImplementedError("checkpoint/restore is task P5")

    def restore(self, checkpoint: bytes) -> None:
        self._require_started()
        raise NotImplementedError("checkpoint/restore is task P5")

    def _require_started(self) -> None:
        _ = self.model
        if self._epoch is None:
            raise RuntimeStateError("NOT_STARTED", "reset() must open a session epoch first")


def _decision(trigger: bool, status: str, score: float | None, score_kind: str, provenance: str) -> dict:
    return {
        "trigger": trigger,
        "status": status,
        "score": score,
        "score_kind": score_kind,
        "provenance": provenance,
    }


def from_environment(env: Mapping[str, str] | None = None) -> LuiRuntime:
    """The factory a backend names in ``SNN_RUNTIME=snn_runtime.runtime:from_environment``.

    ``SNN_MODEL_ARTIFACT_ROOT`` is the directory holding the weights the manifest names. Without it the runtime
    refuses any package that lists artifacts, unless ``SNN_ALLOW_UNVERIFIED_ARTIFACTS=1`` says that is intended."""
    env = os.environ if env is None else env
    root = env.get("SNN_MODEL_ARTIFACT_ROOT", "").strip() or None
    return LuiRuntime(
        artifact_root=root, allow_unverified_artifacts=env.get("SNN_ALLOW_UNVERIFIED_ARTIFACTS", "").strip() == "1",
    )
