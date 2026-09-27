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

That grid is the **encoder's**, not the integrator's. Hop ``k`` starts at
``round(k * hop_samples / sample_rate_hz)`` seconds, about 9984 us at 192/19231,
while ``runtime.dt_us`` is 10000 because that is the step the model was trained
with. Frame counts therefore come from ``model.grid`` (see ``units.FrameGrid``)
and physics from ``dt_us``; conflating the two made every batch the device
actually emits fail as ``invalid``, and only a test driven by the real
``BatchAssembler`` catches it.
"""

from __future__ import annotations

import math
import os
from typing import Any, Mapping, Sequence

from .decoder import KOfWDecoder
from .errors import RuntimeLoadError, RuntimeStateError
from .integrator import LuiIntegrator
from .manifest import LoadedModel, load_manifest
from .telemetry import build_frame, frame_provenance
from .topology import describe_refusal, looks_like_a_draft

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
        # Telemetry (task P3). The identity of the stream is the device's, not
        # ours: it arrives on every SpikeBatch and a session may also declare it
        # up front, so a viewer attached before the first batch still gets a
        # frame that names the session it belongs to.
        self._device_id: str | None = None
        self._session_id: str | None = None
        self._frame_seq = 0
        self._stopped = False
        self._gap_pending = False
        self._observed = False
        self._last_spikes: tuple[bool, ...] = ()

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
        if looks_like_a_draft(manifest):
            # The editor can hand a sketch to anything that takes JSON. Saying
            # what it is beats failing later on a missing key, and it is the
            # guarantee that redrawing the network cannot replace the champion.
            raise RuntimeLoadError("DRAFT_NOT_A_MODEL", describe_refusal(manifest))
        accepted = load_manifest(
            manifest, artifact_root=self._artifact_root, require_artifacts=not self._allow_unverified,
        )
        self._model = accepted
        self._epoch = None
        self._source_time_us = None
        self._integrator = None
        self._decoder = None

    def reset(
        self,
        *,
        epoch: int,
        source_time_us: int,
        device_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Start a session epoch. Membrane, synapses, refractory and decoder all go.

        ``device_id`` and ``session_id`` are optional and only name the stream
        for telemetry; the batches carry them too, and a mismatch between the
        two is refused in ``step`` rather than silently preferred one way.
        """
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
        # Not `device_id or self._device_id`: a new epoch that names no stream
        # has to LEARN the identity from its first batch, as the docstring
        # promises. Carrying the old one forward would label the new session's
        # telemetry with the previous session and then reject its first batch
        # as a SESSION_MISMATCH against an identity nobody set.
        self._device_id = device_id
        self._session_id = session_id
        self._frame_seq = 0
        self._last_spikes = (False,) * len(model.plan.neurons)
        # Nothing has been observed yet, so the state is rest by assumption
        # rather than by measurement. That is exactly what warmup means here,
        # and warmup_frames == 0 still means no input has been seen.
        self._stopped = False
        self._gap_pending = False
        self._observed = False

    # ------------------------------------------------------------------- step

    def step(self, batch: Mapping[str, Any]) -> dict:
        """Consume one SpikeBatch and answer what the decoder now believes."""
        self._require_started()
        model = self.model
        if self._stopped:
            # stop() promises the state is frozen and snapshot() labels it
            # `stopped`. Integrating one more batch would quietly make that
            # label false, so this is refused rather than tolerated.
            raise RuntimeStateError(
                "SESSION_STOPPED", "the session was stopped; open a new epoch to keep streaming",
            )

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

        consumed = self._source_time_us
        assert consumed is not None  # _require_started
        start, end = batch["source_start_us"], batch["source_end_us"]
        frames = model.grid.frames_in(end - start)
        missed = None if start == consumed else model.grid.frames_in(start - consumed)
        if frames is None or start < consumed or (start != consumed and missed is None):
            # Either the window is not a whole number of encoder frames, or it
            # replays time already integrated, or the hole before it is not on
            # the grid either. None of the three can be placed on the timeline,
            # and guessing would put one frame's spikes into another.
            return _decision(False, "invalid", None, model.score_kind, self._provenance)

        # Only now, once nothing can reject the batch, does it get to say which
        # stream this is. A rejected batch that had already bound its identity
        # would make the next good batch fail as a SESSION_MISMATCH.
        self._bind_stream(batch)

        if missed:
            self._absorb_gap(missed)
        self._absorb_quality(batch.get("quality") or {})

        if model.is_scripted:
            # A scripted package has no physics to integrate, and saying
            # otherwise would let a demo fixture look like a detector. It still
            # consumes the timeline, so its telemetry describes the input it was
            # given rather than standing at the reset clock forever.
            self._source_time_us = end
            self._observed = True
            return _decision(False, "invalid", None, "unavailable", "demo")

        triggered, spikes = self._run(self._frame_inputs(batch, frames), frames)
        self._source_time_us = end
        self._observed = True

        status = "warmup" if self._warmup_left > 0 else "valid"
        self._warmup_left = max(0, self._warmup_left - frames)
        return _decision(
            triggered and status == "valid", status, float(spikes), model.score_kind, self._provenance,
        )

    # ----------------------------------------------------------------- pieces

    def _bind_stream(self, batch: Mapping[str, Any]) -> None:
        """Learn which device and session this stream is, and keep it that way."""
        for field, current in (("device_id", self._device_id), ("session_id", self._session_id)):
            incoming = batch.get(field)
            if incoming is None:
                continue
            if current is None:
                setattr(self, f"_{field}", incoming)
            elif incoming != current:
                raise RuntimeStateError(
                    "SESSION_MISMATCH",
                    f"batch reports {field} {incoming!r} but this session is {current!r}",
                )

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
            frame = model.grid.frame_of(spike["dt_us"])
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
            fired_all = integrator.step(grid[frame])
            # Telemetry reports the last frame of the batch, so the spike flags
            # belong to the same instant as the membrane values it reads.
            self._last_spikes = tuple(bool(x) for x in fired_all)
            fired = fired_all[decision_index]
            if fired:
                spikes += 1
            # The decoder runs through warm-up too, so a cooldown opened there
            # still suppresses the duplicate that would follow it. Only the
            # reporting of the trigger is held back, in step().
            if decoder.step(bool(fired)):
                triggered = True
        return triggered, spikes

    def _absorb_quality(self, quality: Mapping[str, Any]) -> None:
        """Take the device's own word for it when a batch is degraded.

        ``IngestService._gaps`` already turns either flag into a ``StreamGap``
        and overrides the decision to ``gap``. Without this the runtime would
        disagree with the backend about the very same batch: the ack would say
        a hole went past while ``snapshot()`` still reported ``running``.

        The two flags are not the same kind of loss. ``dropped_events`` means
        hops the device summarised away, so frames really are missing and the
        decoder must not span the hole. ``adc_clipped`` means the input was
        saturated, not absent, so the timeline is intact and only the viewer
        needs to know the data is degraded.
        """
        if quality.get("dropped_events"):
            decoder = self._decoder
            assert decoder is not None
            decoder.flush()
            self._warmup_left = max(self._warmup_left, self._settle_frames)
            self._gap_pending = True
        elif quality.get("adc_clipped"):
            self._gap_pending = True

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
        self._last_spikes = (False,) * len(self.model.plan.neurons)
        # A viewer may be sampling far more slowly than the stream runs. The
        # next frame it receives has to say a hole went past, so the flag waits
        # for that frame instead of expiring with the batch that absorbed it.
        self._gap_pending = True

    # -------------------------------------------------------------- telemetry

    def stop(self) -> None:
        """Close the session for telemetry. The state stays readable, frozen."""
        self._require_started()
        self._stopped = True

    @property
    def telemetry_status(self) -> str:
        """What a frame taken right now would say about the session."""
        if self._stopped:
            return "stopped"
        if self._gap_pending:
            return "gap"
        # Decision warm-up and "nothing has arrived yet" are both states where
        # the picture rests on an assumption, and the contract has one word for
        # that. The decision path is untouched: this reads state, never sets it.
        return "warmup" if self._warmup_left > 0 or not self._observed else "running"

    def snapshot(self) -> dict:
        """One NeuronFrame for the state the network is in right now.

        Reading telemetry never advances the simulation and never changes a
        decision; the one thing it does change is the gap flag, which is cleared
        by the frame that reports it, so a hole is announced exactly once and to
        somebody rather than being dropped between two sampling points.
        """
        self._require_started()
        model = self.model
        if self._device_id is None or self._session_id is None:
            raise RuntimeStateError(
                "NO_STREAM_IDENTITY",
                "the session has no device_id/session_id yet: pass them to reset() "
                "or take the snapshot after the first batch",
            )

        neurons = model.manifest["topology"]["neurons"]
        by_id = {n["neuron_id"]: n for n in neurons}
        ordered = [by_id[name] for name in model.neuron_order]
        membrane = None if model.is_scripted or self._integrator is None else self._integrator.membrane

        frame = build_frame(
            device_id=self._device_id,
            session_id=self._session_id,
            epoch=self._epoch or 0,
            source_time_us=self._source_time_us or 0,
            model_hash=model.model_hash,
            frame_seq=self._frame_seq,
            topology_version=model.manifest["topology"]["topology_version"],
            status=self.telemetry_status,
            provenance=frame_provenance(scripted=model.is_scripted, calibration=model.calibration),
            potential_unit=model.potential_unit,
            neurons=ordered,
            membrane=membrane,
            spiked=self._last_spikes,
        )
        self._frame_seq += 1
        self._gap_pending = False
        return frame

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
