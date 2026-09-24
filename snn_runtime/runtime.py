from __future__ import annotations

from typing import Any, Mapping

from .errors import RuntimeStateError
from .manifest import LoadedModel, load_manifest


class LuiRuntime:
    """Holds one model package and, later, one continuous session on it."""

    def __init__(self, *, artifact_root: str | None = None) -> None:
        self._artifact_root = artifact_root
        self._model: LoadedModel | None = None
        self._epoch: int | None = None
        self._source_time_us: int | None = None

    @property
    def model(self) -> LoadedModel:
        if self._model is None:
            raise RuntimeStateError("NOT_LOADED", "no model package has been accepted yet")
        return self._model

    @property
    def started(self) -> bool:
        return self._model is not None and self._epoch is not None

    def load(self, manifest: Mapping[str, Any]) -> None:
        """Accept or reject a package. Rejection leaves the previous state intact."""
        accepted = load_manifest(manifest, artifact_root=self._artifact_root)
        self._model = accepted
        self._epoch = None
        self._source_time_us = None

    def reset(self, *, epoch: int, source_time_us: int) -> None:
        """Start a session epoch. Membrane, synapses, refractory and decoder all go."""
        model = self.model
        if epoch < 1:
            raise RuntimeStateError("INVALID_EPOCH", "session epoch counts from 1")
        if source_time_us < 0:
            raise RuntimeStateError("INVALID_CLOCK", "source_time_us is a monotonic microsecond count")
        self._epoch = epoch
        self._source_time_us = source_time_us
        # P2 allocates the integrator state here, one entry per neuron of
        # model.neuron_order, plus the decoder ring buffer sized from
        # model.decoder["window_us"] // model.dt_us.
        _ = model

    def step(self, batch: Mapping[str, Any]) -> dict:
        self._require_started()
        raise NotImplementedError("stateful streaming is task P2")

    def snapshot(self) -> dict:
        self._require_started()
        raise NotImplementedError("NeuronFrame telemetry is task P3")

    def checkpoint(self) -> bytes:
        self._require_started()
        raise NotImplementedError("checkpoint/restore is task P5")

    def restore(self, checkpoint: bytes) -> None:
        _ = self.model
        raise NotImplementedError("checkpoint/restore is task P5")

    def _require_started(self) -> None:
        _ = self.model
        if self._epoch is None:
            raise RuntimeStateError("NOT_STARTED", "reset() must open a session epoch first")
