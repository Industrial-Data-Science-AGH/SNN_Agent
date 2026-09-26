"""Patryk's runtime seam. No simulation implementation is supplied by W0."""

from typing import Protocol


class SNNRuntime(Protocol):
    def load(self, manifest: dict) -> None:
        """Validate artifact hashes, encoder map, topology, parameters and units."""
        ...

    def reset(self, *, epoch: int, source_time_us: int) -> None:
        """Reset membrane, synapses, refractory AND decoder state."""
        ...

    def step(self, batch: dict) -> dict:
        """Preserve state across batches; return SNNDecision, never a device action."""
        ...

    def snapshot(self) -> dict:
        """Return a complete NeuronFrame with source timestamps."""
        ...

    def checkpoint(self) -> bytes:
        """Include model identity, clock and decoder state; data only, no pickle."""
        ...

    def restore(self, checkpoint: bytes) -> None:
        """Reject incompatible state; never silently restart the integrator."""
        ...
