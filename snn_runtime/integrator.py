"""Second order Lu.i dynamics, one frame at a time (task P2, point 1).

The training stack integrates a whole clip at once, layer by layer: ``H`` runs
over all T frames, then ``G``, then ``O`` (``snn_hw_pipeline.py:211-241``). That
is a batching detail, not physics. Every edge in that network is causal and, at
``delay_us = 0``, propagates within the same frame, so evaluating the neurons in
topological order once per frame reproduces the exact same numbers while keeping
the state addressable between frames. That is what a streaming runtime needs and
what this module does.

Per frame, per neuron, mirroring the training code and not its docstring (see
``units.py`` for why those two disagree)::

    I <- alpha*I + sum(signed weight of every firing input)
    V <- beta*V + (1 - beta)*V_leak + I
    spike if V >= V_threshold, then V <- V_reset

with ``alpha = exp(-dt/tau_syn)`` and ``beta = exp(-dt/tau_mem)``. The membrane
starts at ``V_leak``, which is the resting value of that recurrence with no
input, so a fresh session does not have to settle before it means anything.

Everything here is plain Python. ``requirements-w0.lock`` has no numpy, so a
numpy dependency would not install in the ``pr-gate`` and would have to be added
to the Pi image as well; at eight neurons and twenty four synapses per frame
there is nothing to gain from it. The cost is one dict lookup free inner loop
over precompiled tuples, about a hundred frames of real time per second.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .errors import RuntimeLoadError


@dataclass(frozen=True)
class NeuronParams:
    """One board, with the time constants already turned into decay factors."""

    neuron_id: str
    alpha: float
    beta: float
    v_leak: float
    v_threshold: float
    v_reset: float
    refractory_frames: int


@dataclass(frozen=True)
class Wire:
    """One synapse, resolved to indices so the inner loop does no lookups.

    Exactly one of ``channel`` and ``neuron`` is set. ``weight`` already carries
    its polarity, so the integrator never has to look at ``sign`` again.
    """

    weight: float
    channel: int | None
    neuron: int | None
    delay_frames: int


@dataclass(frozen=True)
class NetworkPlan:
    """Everything the integrator can decide once, at load time."""

    neurons: tuple[NeuronParams, ...]
    inputs: tuple[tuple[Wire, ...], ...]
    order: tuple[int, ...]
    index_of: Mapping[str, int]
    decision_index: int
    n_channels: int
    max_delay: int

    @property
    def warmup_frames(self) -> int:
        """Frames before the delay lines hold real history rather than zeros."""
        return self.max_delay


def build_plan(
    *,
    neurons: Sequence[Mapping[str, Any]],
    bindings: Mapping[str, Sequence[Any]],
    channel_index: Mapping[str, int],
    dt_us: int,
    decision_neuron: str,
) -> NetworkPlan:
    """Compile an accepted manifest into an integrable network.

    Raises RuntimeLoadError for the two things that make a topology valid on the
    wire but impossible to integrate: a delay that is not a whole number of
    frames, and a cycle with no delay in it.
    """
    index_of = {n["neuron_id"]: i for i, n in enumerate(neurons)}
    params = tuple(
        NeuronParams(
            neuron_id=n["neuron_id"],
            alpha=math.exp(-dt_us / n["tau_syn_us"]),
            beta=math.exp(-dt_us / n["tau_mem_us"]),
            v_leak=float(n["v_leak"]),
            v_threshold=float(n["v_threshold"]),
            v_reset=float(n["v_reset"]),
            refractory_frames=_refractory_frames(n, dt_us),
        )
        for n in neurons
    )

    wires: list[tuple[Wire, ...]] = []
    for neuron in neurons:
        bound = []
        for binding in bindings[neuron["neuron_id"]]:
            if binding.delay_us % dt_us:
                raise RuntimeLoadError(
                    "DELAY_NOT_ON_GRID",
                    f"connection {binding.source_id} -> {binding.target_id} declares "
                    f"delay_us={binding.delay_us}, which is not a whole number of "
                    f"dt_us={dt_us} frames; the runtime would have to guess where to "
                    "place the spike",
                )
            bound.append(
                Wire(
                    weight=binding.signed_weight,
                    channel=channel_index[binding.source_id] if binding.source_kind == "channel" else None,
                    neuron=index_of[binding.source_id] if binding.source_kind == "neuron" else None,
                    delay_frames=binding.delay_us // dt_us,
                )
            )
        wires.append(tuple(bound))

    return NetworkPlan(
        neurons=params,
        inputs=tuple(wires),
        order=_evaluation_order(params, wires),
        index_of=index_of,
        decision_index=index_of[decision_neuron],
        n_channels=len(channel_index),
        max_delay=max((w.delay_frames for ws in wires for w in ws), default=0),
    )


def _refractory_frames(neuron: Mapping[str, Any], dt_us: int) -> int:
    """Whole frames of refractory, rounded up.

    Rounding up rather than down keeps the simulated neuron no more excitable
    than the board: a refractory shorter than one frame is unobservable on this
    grid, and rounding it to zero would let the model fire on every frame where
    the hardware could not.
    """
    return -(-neuron["refractory_us"] // dt_us)


def _evaluation_order(neurons: Sequence[NeuronParams], wires: Sequence[Sequence[Wire]]) -> tuple[int, ...]:
    """Topological order over the zero delay edges (Kahn).

    A delayed edge reads the previous frame's history, so it never constrains
    the order within a frame; only the instantaneous edges do. A cycle among
    those is unresolvable rather than merely recurrent, and is refused.
    """
    n = len(neurons)
    predecessors = [
        {w.neuron for w in wires[i] if w.neuron is not None and w.delay_frames == 0} for i in range(n)
    ]
    remaining = [len(p) for p in predecessors]
    successors: list[list[int]] = [[] for _ in range(n)]
    for target, preds in enumerate(predecessors):
        for source in preds:
            successors[source].append(target)

    ready = deque(i for i in range(n) if remaining[i] == 0)
    order: list[int] = []
    while ready:
        i = ready.popleft()
        order.append(i)
        for j in successors[i]:
            remaining[j] -= 1
            if remaining[j] == 0:
                ready.append(j)

    if len(order) != n:
        stuck = sorted(neurons[i].neuron_id for i in range(n) if remaining[i] > 0)
        raise RuntimeLoadError(
            "CYCLIC_TOPOLOGY",
            f"neurons {', '.join(stuck)} form a loop of zero delay connections; "
            "the frame has no evaluation order. Give at least one connection in "
            "the loop a delay_us of one frame to make the recurrence explicit",
        )
    return tuple(order)


class LuiIntegrator:
    """Mutable network state. One instance per session epoch."""

    def __init__(self, plan: NetworkPlan) -> None:
        self.plan = plan
        n = len(plan.neurons)
        self._current = [0.0] * n
        self._membrane = [0.0] * n
        self._observed = [0.0] * n
        self._refractory = [0] * n
        self._past_neurons: deque[tuple[float, ...]] = deque(maxlen=max(plan.max_delay, 1))
        self._past_channels: deque[Sequence[float]] = deque(maxlen=max(plan.max_delay, 1))
        self.reset()

    def reset(self) -> None:
        """Membrane to rest, synapses empty, refractory clear, history dropped."""
        for i, neuron in enumerate(self.plan.neurons):
            self._current[i] = 0.0
            self._membrane[i] = neuron.v_leak
            self._observed[i] = neuron.v_leak
            self._refractory[i] = 0
        self._past_neurons.clear()
        self._past_channels.clear()

    @property
    def membrane(self) -> tuple[float, ...]:
        """Potential as the training stack records it: before the reset.

        The value after a reset says only that the neuron fired, which the spike
        already says. P3 telemetry wants the peak the board actually reached.
        """
        return tuple(self._observed)

    @property
    def currents(self) -> tuple[float, ...]:
        return tuple(self._current)

    @property
    def refractory(self) -> tuple[int, ...]:
        return tuple(self._refractory)

    def step(self, channel_spikes: Sequence[float]) -> tuple[float, ...]:
        """Advance one frame. Returns the spike of every neuron, in plan order."""
        plan = self.plan
        spikes = [0.0] * len(plan.neurons)
        delayed = plan.max_delay > 0

        for i in plan.order:
            neuron = plan.neurons[i]
            injected = 0.0
            for wire in plan.inputs[i]:
                if wire.delay_frames:
                    if not delayed or len(self._past_neurons) < wire.delay_frames:
                        continue  # the session is younger than this delay line
                    source = (
                        self._past_channels[-wire.delay_frames][wire.channel]
                        if wire.channel is not None
                        else self._past_neurons[-wire.delay_frames][wire.neuron]
                    )
                elif wire.channel is not None:
                    source = channel_spikes[wire.channel]
                else:
                    source = spikes[wire.neuron]
                if source:
                    injected += wire.weight * source

            current = neuron.alpha * self._current[i] + injected
            self._current[i] = current

            if self._refractory[i] > 0:
                # The board is clamped while it recovers, but the synaptic
                # capacitor keeps charging, so the current above still advanced.
                self._refractory[i] -= 1
                self._membrane[i] = neuron.v_reset
                self._observed[i] = neuron.v_reset
                continue

            potential = neuron.beta * self._membrane[i] + (1.0 - neuron.beta) * neuron.v_leak + current
            self._observed[i] = potential
            if potential >= neuron.v_threshold:
                spikes[i] = 1.0
                self._membrane[i] = neuron.v_reset
                self._refractory[i] = neuron.refractory_frames
            else:
                self._membrane[i] = potential

        if delayed:
            self._past_neurons.append(tuple(spikes))
            self._past_channels.append(tuple(channel_spikes))
        return tuple(spikes)
