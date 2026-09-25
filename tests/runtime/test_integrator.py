"""The frame-by-frame integrator against the training stack's own evaluation.

The interesting risk in P2 is not the arithmetic, it is the reordering. The
training stack runs every frame of layer H, then every frame of G, then every
frame of O (``snn_hw_pipeline.py:211-241``); a streaming runtime has to run every
layer of frame t before it sees frame t+1. ``test_matches_layerwise_reference``
builds the layer-at-a-time version independently, from the same manifest, and
requires the two to agree bit for bit on a long random stream. If the topological
order were wrong, or a spike leaked into the wrong frame, that test fails.
"""

from __future__ import annotations

import math
import random

import pytest

from snn_runtime import LuiIntegrator, RuntimeLoadError, load_manifest

LAYERS = (("H0", "H1", "H2", "H3"), ("G0", "G1", "G2"), ("D",))


def reference_run(model, frames):
    """The training stack's shape: one whole layer at a time, over all frames.

    A deliberately naive transcription of ``LuiLayer.forward``, kept separate
    from the runtime so that agreement means something.
    """
    spikes = {}
    for layer in LAYERS:
        for neuron_id in layer:
            params = next(n for n in model.manifest["topology"]["neurons"] if n["neuron_id"] == neuron_id)
            alpha = math.exp(-model.dt_us / params["tau_syn_us"])
            beta = math.exp(-model.dt_us / params["tau_mem_us"])
            current, potential, out = 0.0, params["v_leak"], []
            for frame in frames:
                injected = 0.0
                for binding in model.bindings[neuron_id]:
                    source = (
                        frame[model.channel_index[binding.source_id]]
                        if binding.source_kind == "channel"
                        else spikes[binding.source_id][len(out)]
                    )
                    injected += binding.signed_weight * source
                current = alpha * current + injected
                potential = beta * potential + (1.0 - beta) * params["v_leak"] + current
                fired = 1.0 if potential >= params["v_threshold"] else 0.0
                potential = params["v_reset"] if fired else potential
                out.append(fired)
            spikes[neuron_id] = out
    return spikes


@pytest.fixture
def model(lui8):
    return load_manifest(lui8)


def test_matches_layerwise_reference(model, channels):
    rng = random.Random(4242)
    frames = [[1.0 if rng.random() < 0.3 else 0.0 for _ in channels] for _ in range(600)]

    integrator = LuiIntegrator(model.plan)
    streamed = [integrator.step(frame) for frame in frames]
    expected = reference_run(model, frames)

    order = {n.neuron_id: i for i, n in enumerate(model.plan.neurons)}
    for neuron_id, column in expected.items():
        got = [row[order[neuron_id]] for row in streamed]
        assert got == column, f"{neuron_id} diverges from the layer-at-a-time reference"
    assert sum(sum(row) for row in streamed) > 0, "the stream never fired, so nothing was compared"


def test_single_spike_follows_the_closed_form(model):
    """V[t] = v_leak + w * sum_j alpha^j beta^(t-j) after one input spike at t=0.

    Pins the second-order shape itself: a first-order neuron, or one that scaled
    the current by (1 - beta) as the docstring of the training stack claims,
    would give a visibly different series. See units.py.
    """
    neuron = next(n for n in model.plan.neurons if n.neuron_id == "H0")
    index = model.plan.index_of["H0"]
    binding = model.bindings["H0"][0]
    channel = model.channel_index[binding.source_id]
    weight = binding.signed_weight

    integrator = LuiIntegrator(model.plan)
    silence = [0.0] * model.plan.n_channels
    first = list(silence)
    first[channel] = 1.0

    integrator.step(first)
    for t in range(6):
        expected = neuron.v_leak + weight * sum(
            neuron.alpha**j * neuron.beta ** (t - j) for j in range(t + 1)
        )
        assert integrator.membrane[index] == pytest.approx(expected, rel=1e-12)
        integrator.step(silence)


def test_silence_rests_at_v_leak(model):
    integrator = LuiIntegrator(model.plan)
    silence = [0.0] * model.plan.n_channels
    for _ in range(50):
        assert not any(integrator.step(silence))
    for neuron, potential in zip(model.plan.neurons, integrator.membrane):
        assert potential == pytest.approx(neuron.v_leak)


def test_reset_returns_to_the_start(model, channels):
    rng = random.Random(7)
    frames = [[1.0 if rng.random() < 0.4 else 0.0 for _ in channels] for _ in range(80)]
    integrator = LuiIntegrator(model.plan)
    before = [integrator.step(frame) for frame in frames]
    integrator.reset()
    after = [integrator.step(frame) for frame in frames]
    assert before == after


def test_refractory_holds_the_board_down(lui8, mutate):
    """A refractory of two frames must cost two frames of firing."""

    def free_running(manifest):
        for neuron in manifest["topology"]["neurons"]:
            # Rest above threshold and reset just below it, so the leak carries
            # the board back over the line within a single frame whatever its
            # tau_mem. Without a refractory it then fires on every frame.
            neuron["v_leak"], neuron["v_threshold"], neuron["v_reset"] = 1.5, 1.0, 0.99
            neuron["refractory_us"] = 0

    model = load_manifest(mutate(lui8, free_running))
    silence = [0.0] * model.plan.n_channels
    integrator = LuiIntegrator(model.plan)
    assert sum(integrator.step(silence)[model.plan.decision_index] for _ in range(30)) == 30

    def blocked(manifest):
        free_running(manifest)
        for neuron in manifest["topology"]["neurons"]:
            neuron["refractory_us"] = 2 * manifest["runtime"]["dt_us"]

    model = load_manifest(mutate(lui8, blocked))
    integrator = LuiIntegrator(model.plan)
    fired = [integrator.step(silence)[model.plan.decision_index] for _ in range(30)]
    assert fired == [1.0, 0.0, 0.0] * 10


def test_a_delay_off_the_frame_grid_is_refused(lui8, mutate):
    def off_grid(manifest):
        manifest["topology"]["connections"][0]["delay_us"] = manifest["runtime"]["dt_us"] // 2

    with pytest.raises(RuntimeLoadError) as excinfo:
        load_manifest(mutate(lui8, off_grid))
    assert excinfo.value.code == "DELAY_NOT_ON_GRID"


def test_an_instantaneous_loop_is_refused(lui8, mutate):
    def loop(manifest):
        # Feed G0 back into H0's first port with no delay: the frame then has no
        # evaluation order, because each of the two needs the other's spike.
        connections = manifest["topology"]["connections"]
        edge = next(c for c in connections if c["target_id"] == "H0")
        edge["source_kind"], edge["source_id"], edge["delay_us"] = "neuron", "G0", 0

    with pytest.raises(RuntimeLoadError) as excinfo:
        load_manifest(mutate(lui8, loop))
    assert excinfo.value.code == "CYCLIC_TOPOLOGY"


def test_a_delayed_loop_is_allowed(lui8, mutate):
    """Recurrence is fine once one edge in the loop names a frame of delay."""

    def delayed_loop(manifest):
        edge = next(c for c in manifest["topology"]["connections"] if c["target_id"] == "H0")
        edge["source_kind"], edge["source_id"] = "neuron", "G0"
        edge["delay_us"] = manifest["runtime"]["dt_us"]

    model = load_manifest(mutate(lui8, delayed_loop))
    assert model.plan.max_delay == 1
    integrator = LuiIntegrator(model.plan)
    silence = [0.0] * model.plan.n_channels
    for _ in range(20):
        integrator.step(silence)  # must not raise: the delay line covers the loop
