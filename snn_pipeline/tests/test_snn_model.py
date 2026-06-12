import sys
from pathlib import Path
import pytest
import torch
from experiments.train_snn_model import GlassBreakSNN


# Make repo root importable when running nested test files
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def test_forward_zero_input_produces_no_spikes():
    model = GlassBreakSNN()
    model.eval()

    x = torch.zeros(2, 3, 16)
    trigger, spikes = model(x)

    assert trigger.shape == (2, 1)
    assert spikes["input"].shape == (2, 3, 16)
    assert spikes["hidden"].shape == (2, 10, 16)
    assert spikes["output"].shape == (2, 1, 16)
    assert spikes["input"].sum() == 0
    assert spikes["hidden"].sum() == 0
    assert spikes["output"].sum() == 0


def test_forward_accepts_2d_input_and_broadcasts_channels():
    model = GlassBreakSNN()
    model.eval()

    x2d = torch.zeros(2, 16)
    trigger_2d, spikes_2d = model(x2d)

    x3d = x2d.unsqueeze(1).repeat(1, 3, 1)
    trigger_3d, spikes_3d = model(x3d)

    assert torch.equal(trigger_2d, trigger_3d)
    assert torch.equal(spikes_2d["input"], spikes_3d["input"])


def test_forward_output_shapes_are_correct():
    model = GlassBreakSNN()
    model.eval()

    x = torch.rand(4, 3, 12)
    trigger, spikes = model(x)

    assert trigger.shape == (4, 1)
    assert set(spikes.keys()) == {"input", "hidden", "output"}
    assert spikes["input"].shape == (4, 3, 12)
    assert spikes["hidden"].shape == (4, 10, 12)
    assert spikes["output"].shape == (4, 1, 12)


def test_parameter_shapes_match_architecture():
    model = GlassBreakSNN()

    assert model.w_input_hidden.shape == (3, 10)
    assert model.w_hidden_output.shape == (10, 1)
    assert model.w_hidden_hidden.shape == (10, 10)
    assert model.vth_input.shape == (3,)
    assert model.vth_hidden.shape == (10,)
    assert model.vth_output.shape == (1,)


def test_invalid_input_dimensions_raise_value_error():
    model = GlassBreakSNN()
    model.eval()

    with pytest.raises(ValueError):
        model(torch.rand(2, 3, 4, 5))
    with pytest.raises(ValueError):
        model(
            torch.rand(
                2,
            )
        )
