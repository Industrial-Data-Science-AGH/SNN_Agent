import sys
from pathlib import Path
import torch
import pytest
from snn_pipeline.snn_model import GlassBreakSNN


# Make repo root importable when running nested test files
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def test_forward_zero_input_produces_no_spikes():
    model = GlassBreakSNN()
    model.eval()

    # batch=2, channels=1, timesteps=16
    x = torch.zeros(2, 1, 16)
    trigger, spikes = model(x)

    assert trigger.shape == (2, 1)
    # Expect no spikes for zero input
    assert spikes["N1"].sum() == 0
    assert spikes["N2"].sum() == 0
    assert spikes["N3"].sum() == 0
    assert spikes["N_inh"].sum() == 0


def test_get_weights_and_thresholds_dict():
    model = GlassBreakSNN()
    wdict = model.get_weights_dict()
    tdict = model.get_thresholds_dict()

    # keys present and values numeric
    for k in ("w_n1", "w_n2", "w_n3_from_n1", "w_n3_from_n2", "w_inh", "w_inh_to_n3"):
        assert k in wdict
        assert isinstance(wdict[k], float)

    for k in ("vth_n1", "vth_n2", "vth_n3", "vth_inh"):
        assert k in tdict
        assert isinstance(tdict[k], float)


def test_clamp_weights_enforces_bounds():
    model = GlassBreakSNN()

    # set extreme values
    model.w_n1.data.fill_(0.001)
    model.w_inh.data.fill_(0.01)
    model.w_inh_to_n3.data.fill_(-10.0)
    model.vth_n1.data.fill_(0.0)

    model.clamp_weights()

    # After clamping, values should be within configured ranges
    assert 0.05 <= model.w_n1.item() <= 0.95
    assert 0.15 <= model.w_inh.item() <= 0.95
    assert -0.95 <= model.w_inh_to_n3.item() <= -0.20
    assert 0.1 <= model.vth_n1.item() <= 0.95


def test_set_quantize_mode_invalid_raises():
    model = GlassBreakSNN()
    with pytest.raises(AssertionError):
        model.set_quantize_mode("no-such-mode")


def test_quantize_modes_do_not_crash_on_forward():
    model = GlassBreakSNN()
    x = torch.rand(1, 1, 12)

    for mode in ("none", "hat", "gumbel", "qat"):
        model.set_quantize_mode(mode)
        # run forward; ensure it executes and returns expected shapes
        trigger, spikes = model(x)
        assert trigger.shape == (1, 1)
        assert all(k in spikes for k in ("N1", "N2", "N3", "N_inh"))
