#!/usr/bin/env python3
"""Map SNN parameters to hardware resistor suggestions (RV1..RV6).

Assumptions / mappings:
- weights in [w_min,w_max] -> resistors in [R_w_min=500, R_w_max=22222] (Ohm)
  We map higher weight -> lower resistance (inverse mapping).
- tau_syn (ms) -> resistor in [0, R_tau_syn_max=22222] using tau_syn_max=80 ms
- tau_mem (ms) -> resistor in [0, R_tau_mem_max=100000] using tau_mem_max=400 ms
- v_leak -> resistor in [R_vleak_min=1000, R_vleak_max=100000]

This script loads a .pt file (state_dict or checkpoint) and writes
`experiments/hw_param_mapping.csv` with per-neuron resistor suggestions.
"""

import csv
import os
import math
from typing import Dict

import torch


def linear_map(x, in_min, in_max, out_min, out_max):
    if in_max == in_min:
        return (out_min + out_max) / 2.0
    frac = (x - in_min) / float(in_max - in_min)
    return out_min + frac * (out_max - out_min)


def weight_to_resistor(w, w_min=0.05, w_max=1.0, R_min=500.0, R_max=22222.0):
    # higher w -> lower R (conductance increases)
    norm = linear_map(w, w_min, w_max, 0.0, 1.0)
    R = R_min + (1.0 - norm) * (R_max - R_min)
    return R


def tau_to_resistor(tau_ms, tau_max_ms, R_max):
    tau = max(0.0, float(tau_ms))
    frac = min(1.0, tau / float(tau_max_ms))
    return frac * R_max


def vleak_to_resistor(v_leak, vmin=0.0, vmax=0.05, R_min=1_000.0, R_max=100_000.0):
    # map leak conductance proxy to resistor
    norm = linear_map(v_leak, vmin, vmax, 0.0, 1.0)
    R = R_min + (1.0 - norm) * (R_max - R_min)
    return R


def load_state(path: str) -> Dict[str, torch.Tensor]:
    data = torch.load(path, map_location=torch.device('cpu'))
    if isinstance(data, dict) and 'state_dict' in data:
        state = data['state_dict']
    else:
        state = data
    if not isinstance(state, dict):
        raise SystemExit('Unexpected checkpoint format')
    return state


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('path', nargs='?', default='experiments/glassbreak_snn_model.pt')
    p.add_argument('--out', default='experiments/hw_param_mapping.csv')
    args = p.parse_args()

    state = load_state(args.path)

    # find weight tensors
    w_ih = state.get('w_input_hidden')
    if w_ih is None:
        w_ih = state.get('w_input_hidden.weight')
    w_ho = state.get('w_hidden_output')
    if w_ho is None:
        w_ho = state.get('w_hidden_output.weight')
    w_hh = state.get('w_hidden_hidden')
    if w_hh is None:
        w_hh = state.get('w_hidden_hidden.weight')

    vth_input = state.get('vth_input')
    vth_hidden = state.get('vth_hidden')
    vth_output = state.get('vth_output')

    # defaults and bounds (can be adjusted later)
    W_MIN = 0.05
    W_MAX = 1.0
    R_W_MIN = 500.0
    R_W_MAX = 22222.0

    # default physical mapping parameters (user-changeable)
    dt_ms = 1.0
    C_mem_f = 200e-9  # 200 nF default for membrane capacitance
    C_syn_f = 100e-9  # 100 nF default for synaptic capacitance

    # Prepare CSV
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []

    if isinstance(w_ih, torch.Tensor):
        w_ih = w_ih.detach().cpu().float()
        n_in, n_hidden = w_ih.shape
        for j in range(n_hidden):
            w1 = float(w_ih[0, j].item()) if n_in > 0 else 0.0
            w2 = float(w_ih[1, j].item()) if n_in > 1 else 0.0
            w3 = float(w_ih[2, j].item()) if n_in > 2 else 0.0
            R1 = weight_to_resistor(w1, W_MIN, W_MAX, R_W_MIN, R_W_MAX)
            R2 = weight_to_resistor(w2, W_MIN, W_MAX, R_W_MIN, R_W_MAX)
            R3 = weight_to_resistor(w3, W_MIN, W_MAX, R_W_MIN, R_W_MAX)
            # clamp to bounds
            R1 = max(R_W_MIN, min(R_W_MAX, R1))
            R2 = max(R_W_MIN, min(R_W_MAX, R2))
            R3 = max(R_W_MIN, min(R_W_MAX, R3))
            rows.append({
                'neuron': j,
                # Primary output: resistor values for RV1..RV3 (ohm)
                'w1_ohm': R1,
                'w2_ohm': R2,
                'w3_ohm': R3,
                # also keep original weights for reference
                'w1_weight': w1,
                'w2_weight': w2,
                'w3_weight': w3,
            })

    # Try to infer tau and leak from known parameter names
    tau_syn = state.get('tau_syn_ms')
    tau_mem = state.get('tau_mem_ms')
    v_leak = state.get('v_leak')

    # Additionally, snntorch Leaky modules store 'beta' and threshold values in keys like
    # 'lif_hidden.0.beta', 'lif_hidden.0.threshold' — we can try to compute tau_mem from beta
    # using a simple discrete-time approximation: beta ~= exp(-dt / tau_mem)
    # so tau_mem ~= -dt / ln(beta)

    # build beta maps per layer if present
    beta_map = {}
    for k, v in state.items():
        if isinstance(k, str) and k.endswith('.beta') and isinstance(v, torch.Tensor):
            # key example: 'lif_hidden.0.beta' -> extract 'hidden' and index 0
            beta_map[k] = float(v.detach().cpu().item())

    # If rows exist, attach inferred params
    for r in rows:
        idx = r['neuron']

        # tau_syn: prefer explicit tau_syn tensor, else empty
        if isinstance(tau_syn, torch.Tensor):
            t = float(tau_syn[idx].item()) if tau_syn.numel() > idx else float(tau_syn[0].item())
            r['tau_syn_ms'] = t
            # compute resistor from tau = C / g  -> g = C / tau ; R = 1/g
            if t > 0:
                g = C_syn_f / (t * 1e-3)
                R_tau_syn = 1.0 / g if g > 0 else ''
                r['R_tau_syn_ohm'] = R_tau_syn
            else:
                r['R_tau_syn_ohm'] = ''
        else:
            r['tau_syn_ms'] = ''
            r['R_tau_syn_ohm'] = ''

        # tau_mem: prefer explicit, else try to infer from beta
        if isinstance(tau_mem, torch.Tensor):
            t = float(tau_mem[idx].item()) if tau_mem.numel() > idx else float(tau_mem[0].item())
            r['tau_mem_ms'] = t
            if t > 0:
                g = C_mem_f / (t * 1e-3)
                r['R_tau_mem_ohm'] = 1.0 / g if g > 0 else ''
            else:
                r['R_tau_mem_ohm'] = ''
        else:
            # try infer from beta stored in typical keys
            # look for any 'lif_hidden.{idx}.beta' key
            beta_key = f'lif_hidden.{idx}.beta'
            beta_val = None
            for k, v in state.items():
                if k == beta_key and isinstance(v, torch.Tensor):
                    beta_val = float(v.detach().cpu().item())
                    break
            if beta_val is not None and beta_val > 0 and beta_val < 1:
                try:
                    tau_est = -dt_ms / (math.log(beta_val))
                except Exception:
                    tau_est = ''
                r['tau_mem_ms'] = tau_est
                if isinstance(tau_est, float) and tau_est > 0:
                    g = C_mem_f / (tau_est * 1e-3)
                    r['R_tau_mem_ohm'] = 1.0 / g if g > 0 else ''
                else:
                    r['R_tau_mem_ohm'] = ''
            else:
                r['tau_mem_ms'] = ''
                r['R_tau_mem_ohm'] = ''

        # v_leak
        if isinstance(v_leak, torch.Tensor):
            v = float(v_leak[idx].item()) if v_leak.numel() > idx else float(v_leak[0].item())
            r['v_leak'] = v
            r['R_v_leak_ohm'] = vleak_to_resistor(v)
        else:
            r['v_leak'] = ''
            r['R_v_leak_ohm'] = ''

    # write CSV
    if rows:
        keys = list(rows[0].keys())
        with open(args.out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f'Wrote hardware mapping to {args.out} ({len(rows)} neurons)')
    else:
        print('No weight matrix found (w_input_hidden). Nothing written.')


if __name__ == '__main__':
    main()
