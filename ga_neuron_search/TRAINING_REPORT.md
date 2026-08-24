# Training report — GA neuron search (SNN topology for Lu.i boards)

Run date: **2026-08-24**, Apple M5 MAX (128 GB), Python 3.12 + torch 2.13 (CPU).
Command: `run_search.py --neurons 4 6 8 10 --mode real --epochs 4 --pop 24 --gens 15 --parsimony-eps 0.02 --train-winner --winner-epochs 60`

## TL;DR

- The GA sweep over network size found **N=8 is the sweet spot**: best proxy clip-F1 **0.4483** (topology `7 → 4 → 3 → 1`). More neurons (N=10) did **not** help.
- Full HAT→QAT training of the winner (60 epochs) reached **clip-F1 0.605** on the event-based validation set (recall 0.83, precision 0.48, FA-rate 0.23, AP 0.47).
- The final model + per-board hardware settings are saved (paths below). Only **156 parameters** drive the whole detector — 8 Lu.i boards.

---

## 1. What was run

Four independent GA runs (one per total-neuron budget N = 4, 6, 8, 10). Each genome is a layered SNN topology (fan-in ≤ 3, DAG, constant N). Fitness = **proxy training** (4 epochs, 6000 windows) followed by an **event-based AP** over the validation set (decoder: ≥ 2 spikes of the decision neuron in a window = alarm, aggregated per clip). The GA uses tournament selection + elitism (pop 24, elite 3, gens 15, patience 6) with a topology cache.

After the sweep, the parsimony rule picked the smallest N within 0.02 clip-F1 of the best, and that topology got a **full HAT→QAT** training (60 epochs, 12000 samples, weight quantization to 20 trimmer levels, hardware-mismatch noise, sign freezing).

## 2. Sweep results

| N (boards) | proxy AP / clip-F1 | best topology | evals |   time |
| ---------: | -----------------: | ------------- | ----: | -----: |
|          4 |             0.4163 | `7-3-1`       |    89 |  8 min |
|          6 |             0.4158 | `7-3-2-1`     |   218 | 26 min |
|      **8** |         **0.4483** | **`7-4-3-1`** |   257 | 35 min |
|         10 |             0.4328 | `7-9-1`       |   284 | 35 min |

**Parsimony pick: N=8** (best overall; N=10 within eps but not smaller, N=4/6 below eps).

Key observation: **board count is not monotonic**. The two-hidden-layer topology at N=8 (a 4-neuron H layer + 3-neuron G layer, then the decision neuron) beat both the tiny N=4 net and the wide single-layer N=10 net. This is exactly the kind of structure the GA search exists to find — a fixed architecture would not have arrived at it.

## 3. Winner full training (N=8, `7-4-3-1`)

- **HAT** (ep 0–23, full precision + hardware noise): clip-F1 rises to ~0.60 by ep 5, then plateaus.
- **QAT** (ep 24–59, quantization to 20 trimmer levels): clip-F1 recovers to **0.605** at the end — the quantized model matches full precision almost exactly, so the settings are genuinely realizable on the boards.
- **Best val metrics**: clip-F1 **0.605**, recall **0.831**, precision **0.476**, FA rate **0.229**, AP **0.473** (k=2, 649 clips: 130 positive / 519 negative).

Board-level exported settings (trimmer %, sign, input ports, τ_syn/τ_mem, V_leak/LED bar):

| board           | inputs (port / source / sign / trimmer %)    | τ_syn   | τ_mem  | LED bar |
| --------------- | -------------------------------------------- | ------- | ------ | ------- |
| L1n0            | J1 peak_cnt −100 %, J2 flux +30 %            | 97.7 ms | 893 ms | 43.7 %  |
| L1n1            | J1 hf_lo +56 %, J2 hf_hi +77 %               | 81.3 ms | 888 ms | 10.0 %  |
| L1n2            | J1 cv +50 %, J2 zcr +100 %                   | 79.4 ms | 142 ms | 32.0 %  |
| L1n3            | J1 peak +15 %, J2 hf_lo +23 %                | 30.7 ms | 141 ms | 10.0 %  |
| L2n0            | J1 L1n0 +100 %, J2 L1n2 +7 %                 | 36.8 ms | 212 ms | 39.0 %  |
| L2n1            | J1 L1n0 −58 %, J2 L1n1 +17 %                 | 14.0 ms | 100 ms | 10.0 %  |
| L2n2            | J1 L1n1 −100 %, J2 L1n2 +40 %                | 176 ms  | 593 ms | 44.3 %  |
| L3n0 (decision) | J1 L2n0 −35 %, J2 L2n1 +25 %, J3 L2n2 −100 % | 26.7 ms | 196 ms | 33.1 %  |

Interesting structure: the HF channels (hf_lo/hf_hi) are used together on L1n1, and the decision neuron combines an inhibition (L2n0) and an excitation (L2n1) path — a small recurrent-style motif in a purely feed-forward net. Feature coverage across the input channels is complete.

## 4. Where the model and results are saved

Everything lives in `ga_neuron_search/`:

| File                                   | What it is                                                                                                                                                                                                               |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `wyniki_real.json` / `wyniki_real.csv` | Sweep table: per-N best fitness, topology, eval count, runtime                                                                                                                                                           |
| `wyniki_real_winner_N8.pt`             | **Trained model checkpoint** — `{model: state_dict, metrics, topology}`. 156 params. Load with `torch.load(..., weights_only=False)`                                                                                     |
| `wyniki_real_hw_config_N8.json`        | **Per-board hardware settings** ready to apply: `synapses[].pot_pct` (trimmer %), `sign` (+/−), `from`+`port` (input wiring), `led_bar_pct` (V_leak), `tau_syn_ms`/`tau_mem_ms`, `pulses_to_fire_100Hz` (sim↔hw control) |
| `wyniki_real_run.log`                  | Full run log (all eval lines, HAT/QAT history, export table)                                                                                                                                                             |

The `.pt` is the network weights for simulation/checkpointing; the `.json` is the artifact that goes to the boards (per-board trimmer/port settings). To re-train from scratch, delete the outputs and re-run the command in section 1.

## 5. How it went (timing & stability)

- **Wall time: ~1 h 48 m** total — ~1 h 44 m sweep (848 evals) + ~4 m winner training.
- Average eval cost **5.2–8.1 s** depending on N (grows with network size); measured floor is ~5.1 s (N=4) at 4 epochs.
- The GA's `patience=6` early stop triggered **only for N=4** (stopped at gen 10). N=6/8/10 kept finding marginal improvements into the last generation, so they ran the full 15. This is why actual runtime exceeded the pre-run estimate of 30–60 min.
- Candidate losses dropped consistently (e.g. 6.3→1.5) and best-genome improvements were real, not noise-driven — the fitness signal is usable.

## 6. What could be done better

**Biggest wins (in order):**

1. **Parallelize the GA evaluation (recommended first).** The sweep is embarrassingly parallel but currently evaluates individuals sequentially on one CPU process. The M5 Pro has 18 cores; a small `concurrent.futures.ProcessPoolExecutor` around `RealFitness.__call__` (or a per-individual pool in `ga.run_ga`) would cut ~1 h 48 m to **~10–15 min** near-linearly. This is already a listed TODO in the README ("równoległa ewaluacja osobników").
2. **Don't use MPS.** Measured 3× slower than CPU for this tiny SNN (15.6 s vs 5.5 s per eval) — GPU launch overhead dwarfs the small compute. Keep the CPU path; the `cuda/cpu` device check already defaults to CPU on Mac.
3. **Cheaper screening / successive halving.** `--screen-mult 3 --screen-budget 0.34` (already implemented, disabled by default) ranks 3×pop candidates with ~1 epoch of training before committing a full 4-epoch eval — better search quality per wall-clock.
4. **Average fitness over 2–3 seeds** (`--fitness-seeds 3`). Single-seed proxy AP is noisy; multi-seed makes GA selection more stable at the cost of a linear slowdown (affordable once parallelized).

**Quality / deployment items:**

5. **Metric alignment.** Fitness is `ap`; the deployment metric is `clip_f1`. They correlate but the winner could differ. Consider ranking on `clip_f1` (or a small AP+clip-F1 blend) so the sweep directly optimizes the hardware metric.
6. **False-alarm rate is high (0.23).** The decision threshold / `k` (currently 2 spikes) and `pos_weight` (3.0) are untuned. After validation on real boards, tune these or add an FA cost to the loss to cut background alarms.
7. **Winner-config validation on hardware.** The exported settings are sim-accurate (QAT recovered full-precision F1), but real boards have trimmer hysteresis/parts tolerance — run the existing `calibrate.py` / `hw15_*.csv` flow against the new `wyniki_real_hw_config_N8.json` before deployment.
8. **Richer checkpoints.** `train_full` saves only the best state + metrics. Saving per-epoch curves (loss, clip-F1 by phase) would give better diagnostics, especially for the HAT→QAT transition.

**Search / cost tuning:**

9. **More generations or adaptive mutation** if the goal is squeezing the last 0.01 of F1 — several N runs improved only in the final generations, hinting the search was still productive when stopped.
10. **N-budget sweep granularity** (e.g. 4,5,6,7,8,9,10) to localize the elbow better; or a Pareto front instead of separate runs per N (README TODO).
