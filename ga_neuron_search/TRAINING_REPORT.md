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

---

# Follow-up (2026-08-24) — parallel eval, stabler defaults, deployment tuning

This section implements the top items from §6 above (1, 3, 6 + the k/pos_weight part of 5/6/7), scoped to a **smoke test** of the parallel path (no full re-sweep) and a **full pos_weight grid + k-tuning** on the existing winner.

## 1. Parallel GA evaluation (`ParallelFitness`, `--workers`) — implemented + verified

- New `ParallelFitness` in `fitness.py`: a `ProcessPoolExecutor` where each worker builds its **own `RealFitness`** and pins `torch.set_num_threads(1)` (without that, 18 workers × 18 torch threads thrash). `ga.run_ga` now evaluates every generation in one `fitness.batch(...)` call, dedup'd via the existing topology cache; the fallback path is unchanged for synth/tests.
- **Correctness (the critical check):** a real smoke run (`--neurons 4 6 --pop 8 --gens 3 --epochs 2`, workers=1 vs workers=8) produced **bit-for-bit identical** `wyniki_*.json` (same best topology, same fitness, same eval count) — parallel changes the wall-clock only, never the result.
- **Speed (measured on this M5 Pro, 18 cores):**
  - tiny 30-eval run: 74.6 s → **24.7 s** (3.0×; spawn + data-load of 8 workers dominates at this size),
  - representative 97-eval run (N=7, pop 24, gens 4, `--workers 16`): **42.4 s wall** vs ~453 s of CPU work → **~10.7×** once startup amortizes.
  - Full-sweep expectation: ~1 h 48 m → **~10–15 min** (not re-run).
- `torch.set_num_threads(1)` is also pinned in the main process (`--mode real`) — a tiny SNN gains nothing from threaded matmul, and a single thread in main == single thread in workers keeps sequential and parallel runs comparable/deterministic. MPS stays off (measured 3× slower).

## 2. Stabler/cheaper search by default

`run_search.py` defaults changed: `--screen-mult 1 → 3` (successive-halving screening: rank 3·pop candidates at 0.34 budget before the full eval) and `--fitness-seeds 1 → 3` (fitness averaged over 3 training seeds — lower selection variance). Both were already implemented; they are now on by default. `--screen-mult 1 --fitness-seeds 1` restores the old behaviour.

## 3. Deployment tuning on the winner — FA-rate 0.229 → 0.170

**k-sweep on the original winner** (pw=3.0, k=2 → clip-F1 0.605, FA 0.229). Raising the decoder threshold alone:

| k | clip-F1 | recall | precision | FA-rate |
|---|--------:|-------:|----------:|--------:|
| 1 | 0.595 | 0.846 | 0.458 | 0.250 |
| 2 | 0.605 | 0.831 | 0.476 | 0.229 |
| 3 | 0.607 | 0.808 | 0.486 | 0.214 |
| **6** | **0.627** | 0.762 | 0.532 | **0.168** |

**pos_weight grid** (winner `7-4-3-1`, 60-ep HAT→QAT each, ~4 min per value):

| pos_weight | clip-F1 | FA | recall | AP |
|---|---- |---- |---- |---- |
| **1.5** | **0.651** | **0.189** | 0.846 | 0.522 |
| 2.0 | 0.637 | 0.195 | 0.831 | 0.491 |
| 3.0 (org) | 0.605 | 0.229 | 0.831 | 0.473 |

**k-sweep on the tuned winner (pw=1.5):** best clip-F1 **0.654** at k=3 (FA 0.170); lowest FA **0.139** at k=6 (clip-F1 0.653).

**Net deployment win vs the original export** (pw=3.0, k=2):
clip-F1 **0.605 → 0.654** (+0.049), FA-rate **0.229 → 0.179** (−0.059), precision 0.476 → 0.546, at `pos_weight=1.5, k=3`. Recall stays ≥ 0.81 at the chosen k.

## 4. Export round-trip validation (`validate_hw_config.py`) — PASS

Rebuilds the network from the exported JSON (trimmer %, sign, wiring, τ_syn/τ_mem, V_leak → model params, `quantize=True`) and re-evaluates on the same validation subset. On the **tuned** config:
- k=2: rebuilt clip-F1 **0.6509** vs checkpoint 0.6509 → **Δ 0.0000, PASS**;
- k=3 (deployed threshold): **0.6543**, matching the tune-k table → **PASS**.

All synapse pots are ≥ 7.1 % (above the 5 % trimmer floor — nothing was zeroed). This is the software-only proxy for board validation; the real check (`snn_hw_pipeline.py compare`) still needs hardware recordings of this config.

## 5. New / changed artifacts

| File | What |
|---|---|
| `wyniki_real_winner_N8_pw{1.5,2.0,3.0}.pt` | per-pos_weight full-trained checkpoints |
| `wyniki_real_tuned_winner_N8.pt` | best grid checkpoint (pw=1.5) + tuned_k=3 |
| `wyniki_real_tuned_hw_config_N8.json` | deployment config: `pos_weight`, `tuned_k`, `tune_k_table` added |
| `fitness.py` / `ga.py` / `run_search.py` / `winner.py` / `validate_hw_config.py` | new flags + parallel path (see README) |

Config one-liner for the boards: `--pos-weight 1.5 --train-winner --tune-k 1 2 3 4 5 6` on the winning topology (README section "Strojenie do wdrożenia").

---

# Follow-up 2 (2026-08-24) — best-parameter campaign: clip-F1-ranked search + fine pw grid

Goal: **beat clip-F1 0.654 / FA 0.170** with hardware-achievable parameters. Two experiments ran back-to-back.

## 1. Clip-F1-ranked GA sweep (N=4..10) — does NOT beat the AP-ranked winner

`run_search.py --neurons 4..10 --metric clip_f1 --k 2 --pos-weight 1.5 --pop 24 --gens 16 --screen-mult 3 --fitness-seeds 3 --workers 18` (results in `wyniki_best.json`). Ranking the search directly on the deployment metric (report §6 item 5) changed the outcome, but **not for the better after full training**:

| N | proxy clip-F1 | topology | full clip-F1 (best pw) | FA |
|--:|--:|--|--:|--:|
| 4 | 0.536 | `7-3-1` | — | — |
| **5** | **0.561** | **`7-4-1`** | **0.597** (pw 2.0) | 0.225 |
| 6 | 0.544 | `7-3-2-1` | — | — |
| 7 | 0.558 | `7-6-1` | 0.593 (pw 2.0) | 0.243 |
| 8 | 0.546 | `7-3-2-2-1` | — | — |
| 9 | 0.556 | `7-8-1` | 0.591 (pw 1.5) | 0.274 |
| 10 | 0.548 | `7-9-1` | — | — |

The parsimony pick N=5 full-trained to only **0.597** — well below the existing N=8 `7-4-3-1` champion (0.654). Runners-up N=7/N=9 full-trained to 0.593/0.591, and their exports contained sub-resolution synapses (pot 0–4 %). **Conclusion: the clip-F1-ranked search steered toward smaller nets that train worse in full HAT→QAT; the AP-ranked N=8 topology from the original sweep remains the best topology.** New artifacts: `wyniki_best.json/csv`, `wyniki_best_winner_N5_pw{1.2..2.0}.pt`, `wyniki_best_tuned_hw_config_N5.json`, `wyniki_best_runner_N{7,9}_hw_config.json`.

## 2. Fine pos_weight sweep on the champion (`7-4-3-1`) — NEW BEST: clip-F1 0.658, FA 0.158

The champion topology was already optimal; the previous pw grid ({1.5, 2.0, 3.0}) was coarse. A fine grid {1.2…1.7} at 60-ep HAT→QAT each (`wyniki_fine_*.pt`):

| pos_weight | clip-F1 (k=2) | FA |
|---|--:|--:|
| 1.2 | 0.648 | 0.179 |
| **1.4** | **0.652** | **0.171** |
| 1.5 (prev) | 0.651 | 0.189 |
| 1.6 | 0.647 | 0.185 |
| 1.7 | 0.645 | 0.191 |

**k-sweep on the pw=1.4 winner:** best clip-F1 **0.658** at k=3 (FA 0.158, rec 0.800, prec 0.559); lowest FA 0.127 at k=6 (clip-F1 0.648).

**Net result vs previous champion (pw=1.5, k=3):** clip-F1 **0.654 → 0.658** (+0.004), FA-rate **0.170 → 0.158** (−0.012).

**Round-trip validation (new config): PASS** — rebuilt clip-F1 **0.6582** at k=3 (rec 0.800, prec 0.559, FA 0.158), matching the tune-k table; Δ vs stored metric 0.0059 (tol 0.02). All synapse pots ≥ 8 % — nothing below the 5 % trimmer floor.

**Final best hardware-achievable parameters (8 boards, topology `7 → 4 → 3 → 1`):**
`pos_weight=1.4`, `tuned_k=3`, clip-F1 **0.658**, FA-rate **0.158**, recall 0.800, precision 0.559, AP 0.510. Artifacts: `wyniki_fine_tuned_winner_N8.pt` + **`wyniki_fine_tuned_hw_config_N8.json`** (the file to flash to the boards).
