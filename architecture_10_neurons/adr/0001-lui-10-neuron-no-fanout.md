# ADR-0001: Use two-combiner 10-neuron architecture without J4 fan-out

- Status: accepted
- Date: 2026-07-09
- Deciders: lab team
- Relates to: Lu.i 10-neuron glass detection architecture

## Context

Each Lu.i board has three inputs (`J1`, `J2`, `J3`) and one output (`J4`). The
lab needs a quick test architecture for exactly 10 mounted boards. The previous
Peak-vote idea used three copies of the Peak signal, which improved redundancy
but spent too many boards on similar evidence.

## Decision

Use seven L0 feature neurons, two L1 combiners and one L2 decision neuron. Do not
use `J4` fan-out in the baseline wiring. Replace the third Peak copy with a
`ZCR/HF proxy` channel so the network observes impact, fragmentation and
continuity separately.

## Alternatives Considered

### Three Peak copies plus Peak-vote

Pros: simple to calibrate, robust against one mistuned Peak board.

Cons: three boards carry nearly the same information, and the network is weaker
at distinguishing glass from other sharp sounds.

Why not: not the best use of exactly 10 boards.

### Fan-out one L0 output into multiple downstream neurons

Pros: lets one strong feature participate in multiple combiners.

Cons: requires electrical confirmation of output drive and input loading.

Why not: avoid extra uncertainty in the first live lab test.

### Fully direct 7-to-1 decision

Pros: simplest conceptual model.

Cons: one Lu.i neuron has only three inputs, so direct 7-to-1 is impossible
without additional aggregation.

Why not: violates the physical input limit.

## Consequences

### Positive

- Every `J4` output drives exactly one downstream input.
- Every neuron uses at most three inputs.
- The architecture is interpretable during LED-based calibration.
- Failure can be isolated by testing L0, then L1, then L2.

### Negative

- No redundancy for a failed Peak board in the baseline.
- Firmware must expose `ZCR/HF proxy` as a separate output.
- `Peak_strict` must be implemented as a real separate threshold, not just a
  renamed Peak copy.

### Risks and mitigation

If `ZCR/HF proxy` is noisy, start with lower weight on `Fragmentation J3` and
let `peak_counting` dominate that combiner. If `Decision` misses glass, increase
`Impact` first, then `Fragmentation`, and only then reduce continuity inhibition.

