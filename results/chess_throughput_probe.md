# Chess RL self-play throughput probe — build-step 0 (ADR 0005)

**Date:** 2026-06-21 · **Machine:** Apple **M3 Max**, 48 GB, macOS 26.5.1 · **Status:** probe complete, **direction GATED GO**.
**Probe:** [`training/training_dynamic/probe_chess.m`](../training/training_dynamic/probe_chess.m) · **Raw run:** [`chess_throughput_probe_raw.txt`](chess_throughput_probe_raw.txt)

This is the single measurement the whole chess-on-ANE project was gated on
([HANDOFF](../docs/chess/HANDOFF.md), [ADR 0005](../docs/adr/0005-chess-rl-self-play-on-ane.md) build-step 0):
can purist-Zero, search-guided self-play climb on **one Mac** in **days vs never**?
Measure first, kernels later.

---

## TL;DR — the verdict: **DAYS, not never. GO to build-step 1.**

- The ANE evaluates **~9,900 chess positions/sec** for the r2_small trunk (batched, fp16),
  **~2,500/sec** for r2_mid. That is the leaf-evaluator throughput the self-play loop runs on.
- Implied self-play rate (pipelined, plies≈100): **r2_small ≈ 0.27–0.54 M games/day** at
  Gumbel sims n=16–32; **r2_mid ≈ 0.07–0.14 M/day**. (Apply a ~0.8× haircut for real-loop
  I/O overhead, below — still 10⁵–10⁶/day.)
- **Surprise vs the handoff's prime suspect:** the loop is **ANE-bound, not CPU-bound.** The
  CPU tree/movegen work is ~0.4 µs/sim/game — *300× too cheap* to be the bottleneck. The
  bottleneck is the **dispatch-bound forward** (18–24 sequential ANE round-trips at ~0.2–0.3 ms each).
- **The lever that turns "never" into "days" is batching.** Per-position cost falls **44×**, from
  4.4 ms (B=1) to **0.10 ms (B=170)**. Without batching, a real Elo climb is ~year ≈ never;
  with it, ~a week.
- **Two hard ANE walls found** (both motivate future kernels, neither blocks the build):
  1. **Batch ceiling B ≤ 170** — the ANE rejects any tensor dim > **16384** (seq=16384 OK,
     16416 REJECTED). Caps leaves-per-forward at 170.
  2. **Dispatch tax** — the forward is 3 evals/layer × ~0.2 ms floor, so even a *tiny* net pays
     ~5 ms/forward. Fewer evals (fusion) is the lever, exactly as the LM side already found.

**Recommendation:** proceed down the build order. Start the trunk at **r2_small** (DIM=256, 6L,
~13 M with heads) — ~4× the throughput of r2_mid, and ADR #12 says scale up only after the loop
is proven. Run the self-play batch at **B≈128–170** (the ceiling) and **n=16–32** sims to start.
Do **not** pre-emptively build kernels — but log the two walls above as the first kernel targets
once the loop is green (G2).

---

## What was measured, and how (method + honesty)

The dynamic trainer's forward is, verified in `train.m`: **3 ANE evals per layer** —
`sdpaFwd` (QKV-proj+RoPE+attention), `woFwd` (output matmul), `ffnFused` (W1/W3+SiLU+W2) —
with RMSNorm, embedding and the head on **CPU**. Activations are `[1, C, 1, SEQ]` with **no batch
dimension**. So to batch **B** leaf positions into one forward, the only available axis is the
**spatial (token) dim**: pack `seq = B × SEQ` for the token-wise projection matmuls (attention
folds B into the head dim). This is the dispatch-bound "batch big, few evals" lever.

The probe is **model-agnostic** (like `probe_dispatch.m`): it drives **chess shapes at runtime**
through `gen_dyn_matmul_mil(ic,oc,seq)`, so it sweeps B without recompiling the monolithic kernels
(which need on-disk RoPE/mask blobs). It reconstructs the real **3-dispatch/layer** forward as
`L × (3 × floor + Σ compute-above-floor of {qkv, wo, ffn-up, ffn-down}) + attention`, where
attention is priced analytically from the measured matmul GFLOP/s **[flagged: estimate]**.

**One ANE client. fp16. Warm engine** (cold-start tax removed before any timing).

### Anchor — the decomposed model matches the *real* fused forward (within ~15%)

Running the actual r2_small trainer (`make MODEL=gen_r2_small && ./train --scratch`) reports, per step:

```
timing: ane_fwd=5.3  io_fwd=0.8  rms=0.4  ane_bwd=10.2  cls=9.9   (ms)   step ≈ 30.6 ms/step
```

`ane_fwd = 5.3 ms` is the pure-ANE time for the 18-eval forward at **SEQ=256**. The probe's
decomposed model gives **t_ane = 4.4 ms at seq=96** (B=1); scaled to seq=256 it lands at ~5.3 ms.
**The ANE-compute estimate is faithful** — I am not under-counting the ANE. The real forward
*additionally* pays `io_fwd+rms ≈ 1.2 ms` (~23%) that the bare-matmul probe omits; an
inference loop with **chained IOSurfaces** (output of eval N = input of eval N+1, no CPU readback)
would avoid most of it. I treat the probe's t_ane as the **optimistic (chained) floor** and apply
a conservative **0.8× haircut** to games/day for the un-chained reality.

---

## Results

### [0] The two hard ANE facts (measured live on this M3 Max)

| Fact | Value | Evidence |
|---|---|---|
| **Per-eval dispatch floor** (warm) | **≈ 0.20 ms/eval** | min over tiny warm matmuls; prior/memory said 0.12 ms — this box is 0.20 |
| **Max spatial (token) dim → batch ceiling** | **seq ≤ 16384 → B ≤ 170** | seq=16320/16384 **OK**; 16416/24576/32768 **REJECTED** (`ANECCompile FAILED`) |

### [A] ANE batched forward-only eval — the batch-scaling curve

**r2_small** (DIM=256, HEADS=8, HD=32, HIDDEN=768, **L=6**, ~5.1 M trunk params):

| B | fwd ms | **ms/position** | **positions/sec** |
|--:|--:|--:|--:|
| 1   | 4.41  | 4.408 | 227 |
| 16  | 4.06  | 0.254 | 3,943 |
| 64  | 8.10  | 0.127 | 7,905 |
| 128 | 13.19 | 0.103 | 9,704 |
| **170** | 17.10 | **0.101** | **9,940** |

**r2_mid** (DIM=512, HEADS=16, HD=32, HIDDEN=1408, **L=8**, ~25.7 M trunk params):

| B | fwd ms | ms/position | positions/sec |
|--:|--:|--:|--:|
| 1   | 5.00  | 4.995 | 200 |
| 64  | 29.73 | 0.465 | 2,153 |
| 128 | 52.56 | 0.411 | 2,435 |
| **170** | 67.00 | **0.394** | **2,537** |

**Reading:** batching amortizes the dispatch tax **44×** (r2_small) up to the B≤170 ceiling, where
per-position cost plateaus (now compute/bandwidth-bound, not dispatch-bound). r2_small is **~4×**
the throughput of r2_mid — more games per day at the same wall-clock.

### [B] CPU self-play orchestration stub (movegen + Gumbel-MCTS bookkeeping)

| B | ms/sim-step | µs/sim/game |
|--:|--:|--:|
| 1   | 0.0005 | 0.47 |
| 64  | 0.027  | 0.43 |
| 170 | 0.075  | 0.44 |

The stub (bitboard-flavored movegen → ~40 moves, depth-12 PUCT descent + expand + backup) costs
**~0.4 µs/sim/game**, flat in B. **Break-even:** the CPU only becomes the bottleneck if real
`(movegen + tree)/sim/game` exceeds `t_ane(B)/B` = the ms/position above, i.e. **~140 µs/sim** at
r2_small B=64 — **~300× the stub**. Even a pessimistic real bitboard generator (~2–5 µs/sim with
make/unmake) stays comfortably **ANE-bound**. *(Movegen realism is the C-engine/perft gate,
build-step 1 — flagged; but the 300× margin means it cannot flip the verdict.)*

### [C] Combine → games/day

`games/day = B × 86.4e6 / (plies × n × max(t_ane(B), t_cpu(B)))`, plies≈100 **[est]**, pipelined.

**r2_small** (selected cells; full table in the raw run):

| B | n | bound | games/day (pipelined) | × 0.8 haircut |
|--:|--:|:--:|--:|--:|
| 170 | 16 | ANE | 536,757 | ~430,000 |
| 170 | 32 | ANE | 268,379 | ~215,000 |
| 128 | 16 | ANE | 524,007 | ~419,000 |
| 64  | 32 | ANE | 213,426 | ~171,000 |
| 170 | 64 | ANE | 134,189 | ~107,000 |
| **1** | 16 | ANE | **12,250** | (no-batch baseline) |

**r2_mid:** B=170/n=16 → 137,018; n=32 → 68,509; n=64 → 34,255 games/day (pipelined).

### Cross-check — is it really **ANE-bound** (not a silent CPU fallback)?

The two-signal rule. The definitive signal is ANE power (needs sudo — command below), but the
**no-sudo signal is decisive on its own:** while hammering the forward (`--sustain --B=128`), the
process holds at **6.5–7.5 % CPU** — it is *blocking* on `evaluateWithQoS`, not spinning a core. A
silent CPU fallback would peg ~100 %+. The sustained rate (0.648 ms/eval) also matches the sweep's
ffn-down at B=128 (0.627 ms). **Confirmed: the work is genuinely on the ANE.**

> Definitive ANE-power confirmation (run in this session via the `!` prefix; needs sudo):
> `! sudo powermetrics --samplers ane_power -i 1000 -n 20` while, in another shell,
> `./probe_chess --sustain --B=128 --sec=25` runs. Expect non-zero ANE power during the hammer.

---

## The honest read — days vs never

**Why ANE-bound, and why that's fine.** At chess scale (seq≈77, d=256) the net is so small that the
forward is *entirely* dispatch-bound: 18–24 **sequential** ANE round-trips at ~0.2–0.3 ms each →
~5 ms/forward *no matter how tiny the net is*. The CPU tree work is three orders of magnitude
cheaper, so it cannot dominate. The handoff's prime suspect (CPU movegen) is **not** the
bottleneck — but the *reason* is benign: the ANE leaf-evaluator is the long pole, and it still
delivers ~10⁴ evals/sec, comparable to a low-end GPU for a net this size.

**Why "days, not never."** Map throughput to the gate ladder:
- **G0 / G1 / G2** (heads overfit; search solves tactics; win-rate climbs vs a random-mover) need
  **~10³–10⁵ games**. At ~10⁵–10⁶ games/day that is **hours to ~1 day**. Cleared with large margin.
- **G3** (a real self-anchored Elo climb, beating *capped* Stockfish — the modest yardstick, not
  grandmaster) is the sample-hungry part: **~10⁶–10⁷ games**. At ~2×10⁵/day (r2_small, n=32, haircut)
  that's **~5–50 days** — a bounded cold-start desert, **not "never."**
- The **"never" counterfactual** is real and instructive: at **B=1** (no batching) the dispatch tax
  gives only ~227 pos/s → a 4 M-game Elo climb takes **~1 year ≈ never**. **Batching to the B≤170
  ceiling is precisely what converts never → a week.** That ceiling is therefore the single most
  important number this probe produced.

**Net-size call.** r2_small is ~4× r2_mid's throughput. Cold-start is sample-hungry, so *more games*
(r2_small, or even the optional ~2–5 M proof-of-life net for the fastest G2 smoke-test) beats *more
parameters* early. This matches ADR #12 ("scale up only after the loop is proven").

---

## Caveats — every estimate, flagged

1. **Optimistic ANE path.** t_ane is the chained-IOSurface floor; the current trainer pays +~23 %
   (io_fwd+rms, **measured**). Real games/day ≈ 0.8× the table → still 10⁵–10⁶/day. A chained-surface
   inference path (no per-eval CPU readback) is needed to hit the un-haircut numbers — that's
   build-step 4 engineering, not a kernel.
2. **Decomposed forward model**, validated against the real `ane_fwd` within ~15 % (anchor above).
   Attention priced analytically (~10–15 % of FFN compute) **[estimate]**.
3. **plies≈100, MCTS-depth≈12 [estimates].** games/day ∝ 1/plies; a 150-ply average → ×0.67.
4. **n is the dominant lever and a throughput⇄quality tradeoff.** Low n (16, Gumbel-low) → cheap
   games but weaker policy targets → more games needed for the same learning. High n (64–128) →
   stronger targets, 4–8× fewer games/day. The *learning-efficiency* side is unmeasurable here
   (it's a G2/G3 question), so the games/day table is **necessary, not sufficient** — it bounds the
   sample budget, not the samples-to-learn.
5. **Learner contention.** The learner time-shares the one ANE client; a training step is ~30 ms
   (r2_small, measured: ane_fwd 5.3 + ane_bwd 10.2 + opt/cls). In AZ the self-play evals dwarf the
   learner's, so this is a small tax — but the real loop interleaves them; the probe measures pure
   self-play throughput. The chess head (4,672-wide policy) is far cheaper than the LM head
   (vocab 32 k, `cls=9.9 ms`), so chess training steps will be cheaper than the numbers above.
6. **Movegen realism** is the C-engine/perft gate (build-step 1); the stub is a placeholder. The
   300× break-even margin means it cannot flip ANE-bound → CPU-bound at B≤170.
7. **Floor = 0.20 ms** here vs the project memory's 0.12 ms — re-measured warm on this M3 Max;
   reported as the live number, not the prior.

---

## Decision & next step

**Gate: PASS.** games/day clears the bar (G2 in hours, G3 in days-to-weeks) with multiple-× margin,
and the loop is ANE-bound with the CPU 300× clear of the bottleneck. **Proceed to build-step 1**
(C bitboard engine + Gumbel-MCTS, **perft**-green) per the ADR build order — one session per gate.

**Carry-forward for the kernel phase (after G2, not before):** the two ANE walls are the first
optimization targets, in priority order —
1. **Fuse the forward** (fewer evals/layer) to cut the dispatch tax — the same lever the LM side
   already validated (qkvBwd −2 ms, SiLU-bwd −2 ms). A fully-fused chess forward could approach
   1 eval/layer → several-× throughput.
2. **Break the B≤170 ceiling** — a real batch dim, or virtual-loss sub-batching to put more leaves
   in one forward, would push past the 16384 spatial limit and amortize further.

## Reproduce

```bash
cd training/training_dynamic
make probe_chess                       # model-agnostic; drives chess shapes at runtime
./probe_chess                          # full sweep + games/day table
./probe_chess --sustain --B=128 --sec=25   # ANE-power hammer (run powermetrics alongside)

# Anchor (real fused forward timing):
make MODEL=gen_r2_small && ./train --scratch --steps 25 --data ../tinystories_data00.bin
```
