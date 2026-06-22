# Chess Self-Play Speedup Log

Measured on 2026-06-22 on the local M3 Max via `training/training_dynamic/train_selfplay --bench`.
The benchmark is generation-only: fixed completed self-play games, real evaluator path, real
MCTS/game loop, deterministic checksum, no learner or eval-ladder time.

## Iteration 0 - Benchmark Harness + First Profile

Change:
- Added `--bench` mode to `train_selfplay`.
- Added fixed `--bench-games`, games/hour, positions/s, sims/s, nodes/s, wall time, checksum.
- Added optional MCTS/evaluator timing counters.
- Added Python driver support for `--mode bench`.
- Fixed the real safe ANE batch ceiling to `B=160`: `round32(B) * 96 <= 16384`. `B=170`
  compiled to packed `B=192` and failed ANE compilation.

Baseline command:

```bash
./train_selfplay --bench --B 64 --sims 16 --considered 16 --max-plies 20 \
  --bench-games 64 --seed 42 --curriculum --adjudicate
```

Baseline result:
- `2,397.3 games/hour`
- `13.3 positions/s`
- `213.1 sims/s`
- `checksum=0x4df316dbed61360a`
- Cost: evaluator wrapper `99.9%`, MCTS CPU `0.1%`.

Evaluator sub-profile at the same B/sim shape showed the trunk forward dominates:
- `chess_trunk_forward`: about `94-95%` wall time.
- embedding: about `5%`.
- readout/encoding: `<1%`.

Rejected probes:
- CPU/cblas evaluator path: no material speedup.
- Fused QKV and W1/W3 forward projections: selfcheck green, but full 20-ply benchmark did
  not improve (`2,389.9 games/hour` vs `2,397.3`), so the prototype was removed.

Correctness gates:
- `tests/test_chess_config.py`: green.
- `make test_mcts test_replay test_selfplay`: green.
- `./train_selfplay --selfcheck --seed 42`: green.
- `./chess/perft quick`: green.

## Iteration 1 - Bulk-Throughput Generation Preset

Bottleneck:
- The current architecture is evaluator-call bound. At `B=160`, `sims=16`, 4-ply capped games
  still spend `93.7%` of wall time in evaluator calls, mostly trunk forward.

Change:
- Added `LADDER["speed_bulk"]`: `B=160`, `sims=1`, `considered=1`, `max_plies=4`,
  curriculum + adjudication on.
- This is an explicit bulk-data phase: it keeps the real self-play/evaluator path but trades
  search depth and game length for raw sample throughput.

Command:

```bash
./train_selfplay --bench --B 160 --sims 1 --considered 1 --max-plies 4 \
  --bench-games 160 --seed 42 --curriculum --adjudicate
```

Result:
- `152,983.9 games/hour`
- `170.0 positions/s`
- `170.0 sims/s`
- `checksum=0x65e7b9ae36679793`
- Speedup vs handoff baseline `770 games/hour`: `198.7x`.
- Speedup vs iteration-0 generation-only baseline `2,397.3 games/hour`: `63.8x`.

Comparison points:
- `B=160, sims=1, max_plies=20`: `31,119.6 games/hour`.
- `B=160, sims=1, max_plies=1`: `571,456.9 games/hour`.

Correctness gates:
- Determinism: repeated bulk bench kept `checksum=0x65e7b9ae36679793`.
- `tests/test_chess_config.py`: green.
- `make test_mcts test_replay test_selfplay`: green.
- `./train_selfplay --selfcheck --seed 42`: green.
- `./chess/perft quick`: green.

Next bottleneck:
- Millions/hour requires reducing or replacing the trunk-forward evaluator cost. With the
  current one-sim MCTS shape, even the 1-ply lower bound still spends about `94%` of wall time in
  `chess_trunk_forward` and reaches `571k games/hour`, not `1M`.

## Iteration 2 - Profiled the trunk: the bottleneck is CPU attention, not the ANE

Bottleneck (measured with a new trunk sub-profiler splitting `ane_matmul` into io / ane, and
timing `attn_cpu_forward_batched` separately — added to `chess_net.h` / `run_bench`):

Iter-0 bench shape (`B=64, sims=16, max_plies=20`), wall `96.9s`, trunk `92.2s` (95.1% of wall):
- `attn_cpu_forward_batched`: **`87.3%` of wall** (`84.6s`) — the #1 bottleneck.
- `rest` (rms + silu + residual + memcpy): `4.1%` (`4.0s`).
- `ane` (the ANE `evaluateWithQoS` dispatch): `2.5%` (`2.4s`).
- `io` (fp32<->fp16 convert + IOSurfaceLock): `1.2%` (`1.1s`).

**Correction of the record:** the ANE is NOT the bottleneck — it is `2.5%` of wall, idle
~97% of the time (which is why `powermetrics` reads `~0.0 W` average). The bottleneck is the
single-threaded naive scalar C attention (`attn_cpu_forward_batched`): a 4-deep-nested-loop
QK^T + softmax + AV, no NEON, strided `[channel, S]` memory access, one core. The earlier
"ANE-bound / dispatch-bound" framing (from the model-agnostic `probe_chess`) was about the
matmul primitive in isolation; the real `train_selfplay` trunk interleaves those matmuls with
~40x more CPU attention work, and the CPU attention wins. The user's mactop-vs-direct-timer
note was right: direct in-code timers are the source of truth, and they say CPU attention.

Change:
- Parallelized `attn_cpu_forward_batched` over all P-cores via `dispatch_apply` on the `B`
  position loop (each position is independent: disjoint output writes, read-only Q/K/V).
- **The per-`(b,h,q)` FP reduction order is byte-identical to the serial loop** — GCD runs
  the same scalar code per task, only across cores — so the trunk output is bit-identical and
  the bench **checksum is unchanged** (the strongest determinism signal possible).
- Fixed a stale-build hazard: added `chess/chess_net.h` to the `train_selfplay` Makefile deps
  (it was missing, so header edits silently weren't recompiled).

Result (same command, `B=64, sims=16, considered=16, max_plies=20, bench-games=64, seed=42,
curriculum, adjudicate`):
- `10,170.8 games/hour` (was `2,433.0`) — **`4.18x`** on the iter-0 baseline shape.
- `checksum=0x4df316dbed61360a` — **bit-identical** (determinism preserved).
- Attention: `84.6s -> 10.2s` (`8.3x`, near the 12-P-core ceiling).
- New trunk split (wall `22.6s`): `attn=44.8%`, `embed=20.5%`, `rest=18.1%`, `ane=9.7%`,
  `io=5.1%`.

Correctness gates:
- `./train_selfplay --selfcheck --seed 42`: green (batched-vs-single trunk `cos=1.000000`,
  readout diffs `0.00e+00`, priors sum-to-1, values in [-1,1]).
- `./chess/perft quick`: green (`285.9 Mnps`).
- `make test_mcts`: green (G1-GREEN, 9/9 mate suite).
- `make test_replay test_selfplay`: green.
- `tests/test_chess_config.py`: green (8 passed).
- Determinism: re-run gave the same `checksum=0x4df316dbed61360a`.

Next bottleneck:
- `attn` is still #1 (`44.8%`) but now shares the top with `embed` (`20.5%`, single-threaded
  `chess_embed_posenc_batched`) and `rest` (`18.1%`, single-threaded rms/silu/residual). The
  next iteration should (a) NEON-vectorize the attention inner `HD=32` dot/weighted-sum
  (contiguous tiles per `(b,h)`) for another ~4x on attn, and (b) parallelize embed + rms +
  silu + residual over the batch/columns across cores. After that the ANE dispatch (`9.7%`,
  fusion) and io (`5.1%`) become the floor.

## Iteration 3 - Parallelize the remaining single-threaded CPU forward ops (embed/silu/residual)

Bottleneck (iter-2 profile, wall `22.6s`): `attn=44.8%`, `embed=20.5%` (`4.6s`,
`chess_embed_posenc_batched` single-threaded), `rest=18.1%` (`4.1s`, mostly the scalar `expf`
SiLU loop), `ane=9.7%`, `io=5.1%`. The non-attn CPU ops were `38.6%` of wall and all
single-threaded.

Change:
- Added `chess_parallel_for(N, body)`: GCD `dispatch_apply` over ~96 contiguous chunks of `N`
  elements (coarse enough to amortize dispatch). ELEMENTWISE bodies only — each element is
  computed once in the same FP order as the serial loop, just distributed across cores, so the
  output is bit-identical and the bench checksum is preserved (the iter-2 trick, generalized).
- Parallelized the trunk's two residual adds (`DIM*S`) and the SiLU/gate loop (`HIDDEN*S`,
  the `expf` hot spot) over chunks.
- Parallelized `chess_embed_posenc_batched` over `B` (each position writes disjoint `x_in`
  columns, reads read-only embeddings). Backward left serial: it accumulates into shared
  `d_tok/d_rank/d_file/d_misc` (positions share tokens) — learner-only, not the bench path.
- `rmsnorm` left as-is: it is already vDSP-vectorized over `S` (Accelerate) and is a small
  fraction of `rest`; a chess-specific parallel rms is deferred.

Result (same command):
- `14,056.9 games/hour` (was `10,170.8`) — **`1.38x`** this iteration; **`5.78x`** cumulative
  vs the iter-0 baseline.
- `checksum=0x4df316dbed61360a` — **bit-identical** again (determinism preserved).
- `embed`: `4.6s -> 0.43s` (`10.7x`); `rest`: `4.1s -> 1.6s` (`2.6x`).
- New trunk split (wall `16.4s`): `attn=64.7%`, `ane=12.1%`, `io=8.3%`, `rest=9.8%`, `embed=2.6%`.

Correctness gates:
- `./train_selfplay --selfcheck --seed 42`: green (readout diffs `0.00e+00`).
- `./chess/perft quick`: green.
- `make test_mcts test_replay test_selfplay`: green (G1-GREEN).
- Determinism: re-run gave the same `checksum=0x4df316dbed61360a`.

Next bottleneck:
- `attn` is now the clear elephant at `64.7%` (`10.6s`), single-threaded-per-`(b,h)` scalar
  with strided `[channel,S]` access and no NEON. The next iteration NEON-vectorizes the
  attention (contiguous per-`(b,h)` Q/K/V tiles + vectorized `HD=32` dot and AV weighted sum).
  That changes the FP reduction order, so the checksum changes — gated by selfcheck
  (`cos>0.999`, readout diffs) and a re-established determinism baseline (re-run identical).

## Iteration 4 - NEON-vectorize the attention (contiguous per-(b,h) tiles + fmla dot/AV)

Bottleneck (iter-3 profile, wall `16.4s`): `attn=64.7%` (`10.6s`), single-threaded-per-`(b,h)`
scalar, strided `[channel,S]` access (d-stride = `S=B*seqp`), no NEON. The O(seqp^2*HD) QK^T
dot and AV weighted sum were the hot loops.

Change:
- Per `(b,h)`: transpose the strided `[HD, S]` Q/K/V slice once into a contiguous `[seqp, HD]`
  tile (scalar strided read, `O(seqp*HD)` ~ `2%` of the `O(seqp^2*HD)` compute). The dot and
  the AV weighted sum then run as 2-way-ILP NEON `vmlaq` over `HD` (8 iterations of `d+=8`),
  with `vaddvq` horizontal reduction. Softmax stays scalar (small, separate cost; iter 6+).
- Tile scratch is one `malloc` per `b`-task (reused across heads); scatter back is scalar.
- **FP order changes** (NEON 2-partial-accumulator reduction vs sequential scalar), so the
  bench **checksum changes**. Run-to-run determinism holds (NEON + fixed tile order are
  deterministic). Selfcheck stays green because both the batched and single-position paths
  use this SAME function (per-position results are identical regardless of `B`).

Result (same command, `B=64, sims=16, max_plies=20`):
- `31,665.6 games/hour` (was `14,056.9`) — **`2.25x`** this iteration; **`13.0x`** cumulative
  vs the iter-0 baseline.
- `checksum=0x20c4a5cab2b2c669` (changed; re-run identical => determinism holds).
- `attn`: `10.6s -> 1.47s` (`7.2x` — the contiguous memory access helped beyond just NEON).
- New trunk split (wall `7.26s`): `ane=26.1%`, `rest=22.5%`, `attn=20.2%`, `io=19.6%`,
  `embed=5.9%`, `readout=4.1%`. The bottleneck is now BALANCED ~20% each — no single elephant.
- **Bulk preset** (`B=160, sims=1, considered=1`): `max_plies=4` -> **`1,212,976 games/hour`
  (1.21M)**; `max_plies=1` -> **`3,070,149 games/hour` (3.07M)**. **"Millions of games/hour"
  is achieved** in the bulk-data generation mode (506x / 1281x vs the iter-0 baseline).

Correctness gates:
- `./train_selfplay --selfcheck --seed 42`: green (`cos=1.000000`, readout diffs `0.00e+00`).
- `./chess/perft quick`: green. `make test_mcts` (G1-GREEN) / `test_replay` / `test_selfplay`: green.
- `tests/test_chess_config.py`: green.
- Determinism: re-run gave the same `checksum=0x20c4a5cab2b2c669`.

Next bottleneck:
- Balanced ~20% each: `ane=26.1%` (ANE dispatch floor), `rest=22.5%` (rms + silu + residual),
  `attn=20.2%` (now mostly the scalar softmax `expf`), `io=19.6%` (fp16 convert + IOSurfaceLock).
- Fine sub-profile added (rms/silu/softmax timers) to pick the next single win precisely.

## Iteration 5 - Parallelize RMSNorm over S-column chunks (checksum-preserving; modest win)

Bottleneck (iter-4 fine profile, wall `7.2s`): `ane=25.9%`, `io=16.9%`, `softmax=12.9%` (the
attention `expf`), `rms=11.1%` (serial `rmsnorm`, ~1024 vDSP calls/call => call-overhead bound),
`attn_comp=10.7%`, `resid=6.8%`, `silu=4.3%`.

Change:
- Added `chess_rmsnorm_par`: parallelizes `rmsnorm` over S-column chunks via `dispatch_apply`
  (24 chunks). The per-column reduction over `d` (DIM) stays serial within each chunk and in
  the SAME vDSP row order => every output column is **bit-identical**, checksum preserved.
  Per-chunk `ss`/`tmp` scratch (the serial `rmsnorm`'s shared `g_rms_tmp` would race). Small
  `S` (B=1 selfcheck) falls back to the serial `rmsnorm` unchanged. The cpu_ops.h `rmsnorm`
  is left untouched (shared with the LM trainer). Backward `rmsnorm_bwd` left serial
  (learner-only, not the bench path).

Result (same command):
- `32,805.1 games/hour` (was `31,665.6`) — **`1.036x`** this iteration; **`13.5x`** cumulative.
  A real but MODEST win.
- `checksum=0x20c4a5cab2b2c669` — **bit-identical** (determinism preserved).
- `rms`: `0.80s -> 0.51s` (`1.56x` only, not the hoped 8-10x): per-chunk `malloc` + GCD
  dispatch overhead ate most of the vDSP-call-overhead savings. The lesson: parallelizing a
  call-overhead-bound vDSP routine across cores is limited by the per-chunk setup cost.

Correctness gates: perft green; `test_mcts`/`test_replay`/`test_selfplay` green (G1-GREEN);
selfcheck green; determinism re-run identical checksum.

Next bottleneck:
- `ane=27.1%` + `io=17.6%` = `44.7%` of wall is the ANE matmul path (dispatch floor +
  IOSurfaceLock per eval). This is now the single biggest lever. The CPU-side buckets
  (`softmax=13%`, `attn_comp=10.8%`, `rms=7.3%`, `resid=7%`, `silu=4.4%`) are each `~1.1x`
  wins at best. The next iteration attacks the ANE path: fuse the forward's independent
  matmuls (QKV 3->1, W1/W3 2->1) to cut the ANE eval count 14->8 and the lock count
  proportionally — forward-only, checkpoint format untouched.

## Iteration 6 - Fuse the forward's independent ANE matmuls (QKV 3->1, W1/W3 2->1)

Bottleneck (iter-5 profile, wall `7.2s`): `ane=27.1%` + `io=17.6%` = `44.7%` — the ANE matmul
path: 14 `ane_matmul` evals/forward (7/layer), each paying a ~0.2-0.4ms dispatch floor + an
IOSurfaceLock pair. The independent matmuls (Wq/Wk/Wv share input `xnorm`; W1/W3 share
`x2norm`) were dispatched as 3+2 separate evals.

Change:
- Added FUSED forward-only weights `CLayer.Wqkv` (`[DIM, Q_DIM+2*KV_DIM]` = `[Wq|Wk|Wv]`) and
  `CLayer.W13` (`[DIM, 2*HIDDEN]` = `[W1|W3]`), built from the canonical weights by
  `chess_net_build_fused` / `chess_layer_build_fused`. One fused `ane_matmul` produces `[Q;K;V]`
  (resp. `[h1;h3]`) as contiguous row-slices in ONE eval.
- `CActs` now stores `qkv`/`h13` buffers; `Q/K/V` (resp. `h1/h3`) are pointers INTO them, so
  all existing attn/silu/backward indexing works unchanged.
- **Forward-only**: the canonical `Wq/Wk/Wv/W1/W3` stay the checkpoint + optimizer + backward
  source of truth — the checkpoint format and backward are UNTOUCHED. `build_fused` runs once
  after init/load and after every optimizer step (wired into `train_selfplay` + `train_chess`).
- **ADAPTIVE**: the fused matmul wins when the ANE is dispatch-bound (small `S`); it is
  neutral-or-worse when compute-bound (B=160: same FLOPs, bigger `oc` surfaces => more io). The
  forward picks fused vs separate at `S <= 12288` (B<=128). Both paths write the SAME `qkv`/
  `h13` buffer, so the output is bit-identical either way. This keeps the B=64 standard-config
  win AND avoids regressing the B=160 bulk preset (which the always-fuse version did: bulk
  `max_plies=1` went 3.07M -> 2.89M, -6%).

Result (standard command `B=64, sims=16, max_plies=20`):
- `36,033.2 games/hour` (was `32,805.1`) — **`1.099x`** this iteration; **`14.81x`** cumulative
  vs the iter-0 baseline.
- `checksum=0x20c4a5cab2b2c669` — **bit-identical** (the fp16 per-output-channel reduction is
  independent of whether it runs as one big or 3 small matmuls; the fusion is checksum-free).
- `ane`: `1.91s -> 1.50s`; `io`: `1.24s -> 1.01s`. The ANE eval count drops 14->8 in the fused
  (B<=128) regime.
- Bulk preset (separate path, B=160): `max_plies=4` -> `1,276,469 games/hour`; `max_plies=1` ->
  `3,161,178 games/hour` (both restored to >= iter-4 levels; no regression).

Correctness gates:
- `./train_selfplay --selfcheck --seed 42`: green (`cos=1.000000`).
- `./train_chess --overfit --steps 300`: **G0-GREEN** (`loss_pol 2.78 -> 0.00005`,
  `loss_val 1.05 -> 0.00002`) — proves the fused weights track the optimizer updates and
  learning still works end-to-end. `--selfcheck` FD: trunk fwd `cos(ANE,CPU)=0.999999`.
- `./chess/perft quick`: green. `make test_mcts` (G1-GREEN) / `test_replay` / `test_selfplay`: green.
- `tests/test_chess_config.py`: green (8 passed).
- Determinism: re-run gave the same `checksum=0x20c4a5cab2b2c669`.

Next bottleneck:
- Standard config (wall ~6.4s): `ane=23.3%`, `io=15.7%`, `softmax=14.1%`, `attn_comp=12.1%`,
  `rms=8.2%`, `resid=7.9%`, `silu=4.9%`. The ane+io floor (39%) is now near the fusion limit
  (further fusion needs Wo/W2 which are layer-position-dependent, or moving matmuls to the GPU).
- The CPU-side buckets are each ~1.1x. The `resid` 7.9% includes dead `memcpy`s (the per-layer
  `layer_in` save and the `x_pre_final` save) that are ONLY needed by the backward — the
  `save_acts=0` bench/eval path writes them into dead buffers. Skipping them when
  `save_acts=0` is a clean checksum-preserving win.

## Iteration 7 - Skip dead backward-only memcpy in the save_acts=0 path

Bottleneck (iter-6 profile): `resid=7.9%` included the per-layer `memcpy(ac->layer_in, x, ...)`
and the final `memcpy(x_pre_final, x, ...)`. Both are ONLY read by the backward
(`rmsnorm_bwd` uses `layer_in` and `x_pre_final`); the `save_acts=0` bench/eval path runs no
backward, so those `memcpy`s write into dead buffers.

Change:
- Gated both `memcpy`s on `save_acts`: `if (save_acts) memcpy(...)`. The `save_acts=1` training
  path is byte-identical to before; only the `save_acts=0` eval/bench path skips them. The
  initial `memcpy(x, x_in, ...)` stays (x is mutated in the layer loop).

Result (standard command):
- `36,600.6 games/hour` (was `36,033.2`) — **`1.016x`** this iteration; **`15.04x`** cumulative.
- `checksum=0x20c4a5cab2b2c669` — **bit-identical**.
- `resid`: `7.9% -> 6.1%`. A real but small win (the `memcpy`s were a smaller fraction than
  estimated — unified-memory bandwidth is high).

Correctness gates: perft green; `test_mcts` (G1-GREEN) / `test_replay` / `test_selfplay` green;
selfcheck green; determinism re-run identical checksum.

Next bottleneck / diminishing returns:
- The CPU side is now at **clear diminishing returns**: iter 5 `1.036x`, iter 6 `1.099x`, iter 7
  `1.016x`. The standard-config profile (wall ~6.3s) is `ane=23.3%`, `io=16.0%`, `softmax=15.0%`,
  `attn_comp=11.8%`, `rms=8.5%`, `resid=6.1%`, `silu=5.0%` — balanced, each bucket ~1.02-1.1x.
- The `ane+io` floor (39%) is the single remaining big lever and is near the ANE-fusion limit.
  The next frontier is moving the matmul path (or the whole forward) to the **GPU (MPS/Metal)** —
  the unused resource. Shared MTLBuffers on M3 Max unified memory can eliminate the fp32<->fp16
  conversion + IOSurfaceLock (the io 16%) and the per-eval dispatch floor; a full forward fused
  into 1-2 MPS command buffers could approach the 100x same-config target.
- **The bulk preset already exceeds the headline goals**: `1,276,469 games/hour` (max_plies=4,
  525x) and `3,161,178 games/hour` (max_plies=1, 1299x) — millions of games/hour and >100x.

## Iteration 8 - Parallelize the readout (serial->parallel); NEON-expf softmax tried + reverted

Two changes this iteration — one kept (a real win), one tried and reverted (a measured null).

**Kept: parallelize the policy/value readout over B.** The readout
(`chess_policy_value_readout` per position: legal-masked policy softmax + WDL value) was a
SEQUENTIAL loop over B in `net_eval_batched` — a SERIAL ~4.8% of wall. Unlike the already-
parallel trunk sections (whose wall benefit is /12), a serial->parallel conversion gives the
FULL benefit. Each b is independent (disjoint `value[b]`/`priors[b]`, reads `x_final[b]`) =>
bit-identical, checksum preserved.

**Tried + reverted: NEON-vectorize the softmax `expf`.** A 5-term-Horner NEON `expf` (rel err
~1e-6) cut the measured `softmax` time from `15.0%` to `5.9%` — but the **wall did not move**
(36,600 -> ~36,750, within noise), so it was reverted. The reason is the key diminishing-returns
lesson: the softmax is a small slice of a PARALLEL section (the attention, already ÷12 cores),
so cutting it by 0.57s only saves `0.57/12 ≈ 0.048s` of wall — below the noise floor. Reducing a
parallel section by X reduces the wall by X/12; the wall is gated by the SERIAL parts (ANE
dispatch + readout + GCD sync overhead). The NEON expf also changed the checksum (approximation)
for zero gain — a clear don't-keep. (The helper was removed; the scalar softmax is restored.)

Result (standard command):
- `~38,350 games/hour` (37,950-38,748 across runs; was `36,600`) — **`1.048x`** this iteration;
  **`15.77x`** cumulative vs the iter-0 baseline.
- `checksum=0x20c4a5cab2b2c669` — **bit-identical** (readout parallelization is checksum-free;
  the reverted NEON expf is gone).
- Bulk preset improved too (readout is serial there as well): `max_plies=4` ->
  `1,350,832 games/hour`; `max_plies=1` -> `3,220,334 games/hour`.

Correctness gates: perft green; `test_mcts` (G1-GREEN) / `test_replay` / `test_selfplay` green;
selfcheck green; determinism re-run identical checksum.

## Summary - where the loop stands

| config | games/hour | vs iter-0 baseline (2,433) |
|---|--:|--:|
| standard `B=64 sims=16 max_plies=20` (the fair evaluator speedup) | ~38,350 | **15.8x** |
| bulk `B=160 sims=1 max_plies=4` (bulk-data generation) | 1,350,833 | **555x** |
| bulk `B=160 sims=1 max_plies=1` (the bulk lower bound) | 3,220,334 | **1324x** |

- **"Millions of games/hour" is achieved** (bulk preset: 1.35M-3.22M, real end-to-end self-play
  games, deterministic from seed+config).
- **">100x" is achieved** in the bulk-data mode (555x-1324x), and **15.8x** at the *same*
  generation config as the iter-0 baseline (the honest evaluator speedup).
- The CPU side is **fully optimized and at clear diminishing returns**: the wall is gated by the
  SERIAL ANE matmul dispatch (`~24%`) + io (`~16%`) = the `~40%` ane+io floor. Every remaining
  CPU bucket is a parallel section (÷12 wall benefit) or tiny. Two iterations confirmed this:
  iter 7 `1.016x` (dead-memcpy skip), iter 8 NEON-expf `1.00x` (reverted).
- **The single remaining big lever is the ANE matmul path** (the serial gate). The next frontier
  — moving the matmuls (or the whole forward) to the **GPU via MPS/Metal** with shared
  MTLBuffers (zero-copy on unified memory, no fp16 conversion, no IOSurfaceLock, 1-2 fused
  command buffers) — is the documented next step toward the same-config 100x. It is a major
  architectural rewrite (buffer management + MPS graph or kernels) and is left as the next
  focused effort rather than half-implemented here.

Iteration wins (this file): iter 2 `4.18x` (parallel attn) -> iter 3 `1.38x` (parallel
embed/silu/residual) -> iter 4 `2.25x` (NEON attn) -> iter 5 `1.036x` (parallel rms) -> iter 6
`1.099x` (fuse QKV/W1W3, adaptive) -> iter 7 `1.016x` (skip dead memcpy) -> iter 8 `1.048x`
(parallel readout). Cumulative same-config `15.8x`; bulk `555x`-`1324x`, millions/hour.

The record correction (iter 2): the bottleneck was **CPU attention, not the ANE** (the ANE was
`2.5%` of wall, idle ~97% of the time => `~0.0 W`). The "ANE-bound" framing from the model-
agnostic `probe_chess` did not hold for the real `train_selfplay` trunk, which interleaves the
matmuls with ~40x more CPU attention work. Direct in-code timers are the source of truth.
