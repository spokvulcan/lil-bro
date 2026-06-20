# lil-bro — Context

Glossary for the DeepSeek-V4 → small-dense-ANE architecture port. Terms here are
the contested/load-bearing ones we've pinned down; implementation decisions live in
[docs/adr/](docs/adr/), phasing in [ROADMAP.md](ROADMAP.md).

## Language

**DeepSeek-V4 parity**:
Implementing V4's architectural *mechanisms* scaled down to a ~20M dense model — a
mechanism counts as "at parity" when it moves our tokens-to-target frontier the way
V4 claims it moves theirs. It is **not** benchmark parity (impossible at this scale)
and **not** the MoE / infra / FP4-FP8 machinery (irrelevant to a dense fp16 model).
_Avoid_: parity (unqualified), V4-equivalent, V4-compatible

**mHC** (manifold-constrained hyper-connections):
The residual upgrade ported from V4: the residual stream is widened to `n_hc` parallel
streams, and each sub-layer's residual add is replaced by three learned maps with the
residual map constrained doubly-stochastic.
_Avoid_: hyper-connections / HC (that's the unconstrained predecessor)

**residual stream**:
In the mHC sense, the `n_hc`-wide bundle of parallel residual vectors (`ℝ^{n_hc×d}`),
not the single `ℝ^d` vector of a plain transformer.

**input / residual / output map** (A / B / C):
mHC's three per-layer matrices — `A` (1×n_hc, collapses streams to the layer input),
`B` (n_hc×n_hc, doubly-stochastic, mixes streams), `C` (n_hc×1, broadcasts the layer
output back to the streams).

**Tier A / B / C / D**:
The keep/drop/adapt buckets for V4 components ([ADR 0001](docs/adr/0001-deepseek-v4-for-dense-ane.md)):
A = cheap stabilizers (adapt now), B = flagship (mHC + ANE MTP), C = CSA/HCA (defer to a
long-context rung), D = MoE / precision / post-training (drop).

**faithful sample** (vs. *plausible* sample):
A generated-text sample produced by the *exact* forward the model trains with — for the
flagship, the FD-verified C `forward_hidden` including the mHC wrapper — so fidelity holds
*by construction*. Contrast a numpy re-implementation, which can read back plausible text
while being subtly wrong; that is a *plausible* sample, not a faithful one ([ADR 0002](docs/adr/0002-observing-mhc-flagship-train.md)).
_Avoid_: "generated text" (unqualified) when the distinction matters.

**in-trainer emission**:
Running the sampler inside the training loop (a sibling of `eval_val_loss`) and printing
`[gen step=N] <token-ids>` to stdout, which the dashboard decodes and displays — rather than
a separate process or a dashboard-side forward. Chosen to avoid a second ANE client
(contention) and a second mHC implementation (drift).

**build-tied checkpoint**:
The on-disk format is defined by the *compile-time* knobs, not self-described in the header:
the loader reads the `#if ATTN_SINK` / `#if QK_NORM` / (v5) `#if N_HC>1` blocks only when
those knobs are set. A reader (e.g. the dashboard) must therefore be told the build config.
`CkptHdr.version` 5 carries the mHC block; a mismatched build refuses to load.
_Avoid_: calling it "self-describing".
