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
