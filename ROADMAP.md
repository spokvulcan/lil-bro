# lil-bro Roadmap

Phases are ordered by **dependency, not schedule** — no time estimates by design.

## Phase 1 — Systems + architecture ablation (current)

Train a small **dense** TinyStories transformer **from scratch on the ANE**, and
measure whether DeepSeek-V4 ideas improve the *small-dense-model efficiency
frontier*. MLX is **not** the trainer — it is the correctness oracle + GPU baseline.

**1a — ANE**
- Parameterize model config (replace compile-time `#define`s in `training/stories_config.h`)
- Validation/eval harness: `data00` train, `data01` val; periodic val loss; generation sampler
- Rung 0 gate: overfit one batch (control = dense + AdamW) → loss → ~0
- Add **Muon** (CPU optimizer swap) and **MTP** behind flags

**1b — MLX**
- MLX twin of dense + Muon + MTP from the shared config
- Rung 1 gate: ANE gradients match MLX within tolerance (correctness, c1)

**1c — Measure**
- Rung 2 ablation matrix: baseline → Muon → MTP × LR sweep, on ANE + MLX (headline = tokens-to-target)
- Rung 3: single confirmation of the winning config + energy/utilization verdict (ANE vs MLX)
- Results table + plots

### Scaling ladder (climb only when the gate is green)
| Rung | Config | Gate |
|---|---|---|
| R0 overfit | 1 layer, d=64, byte-256 vocab, seq=64 | ANE loss → ~0 on one repeated batch |
| R1 grad-diff | same | ANE grads match MLX within tolerance |
| R2 small | d=256, ~6 layers, 32K vocab, seq=256 | val tracks MLX; coherent stories; **headline ablation here** |
| R3 110M | d=768, 12 layers, 32K, seq=256 | reproduces upstream; energy verdict |

### Method invariants
- **iso-loss** (compute-to-target); target = baseline knee val loss @ R2
- **per-config LR sweep (≥3)** — mandatory for a fair optimizer/architecture comparison
- vary **exactly one** component; hold data / seq / batch-tokens / dims / val-set fixed
- a config's ablation number is trusted **only after** it passes the R1 correctness gate
- headline metric is **hardware-independent** (tokens-to-target); energy/wall-clock are systems-side secondaries
- measure ANE utilization ourselves — upstream states both "~2–3% of peak" and "15.5%"; trust neither

## Phase 2 — Tool-first (deferred)

The original "knowledge-minimal, tool-first, non-hallucinating" vision. Requires a
**competent backbone** (fine-tune, not from-scratch-tiny) — out of scope for Phase 1.

## Phase 3 — Heavier DeepSeek-V4 components (future)

- **mHC** (Manifold-Constrained Hyper-Connections) — new ANE fwd/bwd kernels + differentiable Sinkhorn
- **CSA / HCA** hybrid attention — new sparse/compressed-attention kernels; needs long context (seq ≥ 2–8K) to mean anything

Each Phase-3 component is gated on whether it improves the efficiency frontier, and
both require substantial new ANE kernel work.
