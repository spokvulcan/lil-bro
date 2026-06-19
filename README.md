# lil-bro

> A research **fork of [maderix/ANE](https://github.com/maderix/ANE)** (MIT) that
> trains a small **dense** TinyStories transformer **from scratch on the Apple
> Neural Engine** — and uses it as a rigorous testbed for whether **DeepSeek-V4**
> architectural ideas improve the *small-dense-model efficiency frontier*.

## Why this exists

Apple ships the ANE as **inference-only**. The silicon can train; the software
was never allowed to. Upstream proved the barrier was always *software, not
hardware* — full backpropagation runs directly on the Neural Engine via
reverse-engineered private APIs. No CoreML training, no Metal, no GPU.

lil-bro takes that proof and asks the next question. Now that we *can* train on
the ANE: is it **correct**, is it **worth it**, and do modern architecture ideas
actually help a *small dense* model? It turns a weekend proof-of-concept into a
controlled **research instrument**.

The ethos, inherited from upstream and sharpened (see [CLAUDE.md](CLAUDE.md)):
**assume it's possible until the hardware says no, attack every wall from another
angle, and aim past the frontier instead of admiring it.** The workarounds below
are that ethos made concrete — none of them existed until someone refused to take
the first "no" for an answer.

## The thesis

> **Direction update (2026-06-20): ANE-only.** No MLX/torch twin, no fp64 gradient
> oracle, no GPU baseline. Correctness is verified **behaviorally** — R0 overfit on
> the ANE → loss ~0, plus held-out validation — not by a gradient diff. The MLX twin
> and R1 grad-diff gate were the Phase-1 instrument (built, green — R1 caught a real
> GQA bug) and **remain in the repo as history**, but are no longer the method. New
> work follows [ADR 0001](docs/adr/0001-deepseek-v4-for-dense-ane.md).

Systems **+** architecture-ablation. The headline number is **tokens-to-target
validation loss** — a behavioral metric, so conclusions about architecture hold
regardless of optimizer or rung. The systems verdict (energy / wall-clock) is
measured **directly on the ANE**. And nothing counts until a config clears the **R0
overfit gate on the ANE**: no "win" may come from a silently-wrong backward pass.

- **Headline:** tokens-to-target validation loss
- **Secondary:** energy- and wall-clock-to-target, measured directly on the ANE
- **Correctness gate:** R0 overfit on the ANE (loss → ~0) + held-out validation (behavioral)

## What lil-bro adds over upstream

- **Parametric model configs** (tiny → TinyStories scale) — upstream config is compile-time `#define`s
- A held-out **validation/eval harness** (`data00` train / `data01` val) — upstream measures train loss only
- **Muon** optimizer and **Multi-Token Prediction (MTP)** as opt-in ablations
- A **behavioral correctness gate** — overfit-one-batch on the ANE (loss → ~0) + held-out validation (the Phase-1 **MLX twin / gradient oracle** is retained as history but **superseded**; ANE-only)
- A **tokens-to-target** ablation study of DeepSeek-V4 ideas at small dense scale

## The scaling ladder

Climb only when the current rung's gate is green — never scale a broken config.

| Rung | Config | Gate |
|---|---|---|
| R0 overfit | 1 layer, d=64, byte-256 vocab, seq=64 | ANE loss → ~0 on one repeated batch (behavioral correctness gate) |
| ~~R1 grad-diff~~ *(Phase-1 history)* | same | ANE grads matched the MLX oracle — green, caught a real GQA bug; **superseded** by R0 + val, kept as history |
| R2 small | d=256, ~6 layers, 32K vocab, seq=256 | held-out val converges + coherent samples; **headline ablation here** |
| R3 110M | d=768, 12 layers, 32K, seq=256 | reproduces upstream; ANE energy verdict |

**Status:** Phase 1 (baseline → Muon → MTP). Full phasing, method invariants, and
gates in [ROADMAP.md](ROADMAP.md); the complete Phase-1 spec in [docs/PRD.md](docs/PRD.md).

---

## The foundation: backprop on the ANE (from upstream)

The ANE is a ~15.8 TFLOPS FP16 (M4) inference accelerator Apple doesn't expose for
training. Upstream reverse-engineers the `_ANEClient` / `_ANECompiler` private APIs
and the MIL (Model Intermediate Language) format to run custom compute graphs —
including backpropagation — directly on the hardware. lil-bro builds on this code;
the numbers below are **upstream's reported results**, and re-measuring them
honestly (utilization especially) is part of Phase 1's systems verdict.

| Model | Params | ms/step | Pipeline |
|-------|--------|---------|----------|
| Stories110M (12L, dim=768, MHA 12/12) | 109M | **91 ms** | Dynamic (no recompile) |
| Qwen3-0.6B (28L, dim=1024, GQA 16/8) | 596M | **412 ms** | Dynamic (no recompile) |

Forward + backward `dx` on ANE, `dW` gradients on CPU (Accelerate cblas); Adam,
gradient accumulation, checkpoint/resume; GQA support; GPU→ANE zero-copy prefill.
INT8 W8A8 quantization reaches **1.88×** ANE throughput (35.1 vs 18.6 TOPS, M4).
The deep dives: [Part 1 — Reverse Engineering](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine),
[Part 2 — Benchmarks](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615),
[Part 3 — Training](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-c8b).

### Walls, and the angle past each one

Every constraint below is a hardware "no" that got reframed instead of accepted —
the philosophy made concrete:

- **SDPA ignores `attn_mask`** → causal attention decomposed: Q@Kᵀ (ANE) → mask + softmax (CPU) → scores@V (ANE)
- **~119 compile limit** (the ANE compiler leaks resources) → `exec()` restart with checkpoint, bypassing the per-process cap
- **FP16 gradient underflow** in backward matmuls → global loss scaling (`256 × NLAYERS`)
- **Single-input constraint** (multi-input requests error `0x1d`) → activations + weights packed into one spatial dimension, sliced apart inside the kernel

### Building

Requires macOS 15+ on Apple Silicon (tested on M4). No external dependencies —
system frameworks + private ANE APIs resolved at runtime.

```bash
# Dynamic pipeline (recommended) — model selected at build time
cd training/training_dynamic
make MODEL=stories110m     # or MODEL=qwen3_06b (default)
./train --scratch          # train from random init
./train --resume           # resume from checkpoint
```

Training data (pretokenized TinyStories): `cd training && bash download_data.sh`
(`data00` = train, `data01` = val). Legacy/static pipeline, probes, and ANE op
tests are `make` targets in `training/Makefile`. See
[training/README.md](training/README.md) for the kernel-level detail and the full
file map.

## Attribution & license

lil-bro is a research fork of [maderix/ANE](https://github.com/maderix/ANE), used
under the MIT License — original upstream code and this fork's additions are both
MIT. Attribution and the upstream-vs-fork delta are recorded in [NOTICE](NOTICE);
the license text is in [LICENSE](LICENSE).

This project uses Apple's private, undocumented APIs (`_ANEClient`, `_ANECompiler`,
`_ANEInMemoryModelDescriptor`), which carry no stability guarantee and may break
with any macOS update. It is independent research into the Apple Neural Engine via
APIs discovered through runtime introspection, for research and educational
purposes under fair-use / interoperability provisions (cf. *Sega v. Accolade*,
1992; DMCA §1201(f)). No Apple proprietary code or binaries are included. Not
affiliated with or endorsed by Apple Inc. Use at your own risk.

---

*A fork that refuses to admire the frontier — built by a human + Claude.*
