# PRD: lil-bro — Phase 1

**An ANE-trained small dense LM as a DeepSeek-V4 architecture-ablation testbed.**

A research fork of [maderix/ANE](https://github.com/maderix/ANE) (MIT). See [ROADMAP.md](../ROADMAP.md) for phasing and [NOTICE](../NOTICE) for attribution. *(No time estimates by design — phases are ordered by dependency.)*

> **Superseded in part (2026-06-20): ANE-only.** The **MLX twin / fp64 oracle / GPU baseline** described in this Phase-1 PRD were the instrument actually built (R0 & R1 green; R1 caught a real GQA backward bug — `results/r1_grad_diff.md`) and remain in the repo. Going forward the project is **ANE-only**: correctness is verified **behaviorally** (R0 overfit on the ANE + held-out validation), with no gradient oracle and no GPU baseline. The Phase-1 narrative below is preserved as history; new work follows [ADR 0001](adr/0001-deepseek-v4-for-dense-ane.md) and PRD [spokvulcan/lil-bro#2](https://github.com/spokvulcan/lil-bro/issues/2).

## Problem Statement

As an ML-systems researcher with an Apple-Silicon MacBook, I can see that training on the Apple Neural Engine is *possible* — the upstream proof-of-concept shows it — but I can't answer the questions I actually care about, and I can't run controlled experiments at all:

- **Is ANE training correct?** Upstream measures only *training* loss; there is no held-out validation and no comparison against a trusted reference. A subtly buggy backward pass still produces a plausible-looking loss curve, so I cannot distinguish "genuinely learning" from "silently wrong gradients."
- **Is ANE training worth it?** There is no apples-to-apples energy / throughput comparison against the GPU.
- **Do modern architecture ideas help a *small dense* model?** DeepSeek-V4 introduces Muon, MTP, and more — but it's a trillion-parameter MoE built for million-token context. Which of its ideas, if any, improve a small dense model, and does the ANE change that verdict?
- **I can't even sweep.** The model config is hard-coded at compile time, and there is no validation/eval harness.

## Solution

lil-bro turns the upstream ANE trainer into a controlled **research instrument**: a small **dense** TinyStories transformer trained **from scratch on the ANE** across a ladder of sizes, with a held-out validation set, an **MLX twin** that serves as both a **correctness oracle** and a **GPU energy baseline**, and an **ablation runner** that reports — per architecture idea — whether it improves the *efficiency frontier*.

The headline number is **hardware-independent** (tokens-to-target validation loss); the systems verdict (energy / wall-clock, ANE vs MLX) rides alongside. The workflow: pick a config → train on the ANE → confirm gradients match the MLX oracle → measure tokens-to-target → compare baseline vs Muon vs MTP, each as a clean one-variable experiment.

## User Stories

1. As a researcher, I want model dimensions (dim, layers, heads, seq, vocab) to come from a runtime config instead of compile-time constants, so that I can sweep the scaling ladder without editing source and recompiling per cell.
2. As a researcher, I want one config to drive both the ANE trainer and the MLX twin, so that the correctness diff and the energy comparison are guaranteed to compare identical models.
3. As a researcher, I want a held-out validation set from a separate TinyStories shard (`data01`) distinct from the training shard (`data00`), so that val loss has no train/val window leakage.
4. As a researcher, I want periodic validation-loss evaluation during training, so that I have the signal the headline tokens-to-target metric is built on.
5. As a researcher, I want a generation sampler that emits TinyStories text from a checkpoint, so that I can qualitatively sanity-check coherence alongside the numeric loss.
6. As a researcher, I want to overfit a tiny model to one repeated batch and watch loss collapse to ~0, so that I can confirm the full ANE train loop (forward, backward, optimizer, update) works before trusting anything else.
7. As a researcher, I want to run one forward+backward step on the ANE and on the MLX twin from identical init weights and the same batch and compare all parameter gradients within a tolerance, so that I can prove ANE training is correct rather than plausibly-wrong.
8. As a researcher, I want the gradient diff to cover the base model, Muon, and MTP, so that a single correctness gate validates every component I ablate.
9. As a researcher, I want to select the optimizer (AdamW or Muon) via config, so that I can ablate Muon against the AdamW control as a one-variable change.
10. As a researcher, I want Multi-Token Prediction (MTP) as an opt-in flag, so that I can measure its effect independently.
11. As a researcher, I want each architecture component added one at a time and kept only if it improves the efficiency frontier, so that lil-bro stays small and fast by construction.
12. As a researcher, I want the headline metric to be tokens-to-target validation loss, so that conclusions about architecture are hardware-independent and comparable across ANE and MLX.
13. As a researcher, I want energy-to-target and wall-clock-to-target measured as secondary metrics, so that I can render the systems verdict on whether the ANE is worth it.
14. As a researcher, I want to measure ANE utilization myself, so that I don't rely on upstream's conflicting figures.
15. As a researcher, I want the efficiency target defined at the knee of the baseline (dense+AdamW) val curve at rung 2, so that every ablation measures cost-to-reach-the-same-loss (iso-loss).
16. As a researcher, I want a learning-rate sweep (≥3 points) per config, so that I never compare a tuned optimizer against an untuned one.
17. As a researcher, I want all controls (data, sequence length, batch tokens, model dims, val set) held fixed across an ablation, so that any measured delta is attributable to the single varied component.
18. As a researcher, I want to climb the scaling ladder only when the current rung's gate is green, so that I never scale a broken configuration.
19. As a researcher, I want the headline ablation matrix at rung 2 (small) and only a single confirmation + energy run at rung 3 (110M), so that I don't pay for a full sweep at large scale.
20. As a researcher, I want byte-level vocab at the correctness rungs and the fixed 32K vocab at the efficiency rungs, so that the cheap gates stay genuinely tiny while measured results stay comparable.
21. As a researcher, I want a small-vocab variant available as its own later ablation, so that I can ask whether a smaller tokenizer improves the params/efficiency frontier without confounding the main experiments.
22. As a researcher, I want results as a table (Δ tokens/steps/energy/wall-clock per component vs control) plus loss-curve plots, so that the verdict is legible at a glance.
23. As a researcher, I want a config's ablation number trusted only after that config passes the gradient-diff gate, so that no "win" comes from a buggy backward pass.
24. As a researcher, I want results reproducible from a fixed seed + config, so that any reported number can be re-derived.
25. As a researcher, I want to pull upstream improvements via the upstream remote, so that lil-bro can absorb maderix fixes without losing my additions.
26. As a maintainer, I want lil-bro's delta over upstream stated in the README plus a NOTICE preserving MIT attribution, so that the derivative relationship is honest and license-compliant.
27. As a researcher, I want train/val data prepared as flat, memory-mappable token streams from the TinyStories shards, so that the data loader stays simple.
28. As a researcher, I want validation evaluated on a fixed set of batches, so that val loss is comparable step-to-step and run-to-run.
29. As a researcher, I want the existing power/utilization dashboard reused where possible, so that I build new infra only where upstream lacks it.
30. As a researcher, I want a roadmap that explicitly parks the tool-first vision (Phase 2) and heavier DeepSeek-V4 components — mHC, CSA/HCA (Phase 3) — so that the project stays focused.

## Implementation Decisions

**Modules built / modified** (described, not file-pathed):

- **Parametric config layer** — replaces upstream's compile-time `#define` model dimensions with a runtime config consumed *identically* by the ANE trainer and the MLX twin. Fields: `dim, n_layers, n_heads, seq, vocab, optimizer (adamw|muon), mtp_depth, lr, batch_tokens, seed`. **This shared-config contract is the single most important interface in the project** — one schema, two consumers, guaranteeing identical models for both the correctness diff and the energy comparison.
- **Validation/eval harness** — train shard `data00` / val shard `data01`; periodic val-loss on a *fixed* batch set; a generation sampler.
- **Muon optimizer** — a CPU-side optimizer update (Newton-Schulz orthogonalization) applied to 2D weight matrices; embeddings, norms, biases, and the LM head remain on AdamW. Implemented identically in the ANE trainer and the MLX twin.
- **Multi-Token Prediction (MTP)** — opt-in; reuses existing transformer-block kernel *types* for the MTP module(s); adds a combined loss. Implemented identically across trainers.
- **MLX twin** — a from-config reimplementation of the dense + Muon + MTP model, used as (a) correctness oracle (its autograd gradients are ground truth) and (b) GPU energy baseline.
- **Ablation runner** — orchestrates the iso-loss protocol (per-config LR sweep, one-variable changes, fixed controls) and emits the results table + plots.

**Architectural decisions:**

- **Dense only; no MoE.**
- **Sequenced ablation backlog:** baseline (dense+AdamW) → Muon → MTP → *(Phase 3)* mHC → CSA/HCA; each kept only if it improves the efficiency frontier.
- **Headline metric is hardware-independent** (tokens-to-target); energy/wall-clock are systems-side secondaries.
- **iso-loss** protocol; target = baseline knee val loss at rung 2.
- **Scaling ladder with gates:** R0 overfit (byte vocab, 1 layer, d=64, seq=64) → R1 ANE↔MLX gradient diff → R2 small (d=256, ~6 layers, 32K vocab, seq=256) = headline ablation matrix → R3 (d=768, 12 layers, 32K, seq=256) = single confirmation + energy verdict.
- **Vocab:** byte-256 at R0–R1, fixed 32K at R2–R3; small-vocab is a separate later ablation.
- **MLX is the oracle (fp32; PyTorch fp64 reserved as a tie-breaker for ambiguous diffs) and the GPU baseline — never the trainer for the thesis.**

## Testing Decisions

**What makes a good test:** assert on **external behavior** — loss values and parameter gradients at the module boundary — never on internal MIL kernel structure or intermediate ANE buffers. Use the highest seam possible.

**Modules tested (the two seams):**

- **Seam 1 — full-step gradient diff (primary).** From a shared config + fixed seed + identical init + one fixed batch, run a single forward+backward; assert all parameter gradients agree with the oracle within tolerance. Two consumers of one seam: the **MLX twin** vs torch fp64, always-on (`tests/test_grad_diff.py`, covers base + GQA + MTP); and the **ANE** vs torch fp64 on real hardware via `lilbro/ane_bridge` (`r1_gate.py`, dense base + GQA — the ANE has no MTP path). The diff is on the raw gradient *before* the optimizer step, so it is optimizer-agnostic (Muon vs AdamW change the update, not the gradient; Muon's end-to-end correctness is R0's job). This *is* the R1 correctness gate. *(First ANE run caught a real forward/backward GQA-convention mismatch — see `results/r1_grad_diff.md`.)*
- **Seam 2 — overfit-one-batch (behavioral, end-to-end).** Train a tiny config on one repeated batch on the ANE; assert loss collapses toward ~0. Black-box, no oracle; exercises the full loop including the Muon update. This *is* the R0 gate.

**Prior art:** the upstream per-kernel unit tests (RMSNorm-backward, fused-backward, QKV, etc.) serve as **failure localizers** when Seam 1 fails — they are *not* the primary seam. The reference-dump comparison pattern (historically used by an upstream inference verify script) informs Seam 1's design; lil-bro builds the gradient-diff harness fresh against MLX.

**Tolerances:** consumer-dependent. The MLX↔torch diff is ≥fp32 on both sides, so it uses **fp32-scale element-wise relative error** (`1e-4`) — a genuine backward bug diverges far beyond fp32 rounding. The **ANE matmuls are fp16**, so its diff against the fp64 oracle uses an **fp16-scale gate instead**: per-parameter cosine (direction) ≥ 0.99 and relative-L2 (magnitude) ≤ 0.10, thresholds set from the measured clean floor (~0.998 / 0.067) and validated to fail the GQA bug (cosine 0.44 / rel_l2 >1.0) by a wide margin. Element-wise fp32 tolerance does **not** apply to fp16 hardware. PyTorch fp64 is the oracle for both.

## Out of Scope

- The **tool-first / retrieval-first / "knowledge-minimal weights" / non-hallucinating** model — **Phase 2**; it requires a competent backbone, not a from-scratch tiny model.
- Training a **useful general LLM from scratch on a Mac** — infeasible; the ceiling is TinyStories-class.
- **DeepSeekMoE (MoE)** — dense only.
- **CSA / HCA** (DeepSeek-V4 long-context / hybrid attention) — no benefit at seq=256, and require substantial new ANE kernels; **Phase 3**.
- **mHC** (Manifold-Constrained Hyper-Connections) — new kernels + differentiable Sinkhorn, unproven at small depth; **Phase 3**.
- **MLX (or PyTorch) as the trainer** for the thesis — oracle/baseline only.
- A **full ablation sweep at rung 3** (110M) — rung 3 is a single confirmation + energy run.
- A **production training framework** — lil-bro is research code.

## Further Notes

- Derivative of maderix/ANE (MIT); attribution preserved in LICENSE + NOTICE; delta stated in README; upstream wired as a git remote for pulls.
- **Evidence discipline:** upstream reports ANE utilization as both "~2–3% of peak" and "15.5%" — lil-bro measures utilization itself and trusts neither.
- The two theses share one protocol: the **hardware-independent** tokens-to-target axis answers the *architecture-ablation* question; the **hardware-dependent** energy/wall-clock axis answers the *systems "is the ANE worth it"* question.
