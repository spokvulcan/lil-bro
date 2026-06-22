# results

Ablation outputs: tables, loss curves, and the energy verdict.

> **Sibling project — Chess RL self-play on the ANE** ([ADR 0005](../docs/adr/0005-chess-rl-self-play-on-ane.md)):
> [`chess_throughput_probe.md`](chess_throughput_probe.md) — build-step 0 gate (games/day,
> ANE-bound verdict, days-vs-never). Raw: [`chess_throughput_probe_raw.txt`](chess_throughput_probe_raw.txt).

## DeepSeek-V4 parity build status (PRD spokvulcan/lil-bro#2)

Per-mechanism **correctness** state (R0 overfit on the ANE is the gate that must be
green before any ablation number is trusted). Held-out-val ablation numbers (the
headline below) are the follow-up measurement once a mechanism is R0-green.

| Issue | Mechanism | R0 gate (ANE) | Evidence |
|---|---|---|---|
| #3 | Config ablation knobs (prefactor) | ✅ identity | 25 cfg tests; R0 bit-identical 0.0156 |
| #4 | Muon → V4 hybrid Newton-Schulz | ✅ green | R0→0.0000 `--opt muon`; [muon_v4.md](muon_v4.md) |
| #10 | Partial RoPE (last `rope_rotary_dims`) | ✅ green | identity@hd≤64; hd=128→0.0000; [partial_rope.md](partial_rope.md) |
| #9 | SwiGLU clamping | ✅ green | R0 green + FD backward 9/9; [swiglu_clamp.md](swiglu_clamp.md) |
| #5 | mHC Sinkhorn spike (fp16) | ✅ green | τ≥0.5 doubly-stoch.; bwd 7.9e-11; [mhc_sinkhorn_spike.md](mhc_sinkhorn_spike.md) |
| #7 | Q/KV RMSNorm | ✅ green | R0 on→0.0202 (off→0.0150); FD bwd 5.8e-4; [qk_norm.md](qk_norm.md) |
| #8 | Attention sink | ✅ green | R0 on→0.0105 (off→0.0150); FD bwd 2.4e-4; [attn_sink.md](attn_sink.md) |
| #6 | ANE MTP path | ✅ green | R0 combined→0.0105 (off→0.0150); FD e2e 8.2e-5; [mtp.md](mtp.md) |
| #11 | mHC (flagship) | ✅ green | N_HC∈{2,4} R0→0.003 (off→0.0150); B doubly-stoch 1e-7; FD 1.8e-9; [mhc.md](mhc.md) |

## Headline artifact — per component, vs the dense+AdamW control at best-tuned LR

| component | Δ tokens-to-target | Δ steps | Δ energy (ANE) | Δ wall-clock | verdict |
|-----------|--------------------|---------|----------------|--------------|---------|

Plus the systems verdict: ANE energy-/wall-clock-to-target and measured ANE
utilization (measured here directly, not taken from upstream's conflicting figures).
