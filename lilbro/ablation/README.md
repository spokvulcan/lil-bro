# lilbro/ablation

The experiment runner that turns "test DeepSeek-V4 ideas" into honest numbers.

**Backlog (sequenced; each kept only if it improves the efficiency frontier):**
1. baseline = dense + AdamW (control)
2. + Muon
3. + MTP
4. *(Phase 3)* + mHC, then + CSA/HCA

**Per-cell protocol**
- iso-loss: cost to reach the rung-2 target val loss
- per-config **LR sweep (≥3)**; report best-LR result
- vary exactly one component; controls fixed
- gate: config must pass the R1 correctness diff before its number is trusted

**Output:** per component, Δ vs control in `tokens-to-target` (headline) and
`steps / energy / wall-clock to-target` (secondary), on ANE and MLX → `results/`.
