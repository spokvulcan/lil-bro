# results

Ablation outputs: tables, loss curves, and the energy verdict.

Headline artifact — per component, vs the dense+AdamW control at best-tuned LR:

| component | Δ tokens-to-target | Δ steps | Δ energy (ANE) | Δ wall-clock | verdict |
|-----------|--------------------|---------|----------------|--------------|---------|

Plus the systems verdict: ANE vs MLX on energy-/wall-clock-to-target and measured
ANE utilization (measured here, not taken from upstream's conflicting figures).
