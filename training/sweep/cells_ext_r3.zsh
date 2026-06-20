# r3_110m LR-grid EXTENSION (ADR 0003 "extend if optimum on an edge"). At 110M two optima
# landed on grid edges — and these brackets decide the headline verdict:
#   - muon optimum dropped to 3e-3 (LOW edge); it held at 1e-2 through 13M/42M, so the
#     ceiling (+ head_dim 32->64) finally moved it. Extend DOWN: 1e-3, 3e-4. This is
#     verdict-critical — muon is the winner and mhc2@3e-3 (3.437) is currently edging out
#     muon@3e-3 (3.461); whether mHC "inverts to helping" hangs on muon's true floor.
#   - adamw optimum at 3e-4 (HIGH edge of the re-centered grid). Extend UP: 1e-3, to
#     confirm 3e-4 is a real bracketed minimum (r2_mid had 1e-3 worse, expect the same).
# Logs land in /tmp/ladder/r3_110m/logs -> analyze_ladder.py auto-includes them.
#   name | r2_extra | r0_extra | opt | lr
cells=(
"muon_lr1e3|NONE|NONE|muon|1e-3"
"muon_lr3e4|NONE|NONE|muon|3e-4"
"adamw_lr1e3|NONE|NONE|adamw|1e-3"
)
