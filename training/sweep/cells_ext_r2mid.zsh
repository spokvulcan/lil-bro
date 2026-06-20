# r2_mid LR-grid EXTENSION (ADR 0003 "extend if optimum on an edge"). At 42M the adamw
# optimum fell to the 3e-4 low edge and the mHC optimum to the 3e-3 low edge (muon held
# at 1e-2, interior). Extend both downward to bracket the true optimum before the verdict.
# Logs land in the same /tmp/ladder/r2_mid/logs dir -> analyze_ladder.py auto-includes them.
cells=(
"adamw_lr1e4|NONE|NONE|adamw|1e-4"
"adamw_lr3e5|NONE|NONE|adamw|3e-5"
"mhc2_lr1e3|-DN_HC=2|-DN_HC=2|muon|1e-3"
"mhc4_lr1e3|-DN_HC=4|-DN_HC=4|muon|1e-3"
)
