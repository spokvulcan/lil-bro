# Stage E — fair-LR test of the architecture (symmetry with the optimizer fairness).
# mHC/attn_sink were only seen at AdamW's under-tuned 3e-4. Re-test at AdamW's real
# optimum (1e-3), and isolate mHC ON THE BEST OPTIMIZER (Muon @ 1e-2) — the cleanest
# "does the top architecture mechanism help the top optimizer?" test. (see results/v4_sweep.md)
cells=(
"nhc4_adamw1e3|-DN_HC=4|-DN_HC=4|adamw|1e-3"
"nhc4_muon1e2|-DN_HC=4|-DN_HC=4|muon|1e-2"
"attn_sink_adamw1e3|-DATTN_SINK=1|-DATTN_SINK=1|adamw|1e-3"
)
