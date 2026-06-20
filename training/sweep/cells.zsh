# Main sweep cell table — DeepSeek-V4 config sweep (see results/v4_sweep.md).
# Format per line:  name | r2_extra | r0_extra | opt | lr
#   r2_extra : -D flags for the gen_r2_small build (HD=32)
#   r0_extra : -D flags for the gen_r0_overfit correctness build (HD=16; partial-rope dim differs)
#   opt      : adamw | muon   (R2 only; R0 always uses adamw to isolate architecture-backward)
#   lr       : R2 learning rate
# NONE = empty flag set. Architecture knobs map to PRD child mechanisms #6-#11 + Muon.
cells=(
"min|NONE|NONE|adamw|3e-4"
"qk_norm|-DQK_NORM=1|-DQK_NORM=1|adamw|3e-4"
"attn_sink|-DATTN_SINK=1|-DATTN_SINK=1|adamw|3e-4"
"swiglu|-DSWIGLU_CLAMP=1|-DSWIGLU_CLAMP=1|adamw|3e-4"
"prope|-DROPE_ROTARY_DIMS=16|-DROPE_ROTARY_DIMS=8|adamw|3e-4"
"nhc2|-DN_HC=2|-DN_HC=2|adamw|3e-4"
"nhc4|-DN_HC=4|-DN_HC=4|adamw|3e-4"
"mtp1|-DMTP_DEPTH=1|-DMTP_DEPTH=1|adamw|3e-4"
"muon|NONE|NONE|muon|3e-4"
"max|-DQK_NORM=1 -DATTN_SINK=1 -DSWIGLU_CLAMP=1 -DROPE_ROTARY_DIMS=16 -DN_HC=4 -DMTP_DEPTH=1|-DQK_NORM=1 -DATTN_SINK=1 -DSWIGLU_CLAMP=1 -DROPE_ROTARY_DIMS=8 -DN_HC=4 -DMTP_DEPTH=1|muon|3e-4"
)
