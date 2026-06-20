# Scale-ladder R0 correctness gate — the 3 DISTINCT builds per rung (see ADR 0003).
# muon & adamw share the no-knob "plain" build (optimizer is a runtime flag), so only
# the N_HC compile flag splits builds. Each is overfit on one pinned real batch AT THE
# RUNG'S OWN DIMS (run_r0 with MODEL=<rung>) — the (size x config) correctness gate the
# larger d512/d768 builds need (the architecture knobs were already R0-green at tiny
# dims in the V4 sweep; what is new up the ladder is the size). At d768 the hot lr2e-3
# overfit explodes in fp16, so the ladder R0 uses grad clipping (R0_CLIP=1 R0_LR=1e-3).
#
#   name | r2_extra | r0_extra | opt | lr      (run_r0 uses name + r0_extra only)
cells=(
"plain|NONE|NONE|adamw|1e-3"
"mhc2|-DN_HC=2|-DN_HC=2|adamw|1e-3"
"mhc4|-DN_HC=4|-DN_HC=4|adamw|1e-3"
)
