#!/bin/zsh
# Phase 1 — R0 overfit correctness gate for every sweep cell (see results/v4_sweep.md).
# Tests architecture fwd+bwd (adamw, lr 2e-3) drives a pinned batch to ~0. A cell that
# does not collapse (< PASS_THRESH, finite) is NOT trusted at R2. Always uses adamw so
# R0 isolates architecture-backward correctness (the optimizer axis is results/muon_v4.md).
#
# Usage:   zsh training/sweep/run_r0.zsh
# Env:     SWEEP_WORK (artifact dir, default /tmp/sweep) · CELLS_FILE · R0_STEPS · PASS_THRESH · PYTHON
set -u
SCRIPT_DIR=${0:A:h}
REPO=${SCRIPT_DIR:h:h}
TD=$REPO/training/training_dynamic
WORK=${SWEEP_WORK:-/tmp/sweep}
BIN=$WORK/bin; LOGS=$WORK/logs; CK=$WORK/ckpt; SWEEP=$WORK/sweep.log; R0RES=$WORK/r0_results.tsv
CELLS_FILE=${CELLS_FILE:-$SCRIPT_DIR/cells.zsh}
PYTHON=${PYTHON:-$REPO/.venv/bin/python}
STEPS=${R0_STEPS:-800}
PASS_THRESH=${PASS_THRESH:-0.1}
DATA=r0_synthetic.bin

mkdir -p $BIN $LOGS $CK
cd $TD || exit 1
SDK="$(xcrun --show-sdk-path)"
source $CELLS_FILE

# auto-generate the R0 synthetic byte stream (128 random bytes, vocab 256) if missing
if [[ ! -f $TD/$DATA ]]; then
  (cd $REPO && $PYTHON -c "from lilbro.eval import write_token_stream; import numpy as np; write_token_stream('$TD/$DATA', np.random.default_rng(0).integers(0,256,128))")
fi

log(){ print -r -- "[$(date +%H:%M:%S)] $*" | tee -a $SWEEP; }

log "===== PHASE 1: R0 correctness gate (steps=$STEPS, adamw, lr=2e-3, pass<$PASS_THRESH) ====="
printf 'cell\tr0_final_loss\tverdict\n' > $R0RES

for line in $cells; do
  parts=("${(@s:|:)line}")
  name=$parts[1]; r0x=$parts[3]
  [[ $r0x == NONE ]] && r0x=""
  out=$BIN/r0_$name
  # build (zsh ${=r0x} forces word-splitting of multi-flag extras)
  if ! xcrun clang -O2 -DACCELERATE_NEW_LAPACK -framework Foundation -framework IOSurface -framework Accelerate \
        -isysroot "$SDK" -fobjc-arc ${=r0x} -include models/gen_r0_overfit.h -o $out train.m 2>$LOGS/r0build_$name.log; then
    log "R0 $name: BUILD FAILED"; printf '%s\tNA\tBUILD_FAIL\n' "$name" >> $R0RES; continue
  fi
  $out --scratch --overfit --data $DATA --steps $STEPS --accum 1 --wd 0 --lr 2e-3 --warmup 10 \
       --opt adamw --ckpt $CK/r0_$name.bin > $LOGS/r0run_$name.log 2>&1
  floss=$(grep -oE 'loss=[0-9.eE+-]+' $LOGS/r0run_$name.log | tail -1 | sed 's/loss=//')
  verdict="FAIL"
  if [[ -n $floss ]]; then
    if print -r -- "$floss" | grep -qiE 'nan|inf'; then verdict="NAN"
    elif (( floss < PASS_THRESH )); then verdict="PASS"; fi
  else floss="NA"; fi
  log "R0 $name: final_loss=$floss -> $verdict"
  printf '%s\t%s\t%s\n' "$name" "$floss" "$verdict" >> $R0RES
done
log "===== PHASE 1 done. Results: $R0RES ====="
column -t -s$'\t' $R0RES | tee -a $SWEEP
