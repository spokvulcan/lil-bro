#!/bin/zsh
# Phase 2 — R2 fast-learning run for every cell that passed the R0 gate (or all, with
# FORCE_R2=1). r2_small from scratch, fixed budget, identical data/seed. Headline metric =
# held-out val loss vs step (tokens-to-target proxy). (see results/v4_sweep.md)
#
# Usage:   zsh training/sweep/run_r2.zsh                          # main sweep (after run_r0)
#          CELLS_FILE=.../cells_c.zsh R2SUM=/tmp/sweep/r2c_summary.tsv FORCE_R2=1 \
#            zsh training/sweep/run_r2.zsh                         # an LR-fairness stage
# Env:     SWEEP_WORK · CELLS_FILE · R2SUM · FORCE_R2 · R2_STEPS · ACCUM · VAL_EVERY ·
#          VAL_BATCHES · WARMUP · SAMPLE_EVERY · SAMPLE_TOKENS · WD
set -u
SCRIPT_DIR=${0:A:h}
REPO=${SCRIPT_DIR:h:h}
TD=$REPO/training/training_dynamic
WORK=${SWEEP_WORK:-/tmp/sweep}
BIN=$WORK/bin; LOGS=$WORK/logs; CK=$WORK/ckpt; SWEEP=$WORK/sweep.log
R0RES=$WORK/r0_results.tsv; R2SUM=${R2SUM:-$WORK/r2_summary.tsv}
CELLS_FILE=${CELLS_FILE:-$SCRIPT_DIR/cells.zsh}
DATA=$REPO/training/tinystories_data00.bin
VAL=$REPO/training/tinystories_data01.bin
STEPS=${R2_STEPS:-800}
ACCUM=${ACCUM:-4}
VAL_EVERY=${VAL_EVERY:-50}
VAL_BATCHES=${VAL_BATCHES:-10}
WARMUP=${WARMUP:-40}
SAMPLE_EVERY=${SAMPLE_EVERY:-0}
SAMPLE_TOKENS=${SAMPLE_TOKENS:-32}
WD=${WD:-0.1}
FORCE_R2=${FORCE_R2:-0}

mkdir -p $BIN $LOGS $CK
cd $TD || exit 1
SDK="$(xcrun --show-sdk-path)"
source $CELLS_FILE

log(){ print -r -- "[$(date +%H:%M:%S)] $*" | tee -a $SWEEP; }

# which cells passed R0
typeset -A passed
if [[ -f $R0RES ]]; then
  while IFS=$'\t' read -r nm fl vd; do [[ $vd == PASS ]] && passed[$nm]=1; done < $R0RES
fi

log "===== PHASE 2: R2 fast-learning (steps=$STEPS, accum=$ACCUM, val_every=$VAL_EVERY, wd=$WD, force=$FORCE_R2) ====="
[[ -f $R2SUM ]] || printf 'cell\topt\tlr\tfinal_val\tbest_val\twall_s\n' > $R2SUM

for line in $cells; do
  parts=("${(@s:|:)line}")
  name=$parts[1]; r2x=$parts[2]; opt=$parts[4]; lr=$parts[5]
  [[ $r2x == NONE ]] && r2x=""
  if [[ $FORCE_R2 != 1 && -z ${passed[$name]:-} ]]; then
    log "R2 $name: SKIP (did not pass R0)"; continue
  fi
  out=$BIN/r2_$name
  if ! xcrun clang -O2 -DACCELERATE_NEW_LAPACK -framework Foundation -framework IOSurface -framework Accelerate \
        -isysroot "$SDK" -fobjc-arc ${=r2x} -include models/gen_r2_small.h -o $out train.m 2>$LOGS/r2build_$name.log; then
    log "R2 $name: BUILD FAILED"; continue
  fi
  log "R2 $name: START (opt=$opt lr=$lr extra='$r2x')"
  t0=$(date +%s)
  $out --scratch --data $DATA --val-data $VAL --steps $STEPS \
       --val-every $VAL_EVERY --val-batches $VAL_BATCHES --sample-every $SAMPLE_EVERY --sample-tokens $SAMPLE_TOKENS \
       --lr $lr --wd $WD --accum $ACCUM --warmup $WARMUP --opt $opt --ckpt $CK/r2_$name.bin \
       > $LOGS/r2run_$name.log 2>&1
  t1=$(date +%s)
  wall=$(( t1 - t0 ))
  fv=$(grep -oE 'val_loss=[0-9.eE+-]+' $LOGS/r2run_$name.log | tail -1 | sed 's/val_loss=//')
  bv=$(grep -oE 'val_loss=[0-9.eE+-]+' $LOGS/r2run_$name.log | sed 's/val_loss=//' | sort -g | head -1)
  [[ -z $fv ]] && fv=NA; [[ -z $bv ]] && bv=NA
  log "R2 $name: DONE final_val=$fv best_val=$bv wall=${wall}s"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$opt" "$lr" "$fv" "$bv" "$wall" >> $R2SUM
done
log "===== PHASE 2 done. Summary: $R2SUM ====="
column -t -s$'\t' $R2SUM | tee -a $SWEEP
