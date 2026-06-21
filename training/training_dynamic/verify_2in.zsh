#!/bin/zsh
# One-shot rigorous R1 verification that does NOT trust the Makefile mtime:
# `make clean` before EVERY build so the -D flags actually take effect.
# Settles whether the 2-input func-param / conv eval produces CORRECT grads.
set -euo pipefail
cd /Users/owl/projects/lil-bro/training/training_dynamic

build() {  # $1 = extra flags, $2 = output tag
  make clean >/dev/null
  echo -n "  built: "; make MODEL=stories110m EXTRA="$1" 2>&1 | grep "Building for model"
  ./train --scratch --overfit --dump-grads "/tmp/verify_$2.bin" >/dev/null 2>&1
}

echo "== building (clean before each) =="
build "-DW2T_FUNCPARAM=0"                  ref0
build "-DW2T_FUNCPARAM=0"                  ref0b
build "-DW2T_FUNCPARAM=1"                  test_fp
build "-DW2T_FUNCPARAM=1 -DCONV_PROBE=1"   test_conv

echo "\n=== SANITY: ref0 vs ref0b (clean rebuild determinism, expect cos 1.0) ==="
python3 grad_diff.py /tmp/verify_ref0.bin /tmp/verify_ref0b.bin || true
echo "\n=== CLAIM B (matmul 2-in): ref0 vs W2T_FUNCPARAM=1 ==="
python3 grad_diff.py /tmp/verify_ref0.bin /tmp/verify_test_fp.bin || true
echo "\n=== CLAIM B (conv 2-in): ref0 vs W2T_FUNCPARAM=1 CONV_PROBE=1 ==="
python3 grad_diff.py /tmp/verify_ref0.bin /tmp/verify_test_conv.bin || true
