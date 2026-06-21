#!/bin/zsh
# V2 placement-toggle equivalence gate (issue #12 / ADR 0004 seam).
#
# Builds the dynamic trainer twice — KNOB=0 (CPU reference) vs KNOB=1 (ANE) —
# runs the SAME pinned overfit batch through each, and diffs the raw dumped
# gradients (R1 oracle). Scratch init is deterministic (srand48(42) @ train.m),
# so both builds start from identical weights and see the same batch; any grad
# difference is purely the op's CPU-vs-ANE placement. The CPU placement is
# already R1-green vs the torch-fp64 oracle (results/r1_grad_diff.md), so an ANE
# build matching it within the fp16 noise floor is itself correct.
#
#   ./gate_placement.zsh RMSNORM_ANE                 # stories110m, default dims
#   ./gate_placement.zsh CLS_ANE qwen3_06b --dims=32000,1024,3072,2048,1024,28
#
# Exit code is the R1 verdict (0 = PASS). Also runs an R0 overfit smoke check.
set -euo pipefail
KNOB=${1:?need a knob name, e.g. RMSNORM_ANE}
MODEL=${2:-stories110m}
DIMS=${3:-}
HERE=${0:A:h}
cd "$HERE"

echo "== build CPU ($KNOB=0) =="
# CRITICAL: `make clean` first. The Makefile keys only on source mtimes, and
# EXTRA (-D flags) is NOT a prerequisite — so without a clean, a second
# `make EXTRA=...` with unchanged sources is a no-op and the KNOB=1 build never
# happens, silently diffing the KNOB=0 binary against itself (a vacuous cos=1.0).
make clean >/dev/null
make MODEL="$MODEL" EXTRA="-D${KNOB}=0" >/dev/null
./train --scratch --overfit --dump-grads /tmp/g_cpu.bin >/tmp/gate_cpu.log 2>&1 \
  || { echo "CPU run failed:"; tail -20 /tmp/gate_cpu.log; exit 3; }

echo "== build ANE ($KNOB=1) =="
make clean >/dev/null
make MODEL="$MODEL" EXTRA="-D${KNOB}=1" >/dev/null
./train --scratch --overfit --dump-grads /tmp/g_ane.bin >/tmp/gate_ane.log 2>&1 \
  || { echo "ANE run failed:"; tail -20 /tmp/gate_ane.log; exit 3; }

echo "== R1 grad diff (CPU vs ANE placement) =="
set +e
python3 grad_diff.py /tmp/g_cpu.bin /tmp/g_ane.bin ${DIMS:+"$DIMS"}
R1=$?
set -e

echo "== R0 overfit (ANE build, loss must collapse) =="
./train --scratch --overfit --steps 200 2>&1 | grep -E "^step" | tail -3

exit $R1
