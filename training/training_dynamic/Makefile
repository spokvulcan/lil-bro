CC = xcrun clang
CFLAGS = -O2 -DACCELERATE_NEW_LAPACK -framework Foundation -framework IOSurface -framework Accelerate \
         -isysroot $(shell xcrun --show-sdk-path) -fobjc-arc

# Model selection: make MODEL=qwen3_06b  (default)
#   Available: stories110m, qwen3_06b
MODEL ?= qwen3_06b
MODEL_HDR = models/$(MODEL).h

# Ad-hoc knob overrides for R0 gates, e.g. make MODEL=gen_r0_overfit EXTRA=-DATTN_SINK=1
# (the V4 knobs are #ifndef-guarded in config.h / the emitted header, so a -D wins).
EXTRA ?=

train: FORCE train.m config.h io.h cpu_ops.h mil_dynamic.h $(MODEL_HDR)
	@echo "Building for model: $(MODEL) $(EXTRA)"
	$(CC) $(CFLAGS) $(EXTRA) -include $(MODEL_HDR) -o train train.m

# Chess RL throughput probe (build-step 0, ADR 0005). Model-agnostic: drives chess
# shapes at runtime, so any model header satisfies config.h's #error guard.
probe_chess: probe_chess.m config.h io.h mil_dynamic.h models/gen_r2_small.h
	$(CC) $(CFLAGS) $(EXTRA) -include models/gen_r2_small.h -o probe_chess probe_chess.m

# GPU/MPS tracer bullet (issue #18 frontier): measures MPSMatrixMultiplication
# vs the ANE dispatch floor at the real chess matmul shapes. Standalone — no
# config.h deps (dims are #defined in the probe). Links Metal + MPS frameworks.
probe_mps: probe_mps.m
	$(CC) $(CFLAGS) -framework Metal -framework MetalPerformanceShaders \
	  -framework MetalPerformanceShadersGraph -o probe_mps probe_mps.m

# Phase 2 tracer bullet: full 2-layer trunk as ONE MPSGraph (all ops on GPU,
# 1 command buffer, no CPU↔GPU sync). Tests the "100x hypothesis" — whether
# eliminating pipeline bubbles gives the big win.
probe_mps_graph: probe_mps_graph.m
	$(CC) $(CFLAGS) -framework Metal -framework MetalPerformanceShaders \
	  -framework MetalPerformanceShadersGraph -o probe_mps_graph probe_mps_graph.m

# Slice 0 autodiff spike (ADR 0006 build-step 0, issue #23): standalone probe that
# builds the chess trunk as ONE MPSGraph + a scalar source and obtains the backward
# via MPSGraph gradientForPrimaryTensor:withTensors:, then (a) grad-diffs the result
# vs the CPU chess_trunk_backward to fp32 cosine and (b) runs a G0 overfit using the
# autodiff trunk backward + the existing CPU heads/loss/optimizer. Gates the GPU
# backward method (hybrid autodiff vs hand-written). Reuses the chess trunk/heads/net
# headers, so it links the chess deps + Metal/MPS/MPSGraph like train_chess.
probe_autodiff: probe_autodiff.m config.h io.h cpu_ops.h mil_dynamic.h chess/chess_heads.h \
                chess/chess_net.h chess/chess.h models/chess_g0.h $(CHESS_DEPS)
	$(CC) $(CFLAGS) $(EXTRA) -include models/chess_g0.h \
		-framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
		-o probe_autodiff probe_autodiff.m chess/chess.c

# Chess bitboard engine (build-step 1, ADR 0005, issue #15). Pure C, ZERO deps:
# no model header, no ANE/Foundation/Accelerate frameworks. perft is the gate.
CHESS_CFLAGS = -O2 -Wall -Wextra -std=c11 -isysroot $(shell xcrun --show-sdk-path)
CHESS_DEPS = chess/chess.c chess/chess.h

perft: chess/perft.c $(CHESS_DEPS)
	$(CC) $(CHESS_CFLAGS) -o chess/perft chess/perft.c chess/chess.c

# Engine unit tests (codec round-trip, move-index bijection, make/unmake restore).
test_engine: chess/test_chess.c $(CHESS_DEPS)
	$(CC) $(CHESS_CFLAGS) -o chess/test_chess chess/test_chess.c chess/chess.c
	./chess/test_chess

# Chess heads FD gate (build-step 2, issue #16): analytic-vs-numeric backward check
# on the policy/value/posenc heads. Pure C, no ANE — the fast TDD gate before G0.
test_heads: chess/test_heads_cpu.c chess/chess_heads.h
	$(CC) $(CHESS_CFLAGS) -o chess/test_heads chess/test_heads_cpu.c -lm
	./chess/test_heads

# Trunk attention backward FD gate (issue #36): analytic-vs-numeric check on
# attn_cpu_backward_batched's dQ/dK/dV. The heads have test_heads_cpu.c; the trunk
# attention backward had no FD gate, which let the dQ NEON fold bug (HD>8) slip
# through. Exercises the REAL static functions in chess_net.h at chess_g0 dims, so
# it links the chess deps + Metal/MPS/MPSGraph like probe_autodiff.
test_trunk_attn: test_trunk_attn.m config.h io.h cpu_ops.h mil_dynamic.h chess/chess_heads.h \
           chess/chess_net.h chess/chess.h models/chess_g0.h $(CHESS_DEPS)
	$(CC) $(CFLAGS) $(EXTRA) -include models/chess_g0.h \
		-framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
		-o test_trunk_attn test_trunk_attn.m chess/chess.c
	./test_trunk_attn

# Chess optimizer split gate (slice 5): Muon for canonical 2D trunk matrices,
# AdamW for embeddings, RMSNorm gains, and policy/value heads.
test_chess_optimizer: test_chess_optimizer.m config.h io.h cpu_ops.h mil_dynamic.h \
                      chess/chess_heads.h chess/chess_net.h chess/chess.h models/chess_g0.h $(CHESS_DEPS)
	$(CC) $(CFLAGS) $(EXTRA) -include models/chess_g0.h \
		-framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
		-o test_chess_optimizer test_chess_optimizer.m chess/chess.c
	./test_chess_optimizer

# Chess G0 trainer (build-step 2, issue #16): policy + WDL value heads + 2D posenc
# + AlphaZero loss on the ANE. The trunk matmuls run on the ANE via gen_dyn_matmul_mil
# (the same primitive train.m's woFwd/qBwd/wotBwd use); heads + loss on the CPU floor.
# `make g0` builds and runs the overfit gate (both losses -> ~0 == G0-green).
train_chess: train_chess.m config.h io.h cpu_ops.h mil_dynamic.h chess/chess_heads.h chess/chess_net.h chess/chess.h models/chess_g0.h $(CHESS_DEPS)
	$(CC) $(CFLAGS) $(EXTRA) -include models/chess_g0.h \
		-framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
		-o train_chess train_chess.m chess/chess.c

g0: train_chess
	./train_chess --overfit

# Gumbel-AlphaZero MCTS (build-step 3, ADR 0005, issue #17). Pure C, ZERO deps:
# mcts.c expands the engine's legal moves; the leaf evaluator is a pluggable interface
# (stub oracle for G1). -lm for tanh/log/exp. The binary is a file target so test_mcts
# (full component + gate suite) and g1 (the mate-suite gate, exit 0/1) share one build.
CHESS_MCTS_DEPS = chess/mcts.c chess/mcts.h $(CHESS_DEPS)

chess/test_mcts: chess/test_mcts.c $(CHESS_MCTS_DEPS)
	$(CC) $(CHESS_CFLAGS) -o chess/test_mcts chess/test_mcts.c chess/mcts.c chess/chess.c -lm

# Per-component gate (Gumbel root selection, Sequential Halving, backup sign, policy
# readout) + the G1 mate suite.
test_mcts: chess/test_mcts
	./chess/test_mcts

# G1 gate: Gumbel-MCTS picks the forced move on a mate-in-1/-2 suite (with distractors)
# given the oracle value, at a stated low sim budget. Exit 0 == G1-green.
g1: chess/test_mcts
	./chess/test_mcts --g1

# ===== Self-play loop (build-step 4, ADR 0005, issue #18) =====
# The full single-process, one-ANE-client self-play RL loop. The hot path (self-play
# stepping, replay, batched leaf-eval forward + learner train) is C/.m on the ANE; the
# eval-side orchestration (config, Elo/Stockfish, run driver) is Python in lilbro/chess/.
SELFPLAY_DEPS = chess/selfplay.c chess/selfplay.h chess/replay.c chess/replay.h \
                $(CHESS_MCTS_DEPS)

# CPU-only TDD seams (replay-buffer invariants, evaluator readout math): pure C, no ANE,
# the fast gate before any ANE training run.
chess/test_replay: chess/test_replay.c chess/replay.c chess/replay.h $(CHESS_DEPS)
	$(CC) $(CHESS_CFLAGS) -o chess/test_replay chess/test_replay.c chess/replay.c chess/chess.c -lm

test_replay: chess/test_replay
	./chess/test_replay

# Self-play orchestration TDD (oracle-driven, NO ANE): generation produces valid replay
# samples with correct z-labeling (winner's plies +1, loser's -1, draws 0); the eval ladder's
# W/D/L bookkeeping is right and a material+mate searcher beats a random-mover. Pure C — the
# fast gate that proves the loop's logic before spending ANE time on a training run.
chess/test_selfplay: chess/test_selfplay.c chess/selfplay.c chess/selfplay.h chess/replay.c chess/replay.h $(CHESS_MCTS_DEPS)
	$(CC) $(CHESS_CFLAGS) -o chess/test_selfplay chess/test_selfplay.c chess/selfplay.c chess/mcts.c chess/replay.c chess/chess.c -lm

test_selfplay: chess/test_selfplay
	./chess/test_selfplay

# Full self-play trainer (generation + replay + learner) on the one ANE client.
# `make selfplay` builds; the run driver (lilbro/chess/) invokes ./train_selfplay --config ...
# Links Metal + MPS frameworks for the optional --mps matmul backend (chess_net.h).
train_selfplay: train_selfplay.m config.h io.h cpu_ops.h mil_dynamic.h chess/chess_heads.h \
                chess/chess_net.h chess/selfplay.h chess/replay.h chess/mcts.h chess/chess.h models/chess_g0.h \
                chess/selfplay.c chess/replay.c chess/mcts.c chess/chess.c
	$(CC) $(CFLAGS) $(EXTRA) -include models/chess_g0.h \
		-framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
		-o train_selfplay train_selfplay.m \
		chess/selfplay.c chess/replay.c chess/mcts.c chess/chess.c

# G2 gate: vectorized self-play whose win-rate climbs vs a fixed random-mover then a weak
# baseline. `make g2 [G2ARGS=...]` runs the loop's built-in eval harness and exits 0/1.
selfplay: train_selfplay

g2: train_selfplay
	./train_selfplay --g2 $(G2ARGS)

clean:
	rm -f train probe_chess probe_mps probe_mps_graph probe_autodiff train_chess train_selfplay test_trunk_attn test_chess_optimizer \
	      chess/perft chess/test_chess chess/test_heads chess/test_mcts chess/test_replay chess/test_selfplay

FORCE:

.PHONY: clean FORCE test_engine test_heads test_trunk_attn test_chess_optimizer g0 test_mcts g1 test_replay test_selfplay selfplay g2
