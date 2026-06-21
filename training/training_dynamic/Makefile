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

clean:
	rm -f train probe_chess chess/perft chess/test_chess chess/test_mcts

FORCE:

.PHONY: clean FORCE test_engine test_mcts g1
