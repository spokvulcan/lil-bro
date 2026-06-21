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

# Chess heads FD gate (build-step 2, issue #16): analytic-vs-numeric backward check
# on the policy/value/posenc heads. Pure C, no ANE — the fast TDD gate before G0.
test_heads: chess/test_heads_cpu.c chess/chess_heads.h
	$(CC) $(CHESS_CFLAGS) -o chess/test_heads chess/test_heads_cpu.c -lm
	./chess/test_heads

# Chess G0 trainer (build-step 2, issue #16): policy + WDL value heads + 2D posenc
# + AlphaZero loss on the ANE. The trunk matmuls run on the ANE via gen_dyn_matmul_mil
# (the same primitive train.m's woFwd/qBwd/wotBwd use); heads + loss on the CPU floor.
# `make g0` builds and runs the overfit gate (both losses -> ~0 == G0-green).
train_chess: train_chess.m config.h io.h cpu_ops.h mil_dynamic.h chess/chess_heads.h chess/chess.h models/chess_g0.h $(CHESS_DEPS)
	$(CC) $(CFLAGS) $(EXTRA) -include models/chess_g0.h -o train_chess train_chess.m chess/chess.c

g0: train_chess
	./train_chess --overfit

clean:
	rm -f train probe_chess train_chess chess/perft chess/test_chess chess/test_heads

FORCE:

.PHONY: clean FORCE test_engine test_heads g0
