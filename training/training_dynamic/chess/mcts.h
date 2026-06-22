// mcts.h — Gumbel-AlphaZero MCTS (CPU), the policy-improvement operator.
//
// Build-step 3 of the chess-RL-on-ANE build order (ADR 0005, issue #17). This is
// REAL Gumbel-AlphaZero (Danihelka et al., 2022, "Policy improvement by planning
// with Gumbel"): Gumbel root-action selection + Sequential Halving over a low
// simulation budget, a deterministic completed-Q non-root selection, and value
// backup with a per-ply sign flip (negamax). It is NOT the probe's cost-shape stub
// (probe_chess.m mcts_one_sim), which models only the per-sim CPU bookkeeping.
//
// The tree is expanded over the #15 engine's legal moves (chess_legal_moves /
// chess_make); leaf values+priors come from a PLUGGABLE evaluator (ChessEvaluator).
// For the G1 gate the evaluator is a stub oracle (material + terminal mate/stalemate
// detection, uniform priors) — so this step depends only on the engine, not the
// trained heads (#16). At build-step 4 (#18) the real ANE net plugs into the SAME
// ChessEvaluator contract, unchanged: per-legal-move priors + a scalar value.
//
// Gate G1 (ADR 0005 gate ladder): on a mate-in-1/-2 suite the search returns the
// forced move given the oracle value, at a low sim budget. G1 is a MEASURED gate —
// test_mcts.c proves it on a real suite (with distractors) at a stated budget.
//
// Pure C, zero external dependencies (the project law, CLAUDE.md). Deterministic
// from cfg.seed: the Gumbel noise uses an explicit seeded RNG, so the G1 suite is
// reproducible (mirrors the #16 fixed-seed discipline).
#ifndef LILBRO_CHESS_MCTS_H
#define LILBRO_CHESS_MCTS_H

#include "chess.h"
#include <stdint.h>

// =============================================================================
// Pluggable leaf evaluator (the crux: oracle now, ANE net at #18)
// =============================================================================
//
// For a NON-TERMINAL position, fill priors[0..n_legal) (each >= 0, summing to 1)
// over the supplied legal moves (priors[i] is the prior for legal[i]) and RETURN the
// leaf value in [-1, 1] from the SIDE-TO-MOVE's perspective (+1 = side to move is
// winning). The search never calls this on a terminal position (n_legal == 0); it
// assigns terminals the game-theoretic value via chess_terminal_value() instead, so
// the net is never asked to value a checkmate/stalemate (its value there is junk).
//
// This is exactly what the #16 forward produces: a legal-masked policy softmax over
// the 4672 logits (read here as a prior over the position's legal moves) + a scalar
// value (e.g. WDL -> expected score W-L). ctx is opaque: NULL for the oracle, the net
// handle at #18. (Batched leaf eval across B parallel games is a #18 orchestration
// concern layered ABOVE this single-position contract via ctx buffering; it does not
// change this interface.)
typedef struct ChessEvaluator {
    void *ctx;
    float (*evaluate)(void *ctx, const Position *pos,
                      const Move *legal, int n_legal, float *priors);
} ChessEvaluator;

// The G1 stub oracle: value = 0.9*tanh(material_diff/5) (side-to-move pawns; |v|<0.9
// so terminal +-1 always dominates a material edge), uniform priors. Deterministic,
// no net. (The real ANE net replaces this at #18 by supplying a ChessEvaluator with
// the same contract.)
ChessEvaluator chess_oracle_evaluator(void);

// Game-theoretic value of a TERMINAL position (no legal moves) from the side-to-move's
// perspective: -1 if checkmated (no moves AND in check), 0 if stalemate (no moves, not
// in check). Behavior is undefined for non-terminal positions. The search assigns this
// to terminal leaves; #18's loop uses the same value as the game outcome z at terminals.
float chess_terminal_value(const Position *p);

// Side-to-move material balance in pawn units (P=1,N=3,B=3,R=5,Q=9; kings excluded):
// (my material) - (their material). Exposed for the oracle + its unit test.
int chess_material_diff(const Position *p);

// =============================================================================
// Search configuration + result
// =============================================================================

// sigma(q) = (c_visit + max_b N(b)) * c_scale * q   — the monotone Q-transform that
// turns action values into logit-space increments (Danihelka 2022). Paper defaults.
#define MCTS_C_VISIT 50.0f
#define MCTS_C_SCALE 1.0f
#define MCTS_DEFAULT_SEED 42ull
#define MCTS_MAX_PHASES 16    // >= ceil(log2(MAX_MOVES)); Sequential Halving phase bound

typedef struct {
    int      num_simulations; // total leaf evaluations (the budget; "low" is the point)
    int      max_considered;  // Gumbel considered actions at the root (paper default 16).
                              // Considered = min(max_considered, n_legal). With an
                              // UNINFORMATIVE oracle prior, set this >= n_legal so the
                              // forced move is guaranteed considered (the G1 gate does);
                              // at #18 a trained prior makes 16 the right low value.
    float    c_visit, c_scale;
    float    gamma;           // per-ply backup discount for UNPROVEN lines: a value d
                              // plies deep is worth gamma^d, nudging the value search
                              // toward faster wins / slower losses. Default 1.0 = pure
                              // undiscounted Gumbel-AZ (forced-mate distance is handled
                              // exactly by the solver proof, so the G1 gate is gamma-
                              // independent; #18 may set <1 as a play-quality knob).
    float    root_dirichlet_alpha; // Purist-Zero root exploration noise (ADR 0005 dec 8):
    float    root_dirichlet_frac;  // P(a) <- (1-frac)*P(a) + frac*Dir(alpha) at the root.
                              // Both 0 (mcts_default_config) => OFF, so mcts_search / the
                              // G1 + batched-equivalence gates are bit-identical. Self-play
                              // (#18) turns it ON in mcts_search_batched only (root noise is
                              // the cold-start exploration the deterministic G1 search omits).
    uint64_t seed;            // RNG seed for the Gumbel noise (determinism).
} MctsConfig;

// {num_simulations, max_considered=16, c_visit=50, c_scale=1, gamma=1.0, seed=42}.
MctsConfig mcts_default_config(int num_simulations);

typedef struct {
    Move  best_move;           // the action Gumbel root selection returns
    int   n_legal;             // legal moves at the root (0 => terminal: best_move=MOVE_NONE)
    Move  legal[MAX_MOVES];    // root legal moves (parallel to prior/visits/q)
    float prior[MAX_MOVES];    // evaluator prior P(a) at the root
    int   visits[MAX_MOVES];   // root visit count N(a) per legal move
    float q[MAX_MOVES];        // root Q(a) (root side-to-move perspective); 0 if N(a)==0
    float root_value;          // evaluator value at the root (side-to-move perspective)
    int   sims_done;           // simulations actually run (== num_simulations)
    int   nodes_used;          // tree nodes allocated (diagnostics)
    float c_visit, c_scale;    // echoed so the policy readouts are self-contained
} MctsResult;

// Run Gumbel-AlphaZero MCTS from *root using *ev. Fills *out (best_move + per-move
// statistics). *root is not modified. Deterministic given cfg->seed. If *root is
// terminal, out->n_legal==0 and out->best_move==MOVE_NONE.
void mcts_search(const Position *root, const ChessEvaluator *ev,
                 const MctsConfig *cfg, MctsResult *out);

// =============================================================================
// Vectorized batched-leaf search (the crux of build-step 4 / issue #18)
// =============================================================================
//
// The single-position contract above evaluates leaves ONE at a time — fine for the
// G1 oracle, ruinous for the ANE net (per-eval dispatch is the bound; the throughput
// probe shows batching B leaves into one forward is the never->days pivot, ~44x/pos).
// So #18 layers a lockstep driver ABOVE the single-position search: B parallel games
// step together, and every game's pending leaf eval — the root eval and each
// simulation's fresh-leaf eval — collapses into ONE batched forward.
//
// BatchedChessEvaluator evaluates B NON-TERMINAL positions in one call. For game b in
// [0,B): fill priors[b][0..n_legal[b]) (each >=0, summing to 1, prior for legal[b][i])
// and write value[b] in [-1,1] from b's side-to-move. Every supplied position has
// n_legal[b] >= 1 (terminals never reach the net — chess_terminal_value handles them).
// ctx is the net handle. This is the batched twin of ChessEvaluator.evaluate; the #16
// forward fills it via one seq=B*SEQ ANE forward + per-game legal-masked policy softmax
// + WDL->W-L value.
typedef struct BatchedChessEvaluator {
    void *ctx;
    void (*evaluate)(void *ctx, const Position *const *pos, int B,
                     const Move *const *legal, const int *n_legal,
                     float *const *priors, float *value);
} BatchedChessEvaluator;

// Optional coarse profiling for the throughput harness. Off by default; calling
// mcts_profile_reset() enables counters until process exit. Timings are wall seconds.
typedef struct {
    long searches;
    long sims;
    long nodes;
    long eval_calls;
    long eval_positions;
    double alloc_s;
    double root_expand_s;
    double root_eval_s;
    double sim_cpu_s;
    double leaf_eval_s;
} MctsProfile;

void mcts_profile_reset(void);
MctsProfile mcts_profile_snapshot(void);

// The G1 oracle wrapped as a batched evaluator (loops the single oracle per position).
// Exposed so the equivalence gate can prove mcts_search_batched is bit-identical to B
// independent mcts_search calls on the SAME deterministic oracle — i.e. the lockstep
// driver itself is correct, independent of the net's fp16 noise.
BatchedChessEvaluator chess_oracle_batched_evaluator(void);

// Run Gumbel-AlphaZero MCTS over B parallel roots sharing ONE batched evaluator. Game b
// runs EXACTLY mcts_search(roots[b], <equivalent single ev>, &cfgs[b]); the driver
// locksteps the B searches so their pending leaf evals batch into bev->evaluate calls.
// With a deterministic evaluator the result is bit-identical to B separate mcts_search
// calls (proven by test_mcts --batched). cfgs[b].seed gives each game independent Gumbel
// noise (self-play needs distinct games); outs[b] is filled exactly like mcts_search.
// roots/cfgs/outs are B-long arrays. B must be >= 1.
void mcts_search_batched(const Position *roots, int B, const BatchedChessEvaluator *bev,
                         const MctsConfig *cfgs, MctsResult *outs);

// =============================================================================
// Improved-policy readout — the target the self-play loop (#18) trains toward.
// Dense 4672-vectors over chess_move_to_index, zero on illegal indices.
// =============================================================================
//
// visit_policy:    normalized visit counts N(a)/sum_b N(b) — the classic AlphaZero
//                  policy target.
// improved_policy: the Gumbel completed-Q improved policy pi'(a) = softmax(logit(a) +
//                  sigma(completedQ(a))) — Danihelka 2022's recommended target, which
//                  is well-behaved at LOW sim budgets where the raw visit counts are
//                  degenerate (Sequential Halving concentrates visits on the winner).
//                  #18 should train toward THIS. Both are exposed; both sum to 1 over
//                  the root's legal indices.
void mcts_visit_policy   (const MctsResult *r, float out[CHESS_POLICY_SIZE]);
void mcts_improved_policy(const MctsResult *r, float out[CHESS_POLICY_SIZE]);

// =============================================================================
// Exposed for the per-component gate (test_mcts.c). These are the REAL functions
// mcts_search uses, so the test exercises the production code paths.
// =============================================================================

// Sequential-Halving schedule for m considered actions and n simulations. Fills
// size[i] (candidate-set size in phase i) and visits[i] (per-candidate simulations in
// phase i) for the returned number of phases P. P == ceil(log2(m)) (P==1 for m<=1);
// size halves m -> ... -> (2 or 1). The base allocation sum_i size[i]*visits[i] <= n;
// mcts_search spends the small remainder round-robin over the final phase so EXACTLY n
// simulations run. Requires n >= m for a sane schedule. Returns P (<= max_phases).
int mcts_seq_halving(int m, int n, int *size, int *visits, int max_phases);

// Gumbel considered set: the top min(m, n_legal) action indices by (log(prior[a]) +
// gumbel_a), where gumbel_a is seeded Gumbel(0) noise. Fills considered[] (sorted by
// that score, descending) and returns its size. Deterministic from seed.
int mcts_considered_set(const float *prior, int n_legal, int m,
                        uint64_t seed, int *considered);

#endif  // LILBRO_CHESS_MCTS_H
