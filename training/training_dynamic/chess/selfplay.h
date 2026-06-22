// selfplay.h — the self-play orchestration: generation (B games in lockstep), the
// fixed-opponent eval ladder, move selection, and the run config. Build-step 4 of
// chess-RL-on-ANE (ADR 0005, issue #18).
//
// THE SPLIT (why this is its own pure-C file): everything here is evaluator-AGNOSTIC —
// it drives a BatchedChessEvaluator (mcts.h), which is the G1 oracle in the unit test
// (test_selfplay.c, no ANE) and the #16 ANE net in production (train_selfplay.m). So the
// subtle, correctness-critical logic — the game state machine, terminal/draw handling,
// the z-labeling (outcome from each ply's side-to-move), replay population, and the
// W/D/L eval bookkeeping — is testable WITHOUT the ANE, against the deterministic oracle.
// The net glue (the batched forward, the learner, fp16 loss-scaling) lives in the .m.
//
// Pure C, zero deps beyond the engine + mcts + replay (the project law, CLAUDE.md).
// Deterministic from seed + config (no Date/rand): the move-selection + opponent +
// Dirichlet RNGs are all explicit splitmix64 streams.
#ifndef LILBRO_CHESS_SELFPLAY_H
#define LILBRO_CHESS_SELFPLAY_H

#include "chess.h"
#include "mcts.h"
#include "replay.h"
#include <stdint.h>

// ---- deterministic RNG (splitmix64) for move-selection + opponents --------------------
// (search/Dirichlet RNG is mcts's own, seeded per game). static inline so both the .c and
// the .m get a private copy with no link conflict.
static inline uint64_t sm_next(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static inline double sm_uniform(uint64_t *s) { return ((double)(sm_next(s) >> 11) + 1.0) * (1.0/9007199254740994.0); }
static inline int sm_below(uint64_t *s, int n) { return n <= 1 ? 0 : (int)(sm_next(s) % (uint64_t)n); }

// ============================================================================
// Run configuration. The loop is fully reproducible from seed + these fields.
// (Learner-only fields are consumed by train_selfplay.m; selfplay.c ignores them.)
// ============================================================================
typedef struct {
    // generation
    int      B, sims, considered;
    float    dir_alpha, dir_frac;          // Purist-Zero root noise (ADR 0005 dec 8)
    float    temp; int temp_moves;         // move-temperature: sample for the opening plies
    int      max_plies;
    int      use_improved_policy;          // 1 = Gumbel improved-policy target (default), 0 = visits
    int      curriculum, curriculum_plies; // default-OFF random-opening fallback (dec 8)
    int      adjudicate;                    // default-OFF cold-start mitigation: label a game that
                                             // hits the ply/50-move cap by MATERIAL (+-1 if a side is
                                             // up >= SP_ADJ_THRESH pawns, else draw) instead of a flat
                                             // draw, so the value head gets a signal when weak self-play
                                             // never reaches mate. TRAINING-ONLY: the eval ladder always
                                             // scores real game outcomes (a capped eval game is a draw).
    int      warmup_iters;                  // cold-start value-prior warmup (dec 8 fallback, MEASURED-
    float    warmup_frac;                   // triggered): for iter < warmup_iters, blend the net's leaf
                                             // value with a material heuristic at frac = warmup_frac *
                                             // max(0, 1 - iter/warmup_iters) (linear decay to 0). A
                                             // SEARCH PRIOR (like Dirichlet), not labels. 0 = purist-Zero
                                             // (the steady state). Default ON: the cold-start desert is
                                             // measured-real (loss_pol sticks at ln(n_legal) without it).
    float    td_lambda;                      // TD(lambda) for the value target: 1.0 = terminal z (legacy), 0.0 = 1-step TD
    // replay + learner (train_selfplay.m)
    int      replay_cap, learner_batch, learner_steps, iters;
    float    lr, loss_scale, grad_clip, wd, value_weight;
    // eval ladder (eval_considered decoupled from generation's `considered`: a TRAINED
    // policy guides a narrow/cheap eval search, while generation needs a broad search to
    // sharpen the policy target from uninformative cold-start priors — see the G1 note)
    int      eval_games, eval_every, eval_sims, eval_considered, eval_max_plies;
    // benchmark harness
    int      bench_games;
    // diagnostics
    int      profile;                 // --profile: per-iter phase timing (gen/learner/eval/ckpt)
    int      use_mps;                 // --mps: route trunk matmuls through Metal/MPS instead of ANE
    int      use_mps_graph;           // --mps-graph: route eval forward + learner trunk fwd/bwd through MPSGraph
    // bookkeeping
    uint64_t seed;
    const char *ckpt;
    int      resume;
} SPConfig;

SPConfig sp_defaults(void);
// Parse argv into a config; sets *mode = 0 smoke / 1 g2 / 2 selfcheck / 3 bench. Clamps B<=170.
SPConfig sp_parse(int argc, char **argv, int *mode);

// Per-batch generation statistics (diagnostics + the gate's sanity printout).
typedef struct {
    long games, plies, sims, nodes;
    long wins_w, wins_b, draws;
    uint64_t checksum;
} GenStats;

// A fixed opponent: pick a legal move for *p, advancing *rng. NULL move iff no legal moves.
typedef Move (*OpponentFn)(const Position *p, uint64_t *rng);
Move opp_random(const Position *p, uint64_t *rng);  // uniform random legal move
Move opp_greedy(const Position *p, uint64_t *rng);  // 1-ply material-greedy (+ mate-in-1)

// Pick the move to play from a finished search: temperature-sample visits^(1/temp) for the
// opening plies (ply < temp_moves) for self-play diversity, the search's best_move after.
Move select_move(const MctsResult *r, int ply, const SPConfig *cfg, uint64_t *rng);

// Turn a searched position + its MCTS result into a replay sample (sparse policy target
// from the improved/visit distribution; z left 0, filled at game end). dense_scratch is
// CHESS_POLICY_SIZE floats of caller-owned scratch.
void build_sample(ReplaySample *s, const Position *pos, const MctsResult *r,
                  const SPConfig *cfg, float *dense_scratch);

void relabel_value_targets(ReplaySample *plies, const float *leaf_v, int n_plies,
                           const int *side, float fv, int fstm, float td_lambda);

// GENERATION: play cfg->B self-play games in lockstep (one mcts_search_batched per ply, so
// all games' leaf evals batch into one bev->evaluate), recording every searched position as
// a replay sample with the game-outcome z. Deterministic from base_seed + cfg. st may be NULL.
void play_selfplay_batch(const BatchedChessEvaluator *bev, const BatchedChessEvaluator *label_bev,
                         ReplayBuffer *rb, const SPConfig *cfg, uint64_t base_seed, GenStats *st);

// EVAL: play n_games net-vs-opponent games in lockstep, the net alternating colors, the
// net's to-move searches batched. Returns the net's score (W + 0.5 D)/n_games and the raw
// W/D/L counts (net perspective). Deterministic from seed. Net plays greedily (no Dirichlet
// /temperature) at cfg->eval_sims.
double eval_vs_opponent(const BatchedChessEvaluator *bev, const SPConfig *cfg,
                        OpponentFn opp, int n_games, uint64_t seed, int *W, int *D, int *Lo);

// Warmup value-prior wrapper (ADR 0005 decision 8 fallback, MEASURED-triggered): returns a
// BatchedChessEvaluator that wraps `inner` and blends the leaf VALUE with a material
// heuristic: value_out = (1-frac)*inner.value + frac * 0.9*tanh(material_diff/5). The PRIORS
// pass through unchanged (purist-Zero: the net's own policy stays the search prior; the value
// prior just gives MCTS a signal to sharpen the policy against at cold start, where the net's
// value is random). frac in [0,1]: 0 = pure net (purist-Zero, the steady state), 1 = pure
// material heuristic (the G1 oracle's value). This is a SEARCH PRIOR (like Dirichlet root
// noise), NOT external labels — no imitation, no Stockfish. Free with warmup_evaluator_free.
BatchedChessEvaluator make_warmup_evaluator(const BatchedChessEvaluator *inner, float frac);
void warmup_evaluator_free(BatchedChessEvaluator *w);

#endif  // LILBRO_CHESS_SELFPLAY_H
