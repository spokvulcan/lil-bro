// test_selfplay.c — oracle-driven TDD for the self-play orchestration (selfplay.c),
// build-step 4 of chess-RL-on-ANE (ADR 0005, issue #18). NO ANE: the net evaluator is
// swapped for the deterministic G1 oracle (material + mate/stalemate, uniform priors), so
// the loop's LOGIC is proven independently of the ANE forward — the subtle, correctness-
// critical bits the gate ultimately leans on:
//   1. build_sample / generation produce VALID replay samples (tokens decode to the
//      searched position; sparse policy is a distribution over exactly the legal moves).
//   2. z-LABELING sign: a winner's plies are labeled +1, a loser's -1, a draw's 0 — proven
//      against the KNOWN game outcome (white moves on even plies from startpos).
//   3. DETERMINISM: same seed + config => bit-identical replay (the project's law).
//   4. The EVAL ladder's W/D/L bookkeeping is internally consistent AND a material+mate
//      searcher beats a random-mover (a behavioral floor: search must do *something*).
//
// Pure C, zero deps (the project law). Mirrors test_mcts.c / test_replay.c style: asserts +
// a summary line; exit 0 == all green.
#include "selfplay.h"
#include "mcts.h"
#include "chess.h"
#include "replay.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { printf("   FAIL: "); printf(__VA_ARGS__); printf("\n"); g_fail = 1; } } while (0)

// A replay sample is well-formed: tokens decode to a legal position; the sparse policy is a
// distribution over exactly that position's legal moves; z is a terminal outcome label.
static void validate_sample(const ReplaySample *s, const char *where) {
    int16_t t16[CHESS_NUM_TOKENS];
    for (int t = 0; t < CHESS_NUM_TOKENS; t++) t16[t] = (int16_t)s->tokens[t];
    Position p; int ok = chess_decode(&p, t16);
    CHECK(ok, "%s: tokens do not decode to a Position", where);
    Move legal[MAX_MOVES]; int nl = chess_legal_moves(&p, legal);
    CHECK(s->n_policy == nl, "%s: n_policy=%d != legal count=%d", where, s->n_policy, nl);
    // every policy entry maps to a legal move's index, and the probs sum to ~1
    uint8_t mask[CHESS_POLICY_SIZE]; chess_legal_mask(legal, nl, mask);
    double sum = 0; int all_legal = 1, all_nonneg = 1;
    for (int e = 0; e < s->n_policy; e++) {
        if (s->policy_idx[e] < 0 || s->policy_idx[e] >= CHESS_POLICY_SIZE || !mask[s->policy_idx[e]]) all_legal = 0;
        if (s->policy_p[e] < 0) all_nonneg = 0;
        sum += s->policy_p[e];
    }
    CHECK(all_legal, "%s: a policy index is not a legal-move index", where);
    CHECK(all_nonneg, "%s: a policy prob is negative", where);
    CHECK(fabs(sum - 1.0) < 1e-4, "%s: policy probs sum to %.6f (!= 1)", where, sum);
    CHECK(s->z == -1.0f || s->z == 0.0f || s->z == 1.0f, "%s: z=%.3f not in {-1,0,1}", where, s->z);
}

// ---- 1. build_sample on a real search result -------------------------------------------
static void test_build_sample(void) {
    printf("## build_sample: sparse policy is a valid distribution over the legal moves\n");
    Position p; chess_startpos(&p);
    ChessEvaluator oracle = chess_oracle_evaluator();
    MctsConfig cfg = mcts_default_config(64); cfg.max_considered = 32;
    MctsResult r; mcts_search(&p, &oracle, &cfg, &r);
    SPConfig sp = sp_defaults();
    float dense[CHESS_POLICY_SIZE];
    ReplaySample s; build_sample(&s, &p, &r, &sp, dense);
    validate_sample(&s, "build_sample/startpos");
    // tokens must round-trip to the startpos
    int16_t t16[CHESS_NUM_TOKENS]; chess_encode(&p, t16);
    int same = 1; for (int t = 0; t < CHESS_NUM_TOKENS; t++) if ((uint16_t)t16[t] != s.tokens[t]) same = 0;
    CHECK(same, "build_sample: tokens != chess_encode(startpos)");
    // visit-policy variant is also valid
    sp.use_improved_policy = 0;
    ReplaySample s2; build_sample(&s2, &p, &r, &sp, dense);
    validate_sample(&s2, "build_sample/visit-policy");
    printf("   ok (n_policy=%d, improved + visit both valid)\n\n", s.n_policy);
}

// ---- 2. select_move ---------------------------------------------------------------------
static void test_select_move(void) {
    printf("## select_move: greedy after temp_moves, in-distribution during, MOVE_NONE at terminal\n");
    Position p; chess_startpos(&p);
    ChessEvaluator oracle = chess_oracle_evaluator();
    MctsConfig cfg = mcts_default_config(64); cfg.max_considered = 32;
    MctsResult r; mcts_search(&p, &oracle, &cfg, &r);
    SPConfig sp = sp_defaults(); sp.temp = 1.0f; sp.temp_moves = 15;
    uint64_t rng = 777;
    // ply >= temp_moves -> best_move (deterministic)
    CHECK(select_move(&r, 99, &sp, &rng) == r.best_move, "select_move: ply>=temp_moves not greedy");
    // during the temperature window -> always a legal root move
    for (int i = 0; i < 50; i++) {
        Move m = select_move(&r, 0, &sp, &rng);
        int found = 0; for (int a = 0; a < r.n_legal; a++) if (r.legal[a] == m) found = 1;
        CHECK(found, "select_move: sampled move not in root legal set");
    }
    // temp<=0 -> greedy even at ply 0
    sp.temp = 0.0f;
    CHECK(select_move(&r, 0, &sp, &rng) == r.best_move, "select_move: temp<=0 not greedy");
    // terminal search result (no legal moves) -> MOVE_NONE
    MctsResult term; memset(&term, 0, sizeof term); term.n_legal = 0; term.best_move = MOVE_NONE;
    CHECK(select_move(&term, 0, &sp, &rng) == MOVE_NONE, "select_move: terminal not MOVE_NONE");
    printf("   ok\n\n");
}

// ---- 3 + 4. generation: every sample valid, and bit-identical across two same-seed runs --
static int run_generation(ReplayBuffer *rb, const SPConfig *cfg, uint64_t seed, GenStats *gs) {
    BatchedChessEvaluator bev = chess_oracle_batched_evaluator();
    *gs = (GenStats){0};
    play_selfplay_batch(&bev, rb, cfg, seed, gs);
    return rb->count;
}
static void test_generation(void) {
    printf("## generation: B-game lockstep self-play -> valid samples; deterministic from seed\n");
    SPConfig cfg = sp_defaults();
    cfg.B = 16; cfg.sims = 24; cfg.considered = 24; cfg.max_plies = 60; cfg.replay_cap = 200000;
    ReplayBuffer rb; replay_init(&rb, cfg.replay_cap, 99);
    GenStats gs; int n = run_generation(&rb, &cfg, 12345, &gs);
    CHECK(n > 0, "generation produced no samples");
    CHECK(gs.games == cfg.B, "generation: games=%ld != B=%d", gs.games, cfg.B);
    CHECK(gs.plies == n, "generation: plies=%ld != replay count=%d", gs.plies, n);
    for (int i = 0; i < rb.count; i++) validate_sample(&rb.buf[i], "generation");
    printf("   %d samples over %ld games (W:%ld B:%ld draws:%ld); all valid\n",
           n, gs.games, gs.wins_w, gs.wins_b, gs.draws);

    // determinism: same seed + config -> identical buffer contents
    ReplayBuffer rb2; replay_init(&rb2, cfg.replay_cap, 99);
    GenStats gs2; int n2 = run_generation(&rb2, &cfg, 12345, &gs2);
    int identical = (n == n2);
    for (int i = 0; identical && i < n; i++) {
        if (rb.buf[i].z != rb2.buf[i].z || rb.buf[i].n_policy != rb2.buf[i].n_policy) identical = 0;
        for (int t = 0; identical && t < CHESS_NUM_TOKENS; t++) if (rb.buf[i].tokens[t] != rb2.buf[i].tokens[t]) identical = 0;
        for (int e = 0; identical && e < rb.buf[i].n_policy; e++)
            if (rb.buf[i].policy_idx[e] != rb2.buf[i].policy_idx[e] || rb.buf[i].policy_p[e] != rb2.buf[i].policy_p[e]) identical = 0;
    }
    CHECK(identical, "generation is NOT deterministic across two same-seed runs");
    printf("   deterministic: two seed=12345 runs are bit-identical (%d samples)\n\n", n);
    replay_free(&rb); replay_free(&rb2);
}

// ---- 5. z-labeling sign: winner's plies +1, loser's -1, draw 0, derived from the OUTCOME --
// Play B=1 games (the replay then holds exactly one game's plies, in order). From startpos
// the side at recorded ply i is WHITE iff i is even, so the expected label is fully
// determined by who won (GenStats), independent of selfplay.c's formula. Run for BOTH
// adjudicate off (decisive only via mate, outcome fv<0) and on (capped games adjudicated by
// material, exercising the fv>0 winner path + the GenStats winner attribution).
static void test_z_labeling_mode(int adjudicate, int *decisive_out) {
    SPConfig cfg = sp_defaults();
    cfg.B = 1; cfg.sims = 32; cfg.considered = 32; cfg.max_plies = 160; cfg.replay_cap = 100000;
    cfg.adjudicate = adjudicate;
    int decisive_seen = 0, draws_seen = 0, checked = 0;
    for (uint64_t seed = 1; seed <= 30 && (!decisive_seen || !draws_seen); seed++) {
        ReplayBuffer rb; replay_init(&rb, cfg.replay_cap, 7);
        GenStats gs; int n = run_generation(&rb, &cfg, seed*1000003u, &gs);
        CHECK(gs.games == 1, "z-label[adj=%d]: expected 1 game, got %ld", adjudicate, gs.games);
        int white_won = (gs.wins_w == 1), black_won = (gs.wins_b == 1), draw = (gs.draws == 1);
        CHECK(white_won + black_won + draw == 1, "z-label[adj=%d]: ambiguous outcome", adjudicate);
        for (int i = 0; i < n; i++) {
            int side_white = (i % 2 == 0);   // startpos: white moves on even plies
            float expect;
            if (draw) expect = 0.0f;
            else if (white_won) expect = side_white ? +1.0f : -1.0f;
            else                expect = side_white ? -1.0f : +1.0f;
            CHECK(rb.buf[i].z == expect, "z-label[adj=%d] seed=%llu ply=%d: z=%.1f expected %.1f (W%d B%d D%d)",
                  adjudicate, (unsigned long long)seed, i, rb.buf[i].z, expect, white_won, black_won, draw);
        }
        if (draw) draws_seen = 1; else decisive_seen = 1;
        checked++;
        replay_free(&rb);
    }
    CHECK(decisive_seen, "z-label[adj=%d]: no decisive game in 30 seeds (sign path untested)", adjudicate);
    printf("   adj=%d: verified %d B=1 games (decisive:%s draw:%s) — all plies correctly signed\n",
           adjudicate, checked, decisive_seen?"yes":"no", draws_seen?"yes":"no");
    *decisive_out = decisive_seen;
}
static void test_z_labeling(void) {
    printf("## z-labeling: per-ply sign matches the game outcome (winner +1, loser -1, draw 0)\n");
    int d0, d1;
    test_z_labeling_mode(0, &d0);   // mate-only decisive (fv<0)
    test_z_labeling_mode(1, &d1);   // + material adjudication (exercises fv>0 + winner stats)
    printf("\n");
}

// ---- 6. eval ladder: W/D/L bookkeeping + a material+mate searcher beats a random-mover ---
static void test_eval(void) {
    printf("## eval ladder: W/D/L sums to n_games; deterministic; oracle-search beats random\n");
    SPConfig cfg = sp_defaults();
    cfg.eval_sims = 32; cfg.considered = 32; cfg.eval_max_plies = 120;
    BatchedChessEvaluator bev = chess_oracle_batched_evaluator();
    int n_games = 30, w, d, l;
    double sr = eval_vs_opponent(&bev, &cfg, opp_random, n_games, 555, &w, &d, &l);
    CHECK(w + d + l == n_games, "eval: W+D+L=%d != n_games=%d", w+d+l, n_games);
    CHECK(fabs(sr - ((double)w + 0.5*d)/n_games) < 1e-9, "eval: score != (W+0.5D)/n");
    // determinism
    int w2, d2, l2; double sr2 = eval_vs_opponent(&bev, &cfg, opp_random, n_games, 555, &w2, &d2, &l2);
    CHECK(w==w2 && d==d2 && l==l2 && sr==sr2, "eval is NOT deterministic for a fixed seed");
    // behavioral floor: a material+mate searcher must beat a random-mover decisively
    CHECK(sr > 0.65, "eval: oracle-search only scored %.3f vs random (search is not working)", sr);
    printf("   vs random: W/D/L = %d/%d/%d  score=%.3f (deterministic)\n", w, d, l, sr);
    // vs the 1-ply greedy baseline: counts must still be consistent
    int gw, gd, gl; double sg = eval_vs_opponent(&bev, &cfg, opp_greedy, n_games, 556, &gw, &gd, &gl);
    CHECK(gw + gd + gl == n_games, "eval(greedy): W+D+L != n_games");
    printf("   vs greedy: W/D/L = %d/%d/%d  score=%.3f\n\n", gw, gd, gl, sg);
}

int main(void) {
    chess_init();
    printf("# test_selfplay — oracle-driven TDD for the self-play orchestration (no ANE)\n\n");
    test_build_sample();
    test_select_move();
    test_generation();
    test_z_labeling();
    test_eval();
    if (g_fail) { printf("*** test_selfplay: FAILURES ***\n"); return 1; }
    printf("# test_selfplay: ALL TESTS PASSED\n");
    return 0;
}
