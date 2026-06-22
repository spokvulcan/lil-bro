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
    play_selfplay_batch(&bev, NULL, rb, cfg, seed, gs);
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
            int16_t t16d[CHESS_NUM_TOKENS];
            for (int t = 0; t < CHESS_NUM_TOKENS; t++) t16d[t] = (int16_t)rb.buf[i].tokens[t];
            Position pd; chess_decode(&pd, t16d);
            int expected_side = side_white ? WHITE : BLACK;
            CHECK(pd.side == expected_side, "z-label[adj=%d] ply=%d: decoded stm=%d != before-move side %d (encoding/perspective drift)",
                  adjudicate, i, pd.side, expected_side);
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

// ---- 5b. n-step / TD(lambda) value densification (ADR 0006 build-step 7) -----------------
// Pure-math oracle: hand-set leaf_v + a known terminal, assert the TD(lambda) return. No
// net/ANE/GPU. lam=1 -> z_nstep == z (backward-compat); lam=0 -> 1-step TD off the next ply
// (last ply anchors on the terminal); lam=0.5 -> the blend. side[] alternates [W,B,W,B]; the
// terminal is white-mates (fstm=BLACK lost, fv=-1) so z = [+1,-1,+1,-1].
static void test_nstep_relabel_math(void) {
    printf("## n-step relabel: TD(lambda) value target from oracle leaf values + known terminal\n");
    const int N = 4;
    ReplaySample plies[4];
    int side[4] = { WHITE, BLACK, WHITE, BLACK };
    float leaf_v[4] = { 0.2f, -0.3f, 0.4f, -0.1f };
    float fv = -1.0f; int fstm = BLACK;
    const char *where = "n-step";

    memset(plies, 0, sizeof plies);
    relabel_value_targets(plies, leaf_v, N, side, fv, fstm, 1.0f);
    float e_lam1[4] = { +1.0f, -1.0f, +1.0f, -1.0f };
    for (int t = 0; t < N; t++)
        CHECK(fabsf(plies[t].z_nstep - e_lam1[t]) < 1e-6f,
              "%s lam=1 ply=%d: z_nstep=%.6f expected %.6f (==z)", where, t, plies[t].z_nstep, e_lam1[t]);

    memset(plies, 0, sizeof plies);
    relabel_value_targets(plies, leaf_v, N, side, fv, fstm, 0.0f);
    float e_lam0[4] = { +0.3f, -0.4f, +0.1f, -1.0f };
    for (int t = 0; t < N; t++)
        CHECK(fabsf(plies[t].z_nstep - e_lam0[t]) < 1e-6f,
              "%s lam=0 ply=%d: z_nstep=%.6f expected %.6f", where, t, plies[t].z_nstep, e_lam0[t]);

    memset(plies, 0, sizeof plies);
    relabel_value_targets(plies, leaf_v, N, side, fv, fstm, 0.5f);
    float e_lam5[4] = { 0.3875f, -0.475f, 0.55f, -1.0f };
    for (int t = 0; t < N; t++)
        CHECK(fabsf(plies[t].z_nstep - e_lam5[t]) < 1e-6f,
              "%s lam=0.5 ply=%d: z_nstep=%.6f expected %.6f", where, t, plies[t].z_nstep, e_lam5[t]);

    float lv_ext[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    memset(plies, 0, sizeof plies);
    relabel_value_targets(plies, lv_ext, N, side, +1.0f, WHITE, 0.5f);
    for (int t = 0; t < N; t++)
        CHECK(plies[t].z_nstep >= -1.0001f && plies[t].z_nstep <= 1.0001f,
              "%s boundary ply=%d: z_nstep=%.6f out of [-1,1]", where, t, plies[t].z_nstep);

    printf("   ok (lam=1.0->z, lam=0.0->1-step TD, lam=0.5->blend; bounded in [-1,1])\n\n");
}

// ---- 5c. n-step wiring: generation records leaf_v and relabels z_nstep (oracle evaluator) -
// End-to-end at the pure-C seam: a real B=1 self-play game through play_selfplay_batch (the
// oracle stands in for the net). At lam=1.0 z_nstep MUST equal z for every ply (the
// backward-compat contract that lets td_lambda default to 1.0 with zero behavior change).
static void test_nstep_relabel_integration(void) {
    printf("## n-step integration: generation populates z_nstep; == z at lam=1.0\n");
    SPConfig cfg = sp_defaults();
    cfg.td_lambda = 1.0f;
    cfg.B = 1; cfg.sims = 24; cfg.considered = 24; cfg.max_plies = 80; cfg.replay_cap = 100000;
    cfg.adjudicate = 1;
    ReplayBuffer rb; replay_init(&rb, cfg.replay_cap, 11);
    GenStats gs; int n = run_generation(&rb, &cfg, 987654321u, &gs);
    CHECK(n > 0, "n-step integ: generation produced no samples");
    int eq = 1, in_range = 1;
    for (int i = 0; i < n; i++) {
        if (rb.buf[i].z_nstep < -1.0001f || rb.buf[i].z_nstep > 1.0001f) in_range = 0;
        if (fabsf(rb.buf[i].z_nstep - rb.buf[i].z) > 1e-6f) eq = 0;
    }
    CHECK(in_range, "n-step integ: z_nstep out of [-1,1] (n=%d)", n);
    CHECK(eq, "n-step integ: z_nstep != z at lam=1.0 (%d samples) -- backward-compat broken", n);
    printf("   lam=1.0: %d samples, z_nstep == z for all, in [-1,1] (backward-compat holds)\n\n", n);
    replay_free(&rb);

    cfg.td_lambda = 0.5f;
    ReplayBuffer rb2; replay_init(&rb2, cfg.replay_cap, 13);
    GenStats gs2; int n2 = run_generation(&rb2, &cfg, 42424242u, &gs2);
    CHECK(n2 > 0, "n-step integ lam=0.5: no samples");
    int differ = 0;
    for (int i = 0; i < n2; i++) {
        CHECK(rb2.buf[i].z_nstep >= -1.0001f && rb2.buf[i].z_nstep <= 1.0001f,
              "n-step integ lam=0.5 ply=%d: z_nstep=%.3f out of [-1,1]", i, rb2.buf[i].z_nstep);
        if (fabsf(rb2.buf[i].z_nstep - rb2.buf[i].z) > 1e-4f) differ++;
    }
    CHECK(differ > 0, "n-step integ lam=0.5: z_nstep == z for ALL %d plies (relabel inactive at lam<1)", n2);
    printf("   lam=0.5: %d samples, %d differ from z (relabel active at lam<1), all in [-1,1]\n\n",
           n2, differ);
    replay_free(&rb2);
}

// A batched evaluator stub returning a fixed value + uniform priors (a label-value source
// distinct from any search evaluator, to prove leaf_v is sourced from label_bev).
static void const_eval(void *ctx, const Position *const *pos, int B,
                       const Move *const *legal, const int *n_legal,
                       float *const *priors, float *value) {
    (void)pos; (void)legal;
    float v = *(const float*)ctx;
    for (int b = 0; b < B; b++) {
        value[b] = v;
        float p = n_legal[b] > 0 ? 1.0f / (float)n_legal[b] : 0.0f;
        for (int i = 0; i < n_legal[b]; i++) priors[b][i] = p;
    }
}
static BatchedChessEvaluator make_const_evaluator(float v) {
    BatchedChessEvaluator out;
    float *ctx = (float*)malloc(sizeof(float)); *ctx = v;
    out.ctx = ctx; out.evaluate = const_eval;
    return out;
}

// ---- 5d. n-step label separation: leaf_v from label_bev, not the (warmup) search evaluator -
// The warmup wrapper blends the search value with a material heuristic; the TD label must use
// the net's OWN value (label_bev), never the warmup-blended search value (the "search prior,
// not labels" contract, selfplay.h). Fixed search bev (a warmup wrapper) + same seed =>
// identical games; varying label_bev must change z_nstep at lam<1 (proves leaf_v is sourced
// from label_bev). On the bug (leaf_v = root_value from the warmup search), z_nstep would be
// identical across the two label_bev.
static void test_nstep_label_separation(void) {
    printf("## n-step label separation: leaf_v from label_bev, not the warmup search evaluator\n");
    SPConfig cfg = sp_defaults();
    cfg.td_lambda = 0.5f;
    cfg.B = 1; cfg.sims = 16; cfg.considered = 16; cfg.max_plies = 40; cfg.replay_cap = 100000;
    cfg.adjudicate = 1;
    BatchedChessEvaluator inner = chess_oracle_batched_evaluator();
    BatchedChessEvaluator warm = make_warmup_evaluator(&inner, 1.0f);
    BatchedChessEvaluator labA = make_const_evaluator(0.3f);
    BatchedChessEvaluator labB = make_const_evaluator(-0.3f);
    ReplayBuffer ra; replay_init(&ra, cfg.replay_cap, 71);
    GenStats ga; play_selfplay_batch(&warm, &labA, &ra, &cfg, 111, &ga);
    ReplayBuffer rb; replay_init(&rb, cfg.replay_cap, 71);
    GenStats gb; play_selfplay_batch(&warm, &labB, &rb, &cfg, 111, &gb);
    int n = ra.count, differ = 0;
    CHECK(rb.count == n, "n-step label-sep: game lengths differ across label_bev (%d vs %d)", n, rb.count);
    for (int i = 0; i < n; i++)
        if (fabsf(ra.buf[i].z_nstep - rb.buf[i].z_nstep) > 1e-5f) differ++;
    CHECK(differ > 0, "n-step label-sep: z_nstep identical across label_bev -- leaf_v not sourced from label_bev");
    printf("   warm search + 2 label_bev: %d samples, %d z_nstep differ (leaf_v = label_bev, not search)\n\n", n, differ);
    replay_free(&ra); replay_free(&rb);
    warmup_evaluator_free(&warm); free(labA.ctx); free(labB.ctx);
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

// ---- 6b. net-vs-net match (self-anchored Elo core, ADR 0007): a STRONGER evaluator must
// beat a WEAKER one, decisively and with the correct sign. match_net_vs_net is the heart of
// the Elo metric; an always-draw or sign-flipped bug would silently make every Elo number
// meaningless. Strong = the G1 oracle (material + mate value + MCTS-Solver); weak = a const-0
// evaluator (uniform priors, no value signal -> uninformed ~random search). Pure C, no ANE.
static void test_match_net_vs_net(void) {
    printf("## net-vs-net match: a stronger evaluator beats a weaker one (decisive, correct sign)\n");
    SPConfig cfg = sp_defaults();
    cfg.eval_sims = 64; cfg.eval_considered = 8; cfg.eval_max_plies = 160;
    BatchedChessEvaluator strong = chess_oracle_batched_evaluator();
    BatchedChessEvaluator weak   = make_const_evaluator(0.0f);
    const int n = 16, open_plies = 2;

    // strong as A: must score > 0.5 AND produce decisive games (not silently all-draws)
    int Wa, Da, La;
    double sA = match_net_vs_net(&strong, &weak, &cfg, n, open_plies, 909, &Wa, &Da, &La);
    CHECK(Wa + Da + La == n, "match: W+D+L=%d != n=%d", Wa+Da+La, n);
    CHECK(Wa + La > 0, "match: ALL %d games drew — match never resolves (silent always-draw bug)", n);
    CHECK(sA > 0.5, "match: strong-as-A only scored %.3f vs weak (the stronger net should win)", sA);
    printf("   strong(A) vs weak(B):   W/D/L = %2d/%2d/%2d  score=%.3f\n", Wa, Da, La, sA);

    // sign check: swap roles, SAME seed -> weak-as-A must score < 0.5 (A should now LOSE)
    int Wb, Db, Lb;
    double sB = match_net_vs_net(&weak, &strong, &cfg, n, open_plies, 909, &Wb, &Db, &Lb);
    CHECK(sB < 0.5, "match: weak-as-A scored %.3f — sign is wrong (A should lose)", sB);
    printf("   weak(A)  vs strong(B):  W/D/L = %2d/%2d/%2d  score=%.3f (A loses — sign correct)\n", Wb, Db, Lb, sB);

    // determinism: same seed -> identical result
    int Wc, Dc, Lc;
    double sC = match_net_vs_net(&strong, &weak, &cfg, n, open_plies, 909, &Wc, &Dc, &Lc);
    CHECK(Wc==Wa && Dc==Da && Lc==La && sC==sA, "match is NOT deterministic for a fixed seed");

    // symmetric sanity: equal strength -> valid bookkeeping (score near 0.5, not asserted tight)
    int Ws, Ds, Ls;
    double sS = match_net_vs_net(&strong, &strong, &cfg, n, open_plies, 909, &Ws, &Ds, &Ls);
    CHECK(Ws + Ds + Ls == n, "match(self): W+D+L != n");
    printf("   strong vs strong:       W/D/L = %2d/%2d/%2d  score=%.3f (~0.5 expected)\n\n", Ws, Ds, Ls, sS);

    free(weak.ctx);
}

// ---- 7. warmup value-prior wrapper: blends net value with a material heuristic ----
// The G2 cold-start fix (ADR 0005 decision 8 fallback, measured-triggered). At cold start
// the net's value is random, so MCTS can't sharpen the policy prior into a useful target
// (loss_pol sticks at the ln(n_legal) entropy floor). The wrapper blends the net's value
// with a material heuristic for the first warmup_iters, giving MCTS a signal to sharpen
// against — a SEARCH PRIOR (like Dirichlet root noise), not external labels. Priors are
// untouched (purist-Zero: the net's own policy stays the prior). Verified here with the
// deterministic oracle standing in for the net, so the blend math is proven without the ANE.
static void test_warmup_value_prior(void) {
    printf("## warmup value-prior: blends net value with material heuristic; priors untouched\n");
    // Use the oracle as the "net" (material value + uniform priors): blending it with the
    // SAME material heuristic at frac=1.0 must reproduce the oracle value exactly; at frac=0.0
    // it must pass the oracle value through unchanged; priors must always match the oracle's.
    BatchedChessEvaluator inner = chess_oracle_batched_evaluator();
    Position pos[4];
    chess_startpos(&pos[0]);
    // a few distinct positions: short random walks
    uint64_t rr = 4242;
    for (int b = 1; b < 4; b++) {
        chess_startpos(&pos[b]);
        int steps = b * 3;
        for (int s = 0; s < steps; s++) { Move mv[MAX_MOVES]; int n = chess_legal_moves(&pos[b], mv);
            if (n == 0) { chess_startpos(&pos[b]); break; } Undo u; chess_make(&pos[b], mv[sm_below(&rr, n)], &u); }
    }
    const Position *pp[4]; const Move *lp[4]; int nl[4]; float *prp[4]; float val[4];
    static Move leg[4][MAX_MOVES]; static float pri[4][MAX_MOVES];
    for (int b = 0; b < 4; b++) { nl[b] = chess_legal_moves(&pos[b], leg[b]); pp[b] = &pos[b]; lp[b] = leg[b]; prp[b] = pri[b]; }

    // reference: the inner oracle's values + priors
    float val_ref[4]; float pri_ref[4][MAX_MOVES]; float *prp_ref[4] = { pri_ref[0], pri_ref[1], pri_ref[2], pri_ref[3] };
    inner.evaluate(inner.ctx, pp, 4, lp, nl, prp_ref, val_ref);

    // frac = 0.0 -> value passes through (blend is a no-op)
    BatchedChessEvaluator w0 = make_warmup_evaluator(&inner, 0.0f);
    float val0[4]; float pri0[4][MAX_MOVES]; float *prp0[4] = { pri0[0], pri0[1], pri0[2], pri0[3] };
    w0.evaluate(w0.ctx, pp, 4, lp, nl, prp0, val0);
    for (int b = 0; b < 4; b++) {
        CHECK(fabs(val0[b] - val_ref[b]) < 1e-5, "warmup frac=0: value changed (b=%d %.4f vs %.4f)", b, val0[b], val_ref[b]);
        for (int a = 0; a < nl[b]; a++) CHECK(fabs(pri0[b][a] - pri_ref[b][a]) < 1e-6, "warmup frac=0: prior changed");
    }

    // frac = 1.0 -> value == pure material heuristic (0.9*tanh(material/5)); priors unchanged
    BatchedChessEvaluator w1 = make_warmup_evaluator(&inner, 1.0f);
    float val1[4]; float pri1[4][MAX_MOVES]; float *prp1[4] = { pri1[0], pri1[1], pri1[2], pri1[3] };
    w1.evaluate(w1.ctx, pp, 4, lp, nl, prp1, val1);
    for (int b = 0; b < 4; b++) {
        float mat_v = 0.9f * tanhf((float)chess_material_diff(&pos[b]) / 5.0f);
        CHECK(fabs(val1[b] - mat_v) < 1e-5, "warmup frac=1: value != material heuristic (b=%d %.4f vs %.4f)", b, val1[b], mat_v);
        for (int a = 0; a < nl[b]; a++) CHECK(fabs(pri1[b][a] - pri_ref[b][a]) < 1e-6, "warmup frac=1: prior changed");
    }

    // frac = 0.5 -> value == 0.5*net + 0.5*material (the blend); priors unchanged
    BatchedChessEvaluator wh = make_warmup_evaluator(&inner, 0.5f);
    float valh[4]; float prih[4][MAX_MOVES]; float *prph[4] = { prih[0], prih[1], prih[2], prih[3] };
    wh.evaluate(wh.ctx, pp, 4, lp, nl, prph, valh);
    for (int b = 0; b < 4; b++) {
        float mat_v = 0.9f * tanhf((float)chess_material_diff(&pos[b]) / 5.0f);
        float expect = 0.5f * val_ref[b] + 0.5f * mat_v;
        CHECK(fabs(valh[b] - expect) < 1e-5, "warmup frac=0.5: value != blend (b=%d %.4f vs %.4f)", b, valh[b], expect);
        for (int a = 0; a < nl[b]; a++) CHECK(fabs(prih[b][a] - pri_ref[b][a]) < 1e-6, "warmup frac=0.5: prior changed");
    }

    // value stays in [-1, 1] for all frac (the [-0.9, 0.9] material range + [-1,1] net range)
    CHECK(val0[0] >= -1.0001f && val0[0] <= 1.0001f, "warmup: value out of [-1,1]");
    CHECK(val1[0] >= -1.0001f && val1[0] <= 1.0001f, "warmup: value out of [-1,1]");

    warmup_evaluator_free(&w0); warmup_evaluator_free(&w1); warmup_evaluator_free(&wh);
    printf("   ok (frac 0/0.5/1.0 verified; priors always = inner; values in [-1,1])\n\n");
}

int main(void) {
    chess_init();
    printf("# test_selfplay — oracle-driven TDD for the self-play orchestration (no ANE)\n\n");
    test_build_sample();
    test_select_move();
    test_generation();
    test_z_labeling();
    test_nstep_relabel_math();
    test_nstep_relabel_integration();
    test_nstep_label_separation();
    test_eval();
    test_match_net_vs_net();
    test_warmup_value_prior();
    if (g_fail) { printf("*** test_selfplay: FAILURES ***\n"); return 1; }
    printf("# test_selfplay: ALL TESTS PASSED\n");
    return 0;
}
