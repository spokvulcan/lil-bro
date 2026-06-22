// test_mcts.c — the G1 gate + per-component gate for Gumbel-AlphaZero MCTS (#17).
//
// G1 (ADR 0005 gate ladder) is a MEASURED gate: it proves the search returns the
// forced move on a real mate-in-1/-2 suite (with material distractors) at a stated
// low simulation budget — not "the search looks right". The mate suite + every key
// here was cross-checked with python-chess offline (eval-side tooling only, never the
// hot path) and is re-verified at runtime by the engine itself: a "mate-in-1" answer
// must actually deliver checkmate; a "mate-in-2" answer must be a genuine forced-mate
// key (after it, every opponent reply allows an immediate White checkmate). So a wrong
// move cannot pass even if a hardcoded key were mistyped.
//
// Per-component checks (Gumbel root selection, Sequential Halving, value-backup sign,
// policy readout) exercise the REAL functions mcts_search uses. Pattern mirrors
// test_chess.c: per-check OK/FAIL, nonzero exit on any failure.
//
// Build: see Makefile targets `test_mcts` (all) and `g1` (the mate suite only).

#include "mcts.h"
#include "chess.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

static int g_fail = 0;
static void check(int cond, const char *name) {
    printf("  %-54s %s\n", name, cond ? "OK" : "*** FAIL ***");
    if (!cond) g_fail++;
}

// =============================================================================
// Per-component: evaluator / oracle (terminal value, material, priors)
// =============================================================================
static void test_oracle(void) {
    printf("[oracle: terminal value + material + priors]\n");
    Position p;
    // Fool's mate: White to move AND checkmated -> terminal value -1 (loss for stm).
    chess_from_fen(&p, "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3");
    check(chess_terminal_value(&p) == -1.0f, "checkmate (stm mated) -> -1");
    // Stalemate: Black to move, no legal moves, not in check -> 0 (draw).
    chess_from_fen(&p, "7k/5Q2/5K2/8/8/8/8/8 b - - 0 1");
    check(chess_terminal_value(&p) == 0.0f, "stalemate -> 0");

    // Material balance (side-to-move perspective, pawns).
    chess_from_fen(&p, "4k3/8/8/8/8/8/8/R3K3 w - - 0 1");
    check(chess_material_diff(&p) == 5, "white up a rook (stm=W) -> +5");
    chess_from_fen(&p, "4k3/8/8/8/8/8/8/R3K3 b - - 0 1");
    check(chess_material_diff(&p) == -5, "same board, stm=B -> -5");
    chess_startpos(&p);
    check(chess_material_diff(&p) == 0, "startpos -> 0 material");

    // Oracle eval: uniform priors summing to 1; value sign matches material; |v| < 0.9
    // (so a terminal +-1 always dominates any material edge).
    ChessEvaluator ev = chess_oracle_evaluator();
    chess_from_fen(&p, "4k3/8/8/8/8/8/8/R3K3 w - - 0 1");
    Move legal[MAX_MOVES];
    int n = chess_legal_moves(&p, legal);
    float pri[MAX_MOVES];
    float v = ev.evaluate(ev.ctx, &p, legal, n, pri);
    float s = 0; for (int i = 0; i < n; i++) s += pri[i];
    check(fabsf(s - 1.0f) < 1e-5f, "oracle priors sum to 1");
    int uniform = 1; for (int i = 0; i < n; i++) if (fabsf(pri[i] - 1.0f/n) > 1e-6f) uniform = 0;
    check(uniform, "oracle priors uniform");
    check(v > 0.0f && v < 0.9f, "oracle value: up material -> (0, 0.9)");

    // The oracle is a PURE heuristic (like the net at #18): even with a checkmate on
    // the board it just reports material — finding the mate is the SEARCH's job (the
    // MCTS-Solver proven-mate overlay), not the evaluator's.
    chess_from_fen(&p, "6k1/5ppp/8/8/8/8/8/R6K w - - 0 1");  // Ra8# available
    n = chess_legal_moves(&p, legal);
    float vm = ev.evaluate(ev.ctx, &p, legal, n, pri);
    check(vm > 0.0f && vm < 0.9f, "oracle ignores mate-on-board -> material only");
}

// =============================================================================
// Per-component: Sequential Halving schedule
// =============================================================================
static void test_seq_halving(void) {
    printf("[Sequential Halving schedule]\n");
    int size[MCTS_MAX_PHASES], vis[MCTS_MAX_PHASES];

    int P = mcts_seq_halving(16, 512, size, vis, MCTS_MAX_PHASES);
    check(P == 4, "m=16 -> 4 phases (ceil log2 16)");
    check(size[0]==16 && size[1]==8 && size[2]==4 && size[3]==2, "sizes halve 16->8->4->2");
    int allpos = 1; for (int i = 0; i < P; i++) if (vis[i] < 1) allpos = 0;
    check(allpos, "every phase gives >= 1 visit per candidate");
    int sum = 0; for (int i = 0; i < P; i++) sum += size[i]*vis[i];
    check(sum == 512, "m=16,n=512 base budget == 512 (divides evenly)");

    check(mcts_seq_halving(2, 100, size, vis, MCTS_MAX_PHASES) == 1, "m=2 -> 1 phase");
    check(mcts_seq_halving(1, 100, size, vis, MCTS_MAX_PHASES) == 1, "m=1 -> 1 phase");

    P = mcts_seq_halving(30, 512, size, vis, MCTS_MAX_PHASES);
    check(P == 5, "m=30 -> 5 phases");
    check(size[0]==30 && size[1]==15 && size[2]==8 && size[3]==4 && size[4]==2,
          "sizes halve 30->15->8->4->2");
    sum = 0; for (int i = 0; i < P; i++) sum += size[i]*vis[i];
    check(sum <= 512 && sum >= 256, "m=30 base budget in (n/2, n]");
}

// =============================================================================
// Per-component: Gumbel considered set (root action pre-selection)
// =============================================================================
static void test_considered_set(void) {
    printf("[Gumbel considered set]\n");
    // Strongly peaked prior on action 3 of 8: with overwhelming logit mass it must be
    // in the top-4 considered set regardless of the Gumbel perturbation.
    float prior[8] = {0.02f,0.02f,0.02f,0.86f,0.02f,0.02f,0.02f,0.02f};
    int c1[8], c2[8];
    int k = mcts_considered_set(prior, 8, 4, 42, c1);
    check(k == 4, "considered size == min(m, n_legal) = 4");
    int has3 = 0; for (int i = 0; i < k; i++) if (c1[i] == 3) has3 = 1;
    check(has3, "overwhelmingly-likely action is considered");
    mcts_considered_set(prior, 8, 4, 42, c2);
    check(memcmp(c1, c2, (size_t)k*sizeof(int)) == 0, "deterministic for a fixed seed");
    k = mcts_considered_set(prior, 8, 64, 42, c1);
    check(k == 8, "m >= n_legal -> consider all moves");
}

// =============================================================================
// The G1 mate suite + engine-based answer verification (no python-chess at runtime)
// =============================================================================
//
// The stated G1 budget. MEASURED (--sweep + a 40-seed robustness scan): the whole suite
// — incl. mate-in-2 with a queen distractor — is solved for EVERY seed at >= 384 sims
// (mate-in-1 at <= 128; mate-in-2 needs the key to survive Sequential Halving and then
// complete its solver proof). 512 is that threshold plus margin. A trained policy prior
// (#18) will solve these far cheaper — it concentrates the early visits on the key.
#define G1_SIMS       512
#define G1_CONSIDERED 64    // >= max legal moves in the suite -> the forced move is
                            // always in the considered set under the uninformative
                            // oracle prior (#18's trained prior makes 16 the right low
                            // value). The Gumbel/Sequential-Halving machinery still runs.

typedef struct { const char *name, *fen; int dist; } Pos;  // dist: 1 = M1, 2 = M2
static const Pos SUITE[] = {
    {"m1a back-rank Ra8#",           "6k1/5ppp/8/8/8/8/8/R6K w - - 0 1",            1},
    {"m1b K+Q box",                  "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1",              1},
    {"m1c Re8# (pawn distractor)",   "6k1/4Rppp/8/8/8/8/8/6K1 w - - 0 1",           1},
    {"m1d Rf8#",                     "7k/6pp/8/8/8/8/6PP/5R1K w - - 0 1",           1},
    {"m1e Nf7# vs Nxg8 (+rook)",     "r5rk/6pp/7N/8/8/8/8/6QK w - - 0 1",           1},
    {"m2a two-rook ladder",          "7k/8/8/8/8/8/R7/1R5K w - - 0 1",              2},
    {"m2b K+Q mate-in-2",            "6k1/8/5K2/8/8/8/8/1Q6 w - - 0 1",             2},
    {"m2c K+Q corner mate-in-2",     "k7/8/2K5/8/8/8/8/7Q w - - 0 1",               2},
    {"m2d Ra8! vs gxh3 (+queen)",    "6k1/5ppp/8/8/8/7q/5PPP/R5K1 w - - 0 1",       2},
};
#define N_SUITE ((int)(sizeof(SUITE)/sizeof(SUITE[0])))

// Make `m` on a copy of `p` and report whether it is immediate checkmate.
static int delivers_checkmate(const Position *p, Move m) {
    Position c = *p; Undo u; chess_make(&c, m, &u);
    Move r[MAX_MOVES];
    return chess_legal_moves(&c, r) == 0 && chess_in_check(&c);
}
// Does the side to move have ANY checkmating move?
static int side_has_checkmate(const Position *p) {
    Move mv[MAX_MOVES]; int n = chess_legal_moves(p, mv);
    for (int i = 0; i < n; i++) if (delivers_checkmate(p, mv[i])) return 1;
    return 0;
}
// Is `key` a genuine forced-mate-in-2 key? After it (not an immediate mate/stalemate),
// every opponent reply must leave us a checkmating move. Engine-only -> independent of
// the search and of any hardcoded answer (a wrong move cannot pass).
static int is_forced_mate_in_2_key(const Position *p, Move key) {
    Position c = *p; Undo u; chess_make(&c, key, &u);
    Move reply[MAX_MOVES];
    int nr = chess_legal_moves(&c, reply);
    if (nr == 0) return 0;                         // key itself ended the game
    for (int i = 0; i < nr; i++) {
        Position d = c; Undo u2; chess_make(&d, reply[i], &u2);
        Move t[MAX_MOVES];
        if (chess_legal_moves(&d, t) == 0) return 0;  // we are stalemated/mated -> not forced
        if (!side_has_checkmate(&d)) return 0;        // no mate after this reply -> not forced
    }
    return 1;
}

// Run the search and verify the returned move with the engine.
static int solves(const Pos *pos, int n, int considered, uint64_t seed) {
    Position p; chess_from_fen(&p, pos->fen);
    ChessEvaluator ev = chess_oracle_evaluator();
    MctsConfig cfg = mcts_default_config(n);
    cfg.max_considered = considered; cfg.seed = seed;
    MctsResult r; mcts_search(&p, &ev, &cfg, &r);
    if (r.sims_done != n) return 0;                // budget must be spent exactly
    return pos->dist == 1 ? delivers_checkmate(&p, r.best_move)
                          : is_forced_mate_in_2_key(&p, r.best_move);
}

// ---- search: mate-in-1 suite (+ value-backup sign: a win backs up to positive Q) ----
static int find_move(const MctsResult *r, const char *uci) {
    char buf[6];
    for (int a = 0; a < r->n_legal; a++)
        if (strcmp(chess_move_to_uci(r->legal[a], buf), uci) == 0) return a;
    return -1;
}
static void test_search_mate_in_1(void) {
    printf("[search: mate-in-1 suite @ %d sims]\n", G1_SIMS);
    for (int i = 0; i < N_SUITE; i++) {
        if (SUITE[i].dist != 1) continue;
        Position p; chess_from_fen(&p, SUITE[i].fen);
        ChessEvaluator ev = chess_oracle_evaluator();
        MctsConfig cfg = mcts_default_config(G1_SIMS);
        cfg.max_considered = G1_CONSIDERED;
        MctsResult r; mcts_search(&p, &ev, &cfg, &r);
        check(delivers_checkmate(&p, r.best_move), SUITE[i].name);
        // value-backup sign: the chosen mating move backs up to a positive Q near +1.
        for (int a = 0; a < r.n_legal; a++)
            if (chess_move_to_index(r.legal[a]) == chess_move_to_index(r.best_move)) {
                check(r.q[a] > 0.9f, "  -> winning move backs up to Q > +0.9 (sign)");
                break;
            }
    }
}

// ---- search WITHOUT the solver overlay: the core Gumbel-AZ machinery alone ----
// No mate exists here, so the proven-mate overlay never fires; the move can only come
// from Gumbel root selection + Sequential Halving + completed-Q descent + averaging
// backup. This isolates the search machinery the mate-in-1 rung would otherwise mask.
static void test_search_value_only(void) {
    printf("[search: value-only (no mate) -> win the hanging queen]\n");
    Position p; chess_from_fen(&p, "r3k2r/ppp2ppp/8/3q4/4P3/8/PPP2PPP/R3K2R w KQkq - 0 1");
    ChessEvaluator ev = chess_oracle_evaluator();
    MctsConfig cfg = mcts_default_config(G1_SIMS);
    cfg.max_considered = G1_CONSIDERED;
    MctsResult r; mcts_search(&p, &ev, &cfg, &r);
    char buf[6];
    check(strcmp(chess_move_to_uci(r.best_move, buf), "e4d5") == 0,
          "exd5 wins the hanging queen by value alone (overlay inactive)");
    check(!delivers_checkmate(&p, r.best_move), "chosen by value, not a mate proof");
}

// ---- value-backup sign (two-sided) + prefer-mate-over-material, on m2d ----
static void test_backup_sign(void) {
    printf("[value-backup sign: prefer mate over winning a queen (m2d)]\n");
    Position p; chess_from_fen(&p, "6k1/5ppp/8/8/8/7q/5PPP/R5K1 w - - 0 1");
    ChessEvaluator ev = chess_oracle_evaluator();
    MctsConfig cfg = mcts_default_config(G1_SIMS);
    cfg.max_considered = G1_CONSIDERED;
    MctsResult r; mcts_search(&p, &ev, &cfg, &r);
    int ra8 = find_move(&r, "a1a8");      // the forced-mate key
    int gxh3 = find_move(&r, "g2h3");     // the queen grab (a winning-material distractor)
    check(ra8 >= 0 && gxh3 >= 0, "both Ra8 and gxh3 are legal at the root");
    if (ra8 >= 0 && gxh3 >= 0) {
        check(r.q[ra8] > 0.0f, "Ra8 (mate) backs up positive (win)");
        check(r.q[gxh3] > 0.0f, "gxh3 (+queen) backs up positive (material)");
        check(r.q[ra8] > r.q[gxh3], "mate Q > material Q (search prefers the mate)");
        check(chess_move_to_index(r.best_move) == chess_move_to_index(r.legal[ra8]),
              "best_move is the mate, not the queen grab");
    }
}

// ---- search: mate-in-2 suite ----
static void test_search_mate_in_2(void) {
    printf("[search: mate-in-2 suite @ %d sims]\n", G1_SIMS);
    for (int i = 0; i < N_SUITE; i++) {
        if (SUITE[i].dist != 2) continue;
        check(solves(&SUITE[i], G1_SIMS, G1_CONSIDERED, MCTS_DEFAULT_SEED), SUITE[i].name);
    }
}

// ---- policy readout: visit + improved policy as 4672-vectors ----
static void test_policy_readout(void) {
    printf("[policy readout: visit + improved 4672-vectors]\n");
    Position p; chess_from_fen(&p, "6k1/5ppp/8/8/8/8/8/R6K w - - 0 1");  // Ra8#
    ChessEvaluator ev = chess_oracle_evaluator();
    MctsConfig cfg = mcts_default_config(G1_SIMS);
    cfg.max_considered = G1_CONSIDERED;
    MctsResult r; mcts_search(&p, &ev, &cfg, &r);

    static float vp[CHESS_POLICY_SIZE], ip[CHESS_POLICY_SIZE];
    mcts_visit_policy(&r, vp);
    mcts_improved_policy(&r, ip);

    // build the legal-index set
    static uint8_t legal_idx[CHESS_POLICY_SIZE];
    memset(legal_idx, 0, sizeof legal_idx);
    for (int a = 0; a < r.n_legal; a++) legal_idx[chess_move_to_index(r.legal[a])] = 1;

    float sv = 0, si = 0; int off_legal = 0;
    for (int i = 0; i < CHESS_POLICY_SIZE; i++) {
        sv += vp[i]; si += ip[i];
        if (!legal_idx[i] && (vp[i] != 0.0f || ip[i] != 0.0f)) off_legal++;
    }
    check(fabsf(sv - 1.0f) < 1e-4f, "visit policy sums to 1 over legal moves");
    check(fabsf(si - 1.0f) < 1e-4f, "improved policy sums to 1 over legal moves");
    check(off_legal == 0, "both policies are zero on illegal indices");
    int mate_idx = chess_move_to_index(r.best_move);
    int vp_arg = 0; for (int i = 1; i < CHESS_POLICY_SIZE; i++) if (vp[i] > vp[vp_arg]) vp_arg = i;
    int ip_arg = 0; for (int i = 1; i < CHESS_POLICY_SIZE; i++) if (ip[i] > ip[ip_arg]) ip_arg = i;
    check(vp_arg == mate_idx, "visit-policy argmax == the mating move");
    check(ip_arg == mate_idx, "improved-policy argmax == the mating move");
}

// ---- G1 robustness: the gate must not be a lucky seed ----
static void test_g1_robustness(void) {
    printf("[G1 robustness: full suite across seeds @ %d sims]\n", G1_SIMS);
    uint64_t seeds[] = {42, 1, 7, 123, 999, 2024, 31337, 777};
    int ns = (int)(sizeof(seeds)/sizeof(seeds[0]));
    int total = 0, pass = 0;
    for (int s = 0; s < ns; s++) {
        int solved = 0;
        for (int i = 0; i < N_SUITE; i++) {
            total++;
            if (solves(&SUITE[i], G1_SIMS, G1_CONSIDERED, seeds[s])) { pass++; solved++; }
        }
        printf("  seed %-10llu  %d/%d\n", (unsigned long long)seeds[s], solved, N_SUITE);
    }
    check(pass == total, "every suite position solved for every seed");
}

// ---- the G1 gate: the whole suite must pass at the stated budget ----
static int test_g1(void) {
    printf("[G1 gate: forced move on mate-in-1/-2 suite @ %d sims, considered=%d, seed=%llu]\n",
           G1_SIMS, G1_CONSIDERED, (unsigned long long)MCTS_DEFAULT_SEED);
    int fails = 0;
    for (int i = 0; i < N_SUITE; i++) {
        int ok = solves(&SUITE[i], G1_SIMS, G1_CONSIDERED, MCTS_DEFAULT_SEED);
        printf("  %-30s %s\n", SUITE[i].name, ok ? "FORCED-MOVE OK" : "*** FAIL ***");
        if (!ok) fails++;
    }
    printf("  G1: %d/%d positions solved -> %s\n", N_SUITE - fails, N_SUITE,
           fails ? "G1 RED" : "G1-GREEN");
    return fails;
}

// =============================================================================
// Batched (vectorized) search == B independent sync searches (build-step 4 / #18)
// =============================================================================
//
// The crux of #18's throughput: mcts_search_batched runs B games in lockstep so their
// leaf evals batch into one ANE forward. Correctness of that lockstep driver is proven
// HERE, on the CPU, against the untouched (G1-green) synchronous mcts_search: with the
// SAME deterministic oracle and SAME per-game seeds, every game's result must be
// bit-for-bit identical. A scheduling/descent divergence would change the visit
// distribution — so exact visit-count equality is the strong assertion. (The net's fp16
// batched-vs-single agreement is a separate, cosine gate in train_selfplay --selfcheck.)
static void test_batched_equivalence(void) {
    printf("[batched search == B independent sync searches (oracle, bit-for-bit)]\n");
    static const char *fens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",       // startpos
        "r3k2r/ppp2ppp/8/3q4/4P3/8/PPP2PPP/R3K2R w KQkq - 0 1",          // value-only (hanging Q)
        "6k1/5ppp/8/8/8/8/8/R6K w - - 0 1",                              // mate-in-1 (solver overlay)
        "6k1/5ppp/8/8/8/7q/5PPP/R5K1 w - - 0 1",                         // mate-in-2 + Q distractor
        "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1",                               // K+P endgame (few moves)
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", // normal midgame
        "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3", // checkmate (terminal root)
        "7k/5Q2/5K2/8/8/8/8/8 b - - 0 1",                                // stalemate (terminal root)
    };
    const int B = (int)(sizeof(fens)/sizeof(fens[0]));
    const int sims = 64;   // small, but exercises several SH phases + the round-robin tail
    Position roots[16]; MctsConfig cfgs[16]; MctsResult sref[16], sbat[16];
    for (int b = 0; b < B; b++) {
        chess_from_fen(&roots[b], fens[b]);
        cfgs[b] = mcts_default_config(sims);
        cfgs[b].max_considered = 16;
        cfgs[b].seed = 42ull + (uint64_t)b;          // each game its own Gumbel noise
    }
    ChessEvaluator ev = chess_oracle_evaluator();
    for (int b = 0; b < B; b++) mcts_search(&roots[b], &ev, &cfgs[b], &sref[b]);   // references
    BatchedChessEvaluator bev = chess_oracle_batched_evaluator();
    mcts_search_batched(roots, B, &bev, cfgs, sbat);                                // under test

    int all_ok = 1;
    for (int b = 0; b < B; b++) {
        MctsResult *a = &sref[b], *c = &sbat[b];
        int ok = (a->best_move == c->best_move) && (a->n_legal == c->n_legal)
              && (a->sims_done == c->sims_done);
        for (int i = 0; i < a->n_legal && ok; i++) {
            if (a->visits[i] != c->visits[i]) ok = 0;        // exact: a divergence shows here
            if (fabsf(a->q[i] - c->q[i]) > 1e-6f) ok = 0;    // same ops, same order -> ~exact
        }
        static float pa[CHESS_POLICY_SIZE], pc[CHESS_POLICY_SIZE];
        mcts_improved_policy(a, pa); mcts_improved_policy(c, pc);
        float maxd = 0; for (int i = 0; i < CHESS_POLICY_SIZE; i++) { float d = fabsf(pa[i]-pc[i]); if (d > maxd) maxd = d; }
        if (maxd > 1e-6f) ok = 0;
        char nm[80]; snprintf(nm, sizeof nm, "  game %d (%d sims): identical to sync", b, a->sims_done);
        check(ok, nm);
        if (!ok) all_ok = 0;
    }
    check(all_ok, "all B games bit-for-bit identical to independent sync search");
}

// ---- measurement: budget sweep (documents the chosen G1 budget) ----
static void sweep(void) {
    int grid[] = {16, 32, 64, 128, 256, 512, 1024};
    int ng = (int)(sizeof(grid)/sizeof(grid[0]));
    printf("[budget sweep] considered=%d seed=%llu  (P = solved, . = not)\n",
           G1_CONSIDERED, (unsigned long long)MCTS_DEFAULT_SEED);
    printf("  %-30s", "position \\ sims");
    for (int g = 0; g < ng; g++) printf("%6d", grid[g]);
    printf("\n");
    for (int i = 0; i < N_SUITE; i++) {
        printf("  %-30s", SUITE[i].name);
        for (int g = 0; g < ng; g++)
            printf("%6s", solves(&SUITE[i], grid[g], G1_CONSIDERED, MCTS_DEFAULT_SEED) ? "P" : ".");
        printf("\n");
    }
}

int main(int argc, char **argv) {
    chess_init();
    int g1_only = (argc > 1 && strcmp(argv[1], "--g1") == 0);
    int do_sweep = (argc > 1 && strcmp(argv[1], "--sweep") == 0);

    if (do_sweep) { sweep(); return 0; }

    if (g1_only) {
        int fails = test_g1();
        return fails ? 1 : 0;
    }

    test_oracle();
    test_seq_halving();
    test_considered_set();
    test_search_mate_in_1();
    test_search_value_only();
    test_backup_sign();
    test_search_mate_in_2();
    test_policy_readout();
    test_batched_equivalence();
    test_g1_robustness();
    printf("\n--- G1 gate ---\n");
    g_fail += test_g1();

    printf("\n%s (%d failure%s)\n", g_fail ? "TESTS FAILED" : "ALL TESTS PASSED",
           g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
