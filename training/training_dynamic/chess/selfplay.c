// selfplay.c — self-play generation + the fixed-opponent eval ladder (issue #18).
// Evaluator-agnostic (drives a BatchedChessEvaluator): the oracle in test_selfplay.c,
// the ANE net in train_selfplay.m. See selfplay.h for the design rationale.
#include "selfplay.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifndef OPTIMIZER_IS_MUON
#define OPTIMIZER_IS_MUON 1
#endif

// ============================================================================
// Config
// ============================================================================
SPConfig sp_defaults(void) {
    SPConfig c;
    c.B = 64; c.sims = 32; c.considered = 16;
    c.dir_alpha = 0.3f; c.dir_frac = 0.25f;
    c.temp = 1.0f; c.temp_moves = 15; c.max_plies = 100;
    c.use_improved_policy = 1; c.curriculum = 0; c.curriculum_plies = 8; c.adjudicate = 0;
    c.warmup_iters = 20; c.warmup_frac = 1.0f;   // cold-start value-prior warmup: ON by default (dec 8 fallback)
    c.td_lambda = 1.0f;
    c.replay_cap = 30000; c.learner_batch = 64; c.learner_steps = 16; c.iters = 60;
    c.optimizer_muon = OPTIMIZER_IS_MUON;
    c.lr = 2e-3f; c.loss_scale = 256.0f; c.grad_clip = 1.0f; c.wd = 0.0f; c.value_weight = 1.0f;
    c.eval_games = 40; c.eval_every = 5; c.eval_sims = 32; c.eval_considered = 16; c.eval_max_plies = 120;
    c.elo_every = 0; c.elo_games = 32;
    c.bench_games = 256;
    c.profile = 0;
    c.use_mps = 0;
    c.use_mps_graph = 0;
    c.seed = 42; c.ckpt = "ane_chess_g2_ckpt.bin"; c.resume = 0;
    return c;
}

#define ARGF(name,field) else if (!strcmp(argv[i], name) && i+1<argc) c.field = (float)atof(argv[++i])
#define ARGI(name,field) else if (!strcmp(argv[i], name) && i+1<argc) c.field = atoi(argv[++i])
SPConfig sp_parse(int argc, char **argv, int *mode) {
    SPConfig c = sp_defaults(); if (mode) *mode = 0;  // 0 smoke, 1 g2, 2 selfcheck, 3 bench, 4 elo
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--g2"))        { if (mode) *mode = 1; }
        else if (!strcmp(argv[i], "--selfcheck")) { if (mode) *mode = 2; }
        else if (!strcmp(argv[i], "--bench"))     { if (mode) *mode = 3; }
        else if (!strcmp(argv[i], "--elo"))       { if (mode) *mode = 4; }
        else if (!strcmp(argv[i], "--resume"))     c.resume = 1;
        else if (!strcmp(argv[i], "--curriculum")) c.curriculum = 1;
        else if (!strcmp(argv[i], "--adjudicate")) c.adjudicate = 1;
        else if (!strcmp(argv[i], "--no-adjudicate")) c.adjudicate = 0;
        else if (!strcmp(argv[i], "--no-warmup"))  { c.warmup_iters = 0; c.warmup_frac = 0.0f; }
        else if (!strcmp(argv[i], "--visit-policy")) c.use_improved_policy = 0;
        else if (!strcmp(argv[i], "--ckpt") && i+1<argc) c.ckpt = argv[++i];
        else if (!strcmp(argv[i], "--seed") && i+1<argc) c.seed = (uint64_t)strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--opt") && i+1<argc) {
            const char *o = argv[++i];
            if (!strcmp(o, "muon")) c.optimizer_muon = 1;
            else if (!strcmp(o, "adamw")) c.optimizer_muon = 0;
            else {
                fprintf(stderr, "[config] unknown --opt %s (use adamw|muon); keeping %s\n",
                        o, c.optimizer_muon ? "muon" : "adamw");
            }
        }
        ARGI("--B", B); ARGI("--sims", sims); ARGI("--considered", considered);
        ARGI("--replay", replay_cap); ARGI("--lbatch", learner_batch);
        ARGI("--lsteps", learner_steps); ARGI("--iters", iters);
        ARGF("--lr", lr); ARGF("--loss-scale", loss_scale); ARGF("--clip", grad_clip);
        ARGF("--wd", wd); ARGF("--vw", value_weight);
        ARGF("--dir-alpha", dir_alpha); ARGF("--dir-frac", dir_frac);
        ARGF("--temp", temp); ARGI("--temp-moves", temp_moves); ARGI("--max-plies", max_plies);
        ARGI("--curriculum-plies", curriculum_plies);
        ARGI("--warmup-iters", warmup_iters); ARGF("--warmup-frac", warmup_frac);
        ARGF("--td-lambda", td_lambda);
        ARGI("--eval-games", eval_games); ARGI("--eval-every", eval_every);
        ARGI("--eval-sims", eval_sims); ARGI("--eval-considered", eval_considered);
        ARGI("--eval-max-plies", eval_max_plies);
        ARGI("--elo-every", elo_every); ARGI("--elo-games", elo_games);
        ARGI("--bench-games", bench_games);
        else if (!strcmp(argv[i], "--profile")) c.profile = 1;
        else if (!strcmp(argv[i], "--mps"))     c.use_mps = 1;
        else if (!strcmp(argv[i], "--mps-graph")) c.use_mps_graph = 1;
    }
    if (c.B > 160) c.B = 160;   // round32(B)*96 must stay <= 16384; max packed B is 160
    if (c.bench_games < 1) c.bench_games = 1;
    if (c.td_lambda < 0.0f || c.td_lambda > 1.0f) {
        float raw = c.td_lambda;
        c.td_lambda = raw < 0.0f ? 0.0f : 1.0f;
        fprintf(stderr, "[config] --td-lambda %g out of [0,1]; clamped to %g\n", (double)raw, (double)c.td_lambda);
    }
    return c;
}

// ============================================================================
// Move selection + sample construction
// ============================================================================
Move select_move(const MctsResult *r, int ply, const SPConfig *cfg, uint64_t *rng) {
    if (r->n_legal <= 0) return MOVE_NONE;
    if (ply >= cfg->temp_moves || cfg->temp <= 0.0f) return r->best_move;
    double w[MAX_MOVES], sum = 0.0, invt = 1.0/(double)cfg->temp;
    for (int a = 0; a < r->n_legal; a++) { w[a] = pow((double)r->visits[a], invt); sum += w[a]; }
    if (sum <= 0.0) return r->best_move;                 // no visits (shouldn't happen) -> greedy
    double x = sm_uniform(rng) * sum, acc = 0.0;
    for (int a = 0; a < r->n_legal; a++) { acc += w[a]; if (x <= acc) return r->legal[a]; }
    return r->legal[r->n_legal - 1];
}

void build_sample(ReplaySample *s, const Position *pos, const MctsResult *r,
                  const SPConfig *cfg, float *dense_scratch) {
    int16_t t16[CHESS_NUM_TOKENS]; chess_encode(pos, t16);
    for (int t = 0; t < CHESS_NUM_TOKENS; t++) s->tokens[t] = (uint16_t)t16[t];
    for (int t = CHESS_NUM_TOKENS; t < CHESS_PAD_TOKENS; t++) s->tokens[t] = TOK_EMPTY;
    if (cfg->use_improved_policy) mcts_improved_policy(r, dense_scratch);
    else                         mcts_visit_policy(r, dense_scratch);
    s->n_policy = r->n_legal;
    for (int a = 0; a < r->n_legal; a++) {
        int idx = chess_move_to_index(r->legal[a]);
        s->policy_idx[a] = idx; s->policy_p[a] = dense_scratch[idx];
    }
    s->z = 0.0f;   // filled at game end
    s->z_nstep = 0.0f;
}

void relabel_value_targets(ReplaySample *plies, const float *leaf_v, int n_plies,
                           const int *side, float fv, int fstm, float td_lambda) {
    if (n_plies <= 0) return;
    float lam = td_lambda;
    if (lam < 0.0f) lam = 0.0f;
    if (lam > 1.0f) lam = 1.0f;
    int last = n_plies - 1;
    for (int t = 0; t < n_plies; t++) {
        float z_t = (side[t] == fstm) ? fv : -fv;
        float g = 0.0f, lam_pow = 1.0f;
        for (int n = 1; n <= last - t; n++) {
            int tp = t + n;
            float b = (side[tp] == side[t]) ? leaf_v[tp] : -leaf_v[tp];
            g += (1.0f - lam) * lam_pow * b;
            lam_pow *= lam;
        }
        g += lam_pow * z_t;
        if (g > 1.0f) g = 1.0f;
        if (g < -1.0f) g = -1.0f;
        plies[t].z_nstep = g;
    }
}

// ============================================================================
// Fixed opponents
// ============================================================================
Move opp_random(const Position *p, uint64_t *rng) {
    Move mv[MAX_MOVES]; int n = chess_legal_moves(p, mv);
    return n ? mv[sm_below(rng, n)] : MOVE_NONE;
}
// 1-ply material-greedy: maximize my material after the move (+ take any mate-in-1), random
// tiebreak. A weak but non-trivial baseline (never hangs free material to a 1-ply look).
Move opp_greedy(const Position *p, uint64_t *rng) {
    Move mv[MAX_MOVES]; int n = chess_legal_moves(p, mv);
    if (n == 0) return MOVE_NONE;
    int best = -100000000; Move bestmv[MAX_MOVES]; int nb = 0;
    for (int i = 0; i < n; i++) {
        Position c = *p; Undo u; chess_make(&c, mv[i], &u);
        Move tmp[MAX_MOVES]; int nl = chess_legal_moves(&c, tmp);
        int score = -chess_material_diff(&c);                 // my edge (c is opponent-to-move)
        if (nl == 0 && chess_in_check(&c)) score += 100000;   // checkmate
        if (score > best) { best = score; nb = 0; bestmv[nb++] = mv[i]; }
        else if (score == best) bestmv[nb++] = mv[i];
    }
    return bestmv[sm_below(rng, nb)];
}

// ============================================================================
// GENERATION
// ============================================================================
typedef struct { Position cur; int done; float result_value; int result_stm; int n_plies;
                 int *side; float *leaf_v; ReplaySample *plies; } Game;

#define SP_ADJ_THRESH 2   // pawns of material edge that adjudicate a capped game as a win

// Value of a game that ended WITHOUT mate/stalemate (ply cap or 50-move). Without
// adjudication this is a flat draw (0). With adjudication (cold-start mitigation) a clear
// material edge (>= SP_ADJ_THRESH pawns) counts as a win for the side that has it, from the
// side-to-move's perspective — giving the value head a signal when weak self-play never
// reaches mate. The eval ladder never calls this (it scores real outcomes only).
static float adjudicate_value(const Position *p, int adjudicate) {
    if (!adjudicate) return 0.0f;
    int md = chess_material_diff(p);                  // side-to-move material - opponent (pawns)
    if (md >=  SP_ADJ_THRESH) return  1.0f;
    if (md <= -SP_ADJ_THRESH) return -1.0f;
    return 0.0f;
}

static uint64_t hash_mix64(uint64_t h, uint64_t x) {
    h ^= x + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
    return h;
}

static uint64_t hash_sample(uint64_t h, const ReplaySample *s) {
    if (h == 0) h = 1469598103934665603ull;
    for (int t = 0; t < CHESS_PAD_TOKENS; t++) h = hash_mix64(h, s->tokens[t]);
    h = hash_mix64(h, (uint64_t)s->n_policy);
    for (int e = 0; e < s->n_policy; e++) {
        uint32_t pb; memcpy(&pb, &s->policy_p[e], sizeof(pb));
        h = hash_mix64(h, (uint64_t)(uint32_t)s->policy_idx[e]);
        h = hash_mix64(h, pb);
    }
    uint32_t zb; memcpy(&zb, &s->z, sizeof(zb));
    h = hash_mix64(h, zb);
    uint32_t zb2; memcpy(&zb2, &s->z_nstep, sizeof(zb2));
    return hash_mix64(h, zb2);
}

void play_selfplay_batch(const BatchedChessEvaluator *bev, const BatchedChessEvaluator *label_bev,
                         ReplayBuffer *rb, const SPConfig *cfg, uint64_t base_seed, GenStats *st) {
    int B = cfg->B;
    Game *g = (Game*)calloc(B, sizeof(Game));
    float *label_pri = label_bev ? (float*)malloc((size_t)B * MAX_MOVES * sizeof(float)) : NULL;
    float *label_v_buf = label_bev ? (float*)malloc((size_t)B * sizeof(float)) : NULL;
    for (int b = 0; b < B; b++) {
        chess_startpos(&g[b].cur);
        g[b].plies = (ReplaySample*)malloc((size_t)cfg->max_plies * sizeof(ReplaySample));
        g[b].side  = (int*)malloc((size_t)cfg->max_plies * sizeof(int));
        g[b].leaf_v = (float*)malloc((size_t)cfg->max_plies * sizeof(float));
        g[b].done = 0; g[b].n_plies = 0;
    }
    // Optional curriculum (default OFF): start from a random short opening to shorten the
    // cold-start desert (ADR 0005 decision 8: a MEASURED fallback, not on by default).
    if (cfg->curriculum) {
        for (int b = 0; b < B; b++) {
            uint64_t cr = base_seed ^ (0xC0FFEEull + (uint64_t)b*2654435761u);
            for (int k = 0; k < cfg->curriculum_plies; k++) {
                Move mv[MAX_MOVES]; int n = chess_legal_moves(&g[b].cur, mv);
                if (n == 0) { chess_startpos(&g[b].cur); break; }
                Undo u; chess_make(&g[b].cur, mv[sm_below(&cr, n)], &u);
            }
        }
    }
    float *dense = (float*)malloc((size_t)CHESS_POLICY_SIZE * sizeof(float));
    uint64_t mv_rng = base_seed ^ 0x5E1EC7ull;
    Position   *roots  = (Position*)malloc((size_t)B * sizeof(Position));
    MctsConfig *cfgs   = (MctsConfig*)malloc((size_t)B * sizeof(MctsConfig));
    int        *active = (int*)malloc((size_t)B * sizeof(int));
    MctsResult *res    = (MctsResult*)malloc((size_t)B * sizeof(MctsResult));

    for (int ply = 0; ply < cfg->max_plies; ply++) {
        int na = 0;
        for (int b = 0; b < B; b++) {
            if (g[b].done) continue;
            roots[na] = g[b].cur;
            cfgs[na] = mcts_default_config(cfg->sims);
            cfgs[na].max_considered = cfg->considered;
            cfgs[na].root_dirichlet_alpha = cfg->dir_alpha;
            cfgs[na].root_dirichlet_frac  = cfg->dir_frac;
            cfgs[na].seed = base_seed + (uint64_t)ply*100003u*(uint64_t)B + (uint64_t)b*100019u + 1;
            active[na] = b; na++;
        }
        if (na == 0) break;
        mcts_search_batched(roots, na, bev, cfgs, res);
        if (st) {
            for (int k = 0; k < na; k++) {
                st->sims += res[k].sims_done;
                st->nodes += res[k].nodes_used;
            }
        }
        if (label_bev) {
            const Position *lpos[na]; const Move *lleg[na]; int lnleg[na]; float *lpri[na];
            for (int k = 0; k < na; k++) {
                lpos[k] = &roots[k]; lleg[k] = res[k].legal; lnleg[k] = res[k].n_legal;
                lpri[k] = &label_pri[(size_t)k * MAX_MOVES];
            }
            label_bev->evaluate(label_bev->ctx, lpos, na, lleg, lnleg, lpri, label_v_buf);
        }
        for (int k = 0; k < na; k++) {
            int b = active[k];
            // record the searched position as a training sample (z filled at game end)
            ReplaySample *smp = &g[b].plies[g[b].n_plies];
            build_sample(smp, &g[b].cur, &res[k], cfg, dense);
            g[b].side[g[b].n_plies] = g[b].cur.side;
            g[b].leaf_v[g[b].n_plies] = label_bev ? label_v_buf[k] : res[k].root_value;
            g[b].n_plies++;
            // select + apply a move. Search is never run on a terminal node (the seam
            // contract guarantees n_legal>=1), so select_move returns a real move; guard
            // defensively so a future search change can't feed MOVE_NONE into chess_make.
            Move mv = select_move(&res[k], ply, cfg, &mv_rng);
            if (mv == MOVE_NONE) { g[b].done = 1; g[b].result_stm = g[b].cur.side;
                                   g[b].result_value = chess_terminal_value(&g[b].cur); continue; }
            Undo u; chess_make(&g[b].cur, mv, &u);
            // terminal (mate/stalemate) or 50-move / length-cap draw?
            Move tmp[MAX_MOVES]; int nl = chess_legal_moves(&g[b].cur, tmp);
            if (nl == 0) {
                g[b].done = 1; g[b].result_stm = g[b].cur.side;
                g[b].result_value = chess_terminal_value(&g[b].cur);   // -1 mate (stm lost), 0 stalemate
            } else if (g[b].cur.halfmove >= 100) {
                g[b].done = 1; g[b].result_stm = g[b].cur.side;
                g[b].result_value = adjudicate_value(&g[b].cur, cfg->adjudicate);  // 50-move
            } else if (g[b].n_plies >= cfg->max_plies) {
                g[b].done = 1; g[b].result_stm = g[b].cur.side;
                g[b].result_value = adjudicate_value(&g[b].cur, cfg->adjudicate);  // ply cap
            }
        }
    }
    // label z per recorded ply (from THAT ply's side-to-move) and push to replay
    for (int b = 0; b < B; b++) {
        if (!g[b].done) { g[b].done = 1; g[b].result_stm = g[b].cur.side;
                          g[b].result_value = adjudicate_value(&g[b].cur, cfg->adjudicate); }
        float fv = g[b].result_value; int fstm = g[b].result_stm;
        if (st) {
            // winner = final-stm if fv>0 (adjudicated material win), else the OTHER side
            // (fv<0: mate/adjudicated loss). fv==0 is a draw.
            if (fv == 0.0f) st->draws++;
            else { int white_won = (fv > 0.0f) ? (fstm == WHITE) : (fstm != WHITE);
                   if (white_won) st->wins_w++; else st->wins_b++; }
            st->games++;
        }
        relabel_value_targets(g[b].plies, g[b].leaf_v, g[b].n_plies, g[b].side, fv, fstm, cfg->td_lambda);
        for (int p = 0; p < g[b].n_plies; p++) {
            int s = g[b].side[p];
            g[b].plies[p].z = (s == fstm) ? fv : -fv;
            replay_add(rb, &g[b].plies[p]);
            if (st) { st->plies++; st->checksum = hash_sample(st->checksum, &g[b].plies[p]); }
        }
        free(g[b].plies); free(g[b].side); free(g[b].leaf_v);
    }
    free(g); free(dense); free(roots); free(cfgs); free(active); free(res);
    free(label_pri); free(label_v_buf);
}

// ============================================================================
// EVAL: net vs a fixed opponent (net alternates colors), n_games in lockstep.
// ============================================================================
typedef struct { Position cur; int done; int net_white; int result; } EvalGame;  // result: +1 net win, -1 loss, 0 draw

// Resolve a just-reached position as terminal if it has no legal moves / hits the 50-move
// clock. Returns 1 and sets *result (net perspective) if the game is over, else 0.
static int eval_terminal(EvalGame *e) {
    Move tmp[MAX_MOVES]; int nl = chess_legal_moves(&e->cur, tmp);
    if (nl == 0) {
        if (chess_in_check(&e->cur)) {                       // checkmate: side-to-move lost
            int loser_white = (e->cur.side == WHITE);
            e->result = (loser_white == e->net_white) ? -1 : +1;
        } else e->result = 0;                                // stalemate
        return 1;
    }
    if (e->cur.halfmove >= 100) { e->result = 0; return 1; } // 50-move draw
    return 0;
}

double eval_vs_opponent(const BatchedChessEvaluator *bev, const SPConfig *cfg,
                        OpponentFn opp, int n_games, uint64_t seed, int *W, int *D, int *Lo) {
    EvalGame *g = (EvalGame*)calloc(n_games, sizeof(EvalGame));
    for (int i = 0; i < n_games; i++) { chess_startpos(&g[i].cur); g[i].done = 0; g[i].net_white = (i % 2 == 0); g[i].result = 0; }
    uint64_t orng = seed ^ 0xA11CE5ull;
    Position   *roots = (Position*)malloc((size_t)n_games*sizeof(Position));
    MctsConfig *cfgs  = (MctsConfig*)malloc((size_t)n_games*sizeof(MctsConfig));
    int        *idx   = (int*)malloc((size_t)n_games*sizeof(int));
    MctsResult *res   = (MctsResult*)malloc((size_t)n_games*sizeof(MctsResult));

    for (int ply = 0; ply < cfg->eval_max_plies; ply++) {
        // NET-to-move games -> one batched search (deterministic: no Dirichlet/temperature)
        int na = 0;
        for (int i = 0; i < n_games; i++) {
            if (g[i].done) continue;
            int net_to_move = ((g[i].cur.side == WHITE) == g[i].net_white);
            if (!net_to_move) continue;
            roots[na] = g[i].cur;
            cfgs[na] = mcts_default_config(cfg->eval_sims);
            cfgs[na].max_considered = cfg->eval_considered;
            cfgs[na].seed = seed + (uint64_t)ply*100003u*(uint64_t)n_games + (uint64_t)i*100019u + 7;
            idx[na] = i; na++;
        }
        if (na > 0) {
            mcts_search_batched(roots, na, bev, cfgs, res);
            for (int k = 0; k < na; k++) {
                int i = idx[k];
                if (res[k].best_move == MOVE_NONE) { g[i].done = eval_terminal(&g[i]); continue; }
                Undo u; chess_make(&g[i].cur, res[k].best_move, &u);
                g[i].done = eval_terminal(&g[i]);
            }
        }
        // OPPONENT-to-move games (CPU; no batching needed)
        for (int i = 0; i < n_games; i++) {
            if (g[i].done) continue;
            int net_to_move = ((g[i].cur.side == WHITE) == g[i].net_white);
            if (net_to_move) continue;
            Move m = opp(&g[i].cur, &orng);
            if (m == MOVE_NONE) { g[i].done = eval_terminal(&g[i]); continue; }
            Undo u; chess_make(&g[i].cur, m, &u);
            g[i].done = eval_terminal(&g[i]);
        }
        int any = 0; for (int i = 0; i < n_games; i++) if (!g[i].done) { any = 1; break; }
        if (!any) break;
    }
    int w = 0, d = 0, l = 0;
    for (int i = 0; i < n_games; i++) { if (g[i].result > 0) w++; else if (g[i].result < 0) l++; else d++; }
    if (W) *W = w; if (D) *D = d; if (Lo) *Lo = l;
    free(g); free(roots); free(cfgs); free(idx); free(res);
    return ((double)w + 0.5*(double)d) / (double)n_games;
}

// ============================================================================
// MATCH: net A vs net B (self-anchored Elo, ADR 0007). See selfplay.h for the contract.
// Two evaluators; at each ply every active game has exactly one side to move, so we snapshot
// each game's mover BEFORE applying any move this ply (else a game just moved by A would be
// re-selected for B in the same ply and double-move), then run one batched search per side.
// ============================================================================
typedef struct { Position cur; int done; int a_white; int result; } MatchGame; // result: +1 A win, -1 A loss, 0 draw

static int match_terminal(MatchGame *e) {
    Move tmp[MAX_MOVES]; int nl = chess_legal_moves(&e->cur, tmp);
    if (nl == 0) {
        if (chess_in_check(&e->cur)) {                        // side-to-move is checkmated -> loses
            int loser_white = (e->cur.side == WHITE);
            e->result = (loser_white == e->a_white) ? -1 : +1;
        } else e->result = 0;                                 // stalemate
        return 1;
    }
    if (e->cur.halfmove >= 100) { e->result = 0; return 1; }  // 50-move draw
    return 0;
}

double match_net_vs_net(const BatchedChessEvaluator *bevA, const BatchedChessEvaluator *bevB,
                        const SPConfig *cfg, int n_games, int open_plies, uint64_t seed,
                        int *Wa, int *Da, int *La) {
    MatchGame *g = (MatchGame*)calloc(n_games, sizeof(MatchGame));
    for (int i = 0; i < n_games; i++) {
        chess_startpos(&g[i].cur);
        g[i].a_white = (i % 2 == 0);   // color-swapped pairs: games 2k / 2k+1 share an opening
        g[i].result = 0; g[i].done = 0;
        // distinct random opening per color-swap PAIR (k=i/2): both halves replay the same
        // uniform moves from the same seed -> identical opening, swapped colors (bias cancels).
        uint64_t orng = seed ^ (0x09E3779B1ull + (uint64_t)(i / 2) * 2654435761u);
        for (int k = 0; k < open_plies; k++) {
            Move mv[MAX_MOVES]; int n = chess_legal_moves(&g[i].cur, mv);
            if (n == 0) { chess_startpos(&g[i].cur); break; }
            Undo u; chess_make(&g[i].cur, mv[sm_below(&orng, n)], &u);
        }
        g[i].done = match_terminal(&g[i]);   // a random opening could already be terminal
    }
    Position   *roots = (Position*)malloc((size_t)n_games*sizeof(Position));
    MctsConfig *cfgs  = (MctsConfig*)malloc((size_t)n_games*sizeof(MctsConfig));
    int        *idx   = (int*)malloc((size_t)n_games*sizeof(int));
    int        *mover = (int*)malloc((size_t)n_games*sizeof(int));
    MctsResult *res   = (MctsResult*)malloc((size_t)n_games*sizeof(MctsResult));

    for (int ply = 0; ply < cfg->eval_max_plies; ply++) {
        // snapshot whose turn it is this ply (0=A's evaluator, 1=B's, -1=done) BEFORE any move
        for (int i = 0; i < n_games; i++) {
            if (g[i].done) { mover[i] = -1; continue; }
            mover[i] = ((g[i].cur.side == WHITE) == g[i].a_white) ? 0 : 1;
        }
        for (int side = 0; side < 2; side++) {
            const BatchedChessEvaluator *bev = (side == 0) ? bevA : bevB;
            int na = 0;
            for (int i = 0; i < n_games; i++) {
                if (mover[i] != side) continue;
                roots[na] = g[i].cur;
                cfgs[na] = mcts_default_config(cfg->eval_sims);
                cfgs[na].max_considered = cfg->eval_considered;
                cfgs[na].seed = seed + (uint64_t)ply*100003u*(uint64_t)n_games
                                     + (uint64_t)i*100019u + (uint64_t)side*13u + 7u;
                idx[na] = i; na++;
            }
            if (na == 0) continue;
            mcts_search_batched(roots, na, bev, cfgs, res);
            for (int k = 0; k < na; k++) {
                int i = idx[k];
                if (res[k].best_move == MOVE_NONE) { g[i].done = match_terminal(&g[i]); continue; }
                Undo u; chess_make(&g[i].cur, res[k].best_move, &u);
                g[i].done = match_terminal(&g[i]);
            }
        }
        int any = 0; for (int i = 0; i < n_games; i++) if (!g[i].done) { any = 1; break; }
        if (!any) break;
    }
    int w = 0, d = 0, l = 0;
    for (int i = 0; i < n_games; i++) { if (g[i].result > 0) w++; else if (g[i].result < 0) l++; else d++; }
    if (Wa) *Wa = w; if (Da) *Da = d; if (La) *La = l;
    free(g); free(roots); free(cfgs); free(idx); free(mover); free(res);
    return ((double)w + 0.5*(double)d) / (double)n_games;
}

// ============================================================================
// Warmup value-prior wrapper (ADR 0005 decision 8 fallback, MEASURED-triggered).
// See selfplay.h for the contract. Blends the inner evaluator's leaf VALUE with a
// material heuristic (the G1 oracle's value: 0.9*tanh(material_diff/5)), leaving the
// priors untouched. A search prior, not labels. The wrapper owns a small context (the
// inner pointer + frac); warmup_evaluator_free releases it.
// ============================================================================
typedef struct { const BatchedChessEvaluator *inner; float frac; } WarmupCtx;

static void warmup_evaluate(void *ctx, const Position *const *pos, int B,
                            const Move *const *legal, const int *n_legal,
                            float *const *priors, float *value) {
    WarmupCtx *w = (WarmupCtx*)ctx;
    w->inner->evaluate(w->inner->ctx, pos, B, legal, n_legal, priors, value);
    if (w->frac <= 0.0f) return;                  // frac=0: pure net (the steady state)
    float f = w->frac;
    for (int b = 0; b < B; b++) {
        float mat_v = 0.9f * tanhf((float)chess_material_diff(pos[b]) / 5.0f);
        value[b] = (1.0f - f) * value[b] + f * mat_v;
    }
}

BatchedChessEvaluator make_warmup_evaluator(const BatchedChessEvaluator *inner, float frac) {
    BatchedChessEvaluator out;
    WarmupCtx *w = (WarmupCtx*)malloc(sizeof(WarmupCtx));
    w->inner = inner; w->frac = frac;
    out.ctx = w; out.evaluate = warmup_evaluate;
    return out;
}

void warmup_evaluator_free(BatchedChessEvaluator *w) {
    if (w && w->ctx) { free(w->ctx); w->ctx = NULL; }
}
