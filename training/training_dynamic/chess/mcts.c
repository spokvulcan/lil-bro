// mcts.c — Gumbel-AlphaZero MCTS (Danihelka et al., 2022). See mcts.h for the contract.
// Pure C, zero external dependencies (math.h/stdlib.h/string.h only). All node-local
// stats are kept from the node's side-to-move perspective; the negamax sign flip on
// backup keeps that invariant, so v_mix / completed-Q / sigma compose consistently.

#include "mcts.h"
#include "chess.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static MctsProfile g_prof;
static int g_prof_enabled = 0;

static double prof_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

void mcts_profile_reset(void) {
    memset(&g_prof, 0, sizeof(g_prof));
    g_prof_enabled = 1;
}

MctsProfile mcts_profile_snapshot(void) {
    return g_prof;
}

static inline double prof_start(void) { return g_prof_enabled ? prof_now() : 0.0; }
static inline void prof_add(double *dst, double t0) { if (g_prof_enabled) *dst += prof_now() - t0; }

// =============================================================================
// Evaluator: the G1 stub oracle (material + terminal detection, uniform priors)
// =============================================================================

static const int PIECE_VAL[6] = { 1, 3, 3, 5, 9, 0 };  // P,N,B,R,Q,K (king excluded)

int chess_material_diff(const Position *p) {
    int mat[2] = { 0, 0 };
    for (int c = 0; c < 2; c++)
        for (int pt = PAWN; pt <= QUEEN; pt++)
            mat[c] += __builtin_popcountll(p->bb[c][pt]) * PIECE_VAL[pt];
    return mat[p->side] - mat[p->side ^ 1];
}

float chess_terminal_value(const Position *p) {
    // Caller guarantees no legal moves: checkmate (in check) is a loss for stm, else draw.
    return chess_in_check(p) ? -1.0f : 0.0f;
}

static float oracle_eval(void *ctx, const Position *pos,
                         const Move *legal, int n_legal, float *priors) {
    (void)ctx; (void)legal;
    float u = 1.0f / (float)n_legal;
    for (int i = 0; i < n_legal; i++) priors[i] = u;       // uniform prior (no policy)
    // Pure heuristic value (like the net at #18 will be): compress material so |value|
    // < 0.9, leaving a margin below the proven +-1 the search assigns to forced mates.
    return 0.9f * tanhf((float)chess_material_diff(pos) / 5.0f);
}

ChessEvaluator chess_oracle_evaluator(void) {
    ChessEvaluator e;
    e.ctx = NULL;
    e.evaluate = oracle_eval;
    return e;
}

// =============================================================================
// Deterministic RNG (splitmix64) + Gumbel(0) sampling
// =============================================================================

typedef struct { uint64_t s; } Rng;

static inline uint64_t rng_next(Rng *r) {
    uint64_t z = (r->s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
// Uniform in the OPEN interval (0,1), so -log(-log(u)) is always finite.
static inline double rng_uniform(Rng *r) {
    return ((double)(rng_next(r) >> 11) + 1.0) * (1.0 / 9007199254740994.0);
}
static inline float sample_gumbel(Rng *r) {
    return (float)(-log(-log(rng_uniform(r))));
}

// Fill score[a] = log(prior[a]) + Gumbel(a) for all legal actions, using a fixed seeding
// + iteration order so mcts_considered_set and mcts_search agree on the same noise.
static void gumbel_scores(const float *prior, int n, uint64_t seed, float *score) {
    Rng r; r.s = seed;
    for (int a = 0; a < n; a++)
        score[a] = logf(prior[a] + 1e-9f) + sample_gumbel(&r);
}

// Indices of the top-k entries of score[0..n), written into out[] sorted by score
// descending (simple selection — n is small: <= MAX_MOVES). Returns k (clamped to n).
static int top_k(const float *score, int n, int k, int *out) {
    if (k > n) k = n;
    char taken[MAX_MOVES]; memset(taken, 0, (size_t)n);
    for (int i = 0; i < k; i++) {
        int best = -1; float bv = -INFINITY;
        for (int a = 0; a < n; a++) {
            if (taken[a]) continue;
            if (score[a] > bv) { bv = score[a]; best = a; }
        }
        taken[best] = 1;
        out[i] = best;
    }
    return k;
}

int mcts_considered_set(const float *prior, int n_legal, int m,
                        uint64_t seed, int *considered) {
    int k = (m < n_legal) ? m : n_legal;
    float score[MAX_MOVES];
    gumbel_scores(prior, n_legal, seed, score);
    return top_k(score, n_legal, k, considered);
}

// =============================================================================
// Sequential Halving schedule
// =============================================================================

int mcts_seq_halving(int m, int n, int *size, int *visits, int max_phases) {
    if (m < 1) m = 1;
    // P = ceil(log2(m)) phases (P>=1), using ceil-halving for the candidate-set sizes.
    int P = 0;
    for (int t = m; t > 1; t = (t + 1) / 2) P++;
    if (P < 1) P = 1;
    if (P > max_phases) P = max_phases;
    int per_phase = (n > 0) ? n / P : 0;
    int s = m;
    for (int i = 0; i < P; i++) {
        size[i] = s;
        int v = (s > 0) ? per_phase / s : 0;
        visits[i] = v < 1 ? 1 : v;       // at least one visit per candidate per phase
        s = (s + 1) / 2;                 // halve (ceil) for the next phase
    }
    return P;
}

// =============================================================================
// Search tree
// =============================================================================
//
// One node owns its Position snapshot (so descent never needs unmake) + per-child
// edge statistics. All edge values W[a] are summed from THIS node's side-to-move
// perspective; the negamax flip on backup maintains that invariant.
typedef struct {
    Position pos;
    Move  moves[MAX_MOVES];
    int   child[MAX_MOVES];   // pool index of the child reached by moves[a], or -1
    float P[MAX_MOVES];       // prior over this node's moves
    float W[MAX_MOVES];       // summed backed-up value per child (this node's perspective)
    int   N[MAX_MOVES];       // visit count per child
    int   n_children;         // 0 => terminal
    int   visits;             // sum_a N[a]
    float value;              // evaluator (or terminal) value, this node's perspective
    int   terminal;
    // MCTS-Solver proof (this node's side-to-move perspective): +1 proven win, -1 proven
    // loss, 0 unknown/drawn. pdist = plies to mate when proven (shorter = preferred).
    int   proven, pdist;
} Node;

typedef struct { Node *nodes; int n, cap; } Pool;

static int pool_alloc(Pool *pl) { return pl->n++; }  // cap is preallocated; never reallocs

// MCTS-Solver overlay (Winands et al. 2008): does the side to move have a move that
// delivers immediate checkmate? Engine-derived ground truth, independent of the
// evaluator. A node where this holds is a PROVEN WIN (value +1): there is no reason to
// search it, and DOING so is harmful — averaging backup would let the many inferior
// continuations below a won node drag the forced-mate line's Q negative (the diagnosed
// failure: a mate-in-2 key whose true value is +1 averaging out to -0.5). Treating it
// as a +1 terminal removes exactly that dilution and lets a low-sim search convert
// forced mates the oracle's heuristic value would otherwise miss. Not applied at the
// root (we must expand the root to choose a move).
static int has_mate_in_1(const Position *p) {
    Move mv[MAX_MOVES];
    int n = chess_legal_moves(p, mv);
    for (int i = 0; i < n; i++) {
        Position c = *p;
        Undo u;
        chess_make(&c, mv[i], &u);
        Move r[MAX_MOVES];
        if (chess_legal_moves(&c, r) == 0 && chess_in_check(&c)) return 1;  // checkmate
    }
    return 0;
}

// Expand, engine-only: legal moves + terminal/proven detection, NO evaluator call.
// Returns 1 iff the node is a non-terminal, non-proven leaf that still needs a leaf
// eval (value + priors) — the caller then supplies them (the single ev->evaluate for
// the sync path, or one slot of a batched forward for #18). Returns 0 for terminals
// (mate = proven loss for stm; stalemate = drawn) and for a non-root node with an
// immediate mate (proven win at distance 1, kept un-expanded; treat_proven_mate=1);
// in those cases nd->value is already the game-theoretic value. The root is always
// expanded normally (treat_proven_mate=0). Splitting the engine work out of the eval is
// what lets #18 batch B games' leaf evals into one ANE forward without touching the
// search math — the sync expand() below is unchanged in behavior.
static int expand_engine(Node *nd, int treat_proven_mate) {
    nd->n_children = chess_legal_moves(&nd->pos, nd->moves);
    nd->visits = 0;
    nd->proven = 0; nd->pdist = 0;
    if (nd->n_children == 0) {
        nd->terminal = 1;
        nd->value = chess_terminal_value(&nd->pos);
        if (nd->value < 0.0f) { nd->proven = -1; nd->pdist = 0; }   // checkmated
        return 0;                                                   // stalemate: drawn (proven=0)
    }
    if (treat_proven_mate && has_mate_in_1(&nd->pos)) {
        nd->terminal = 1;                 // proven win: side to move mates next ply
        nd->value = 1.0f;
        nd->proven = 1; nd->pdist = 1;
        return 0;
    }
    nd->terminal = 0;
    for (int a = 0; a < nd->n_children; a++) { nd->child[a] = -1; nd->W[a] = 0.0f; nd->N[a] = 0; }
    return 1;                             // caller fills nd->value + nd->P[0..n_children)
}

// Sync expand = engine work + (if a net leaf) the single-position evaluator. Behavior is
// byte-identical to the pre-#18 expand, so mcts_search / G1 are unchanged.
static void expand(Node *nd, const ChessEvaluator *ev, int treat_proven_mate) {
    if (expand_engine(nd, treat_proven_mate))
        nd->value = ev->evaluate(ev->ctx, &nd->pos, nd->moves, nd->n_children, nd->P);
}

// MCTS-Solver proof propagation: recompute this node's proof from its children's. A
// child C is reached by a move, so C's perspective is the OPPONENT's: C.proven == -1
// (opponent loses) means THIS node can force a win; C.proven == +1 (opponent wins)
// means this move loses. Win if ANY child is a proven loss-for-opponent (shortest
// mate); loss only if ALL children are expanded and proven wins-for-opponent (longest
// resistance). Idempotent; safe to re-run as the tree grows.
static void propagate(Pool *pl, int idx) {
    Node *nd = &pl->nodes[idx];
    if (nd->terminal || nd->proven != 0) return;   // terminals/decided nodes are fixed
    int best_win = -1, worst_loss = -1, all_children_lost = 1;
    for (int a = 0; a < nd->n_children; a++) {
        int ci = nd->child[a];
        if (ci < 0) { all_children_lost = 0; continue; }   // unexpanded -> can't prove a loss
        int cp = pl->nodes[ci].proven, cd = pl->nodes[ci].pdist;
        if (cp == -1) { if (best_win < 0 || cd < best_win) best_win = cd; }   // we can win here
        else { all_children_lost = 0; if (cp == 1 && cd > worst_loss) worst_loss = cd; }
    }
    if (best_win >= 0) { nd->proven = 1;  nd->pdist = best_win + 1; }
    else if (all_children_lost) { nd->proven = -1; nd->pdist = worst_loss + 1; }
}

// Deterministic completed-Q non-root action selection (Danihelka 2022): build the
// improved policy pi'(a) = softmax(log P(a) + sigma(completedQ(a))) where unvisited
// actions take the mixed value v_mix, then pick argmax_a [ pi'(a) - N(a)/(1+sum_b N) ].
static int nonroot_select(const Node *nd, float c_visit, float c_scale) {
    int n = nd->n_children, sumN = nd->visits, maxN = 0;
    float sum_pi_vis = 0.0f, wq = 0.0f;
    for (int a = 0; a < n; a++) {
        if (nd->N[a] > 0) {
            sum_pi_vis += nd->P[a];
            wq += nd->P[a] * (nd->W[a] / (float)nd->N[a]);
            if (nd->N[a] > maxN) maxN = nd->N[a];
        }
    }
    float v_mix = (sum_pi_vis > 0.0f)
        ? (nd->value + (float)sumN * (wq / sum_pi_vis)) / (1.0f + (float)sumN)
        : nd->value;
    float sigma_scale = (c_visit + (float)maxN) * c_scale;
    float il[MAX_MOVES], mx = -INFINITY;
    for (int a = 0; a < n; a++) {
        float q = (nd->N[a] > 0) ? nd->W[a] / (float)nd->N[a] : v_mix;
        il[a] = logf(nd->P[a] + 1e-9f) + sigma_scale * q;
        if (il[a] > mx) mx = il[a];
    }
    float Z = 0.0f, pi[MAX_MOVES];
    for (int a = 0; a < n; a++) { pi[a] = expf(il[a] - mx); Z += pi[a]; }
    float invZ = 1.0f / Z;
    int best = 0; float bv = -INFINITY;
    for (int a = 0; a < n; a++) {
        float s = pi[a] * invZ - (float)nd->N[a] / (1.0f + (float)sumN);
        if (s > bv) { bv = s; best = a; }
    }
    return best;
}

// One simulation: from the root take the SH-forced action, then descend by non-root
// selection to a new/terminal leaf, evaluate it, and back the value up with a per-ply
// sign flip (negamax) and a gamma discount. path_node/path_act are scratch (>= height+1).
static void simulate(Pool *pl, int root_idx, int forced_action, const ChessEvaluator *ev,
                     float c_visit, float c_scale, float gamma, int *path_node, int *path_act) {
    int depth = 0, cur = root_idx, a = forced_action;
    float v_leaf;
    for (;;) {
        path_node[depth] = cur; path_act[depth] = a; depth++;
        Node *nd = &pl->nodes[cur];
        int cidx = nd->child[a];
        if (cidx < 0) {                                  // create + expand the leaf
            int ni = pool_alloc(pl);
            Node *child = &pl->nodes[ni];
            child->pos = nd->pos;
            Undo u; chess_make(&child->pos, nd->moves[a], &u);
            expand(child, ev, 1);            // non-root: proven-mate overlay on
            nd->child[a] = ni;
            v_leaf = child->value;
            break;
        }
        Node *child = &pl->nodes[cidx];
        if (child->terminal) { v_leaf = child->value; break; }
        cur = cidx;
        a = nonroot_select(child, c_visit, c_scale);
    }
    float v = v_leaf;
    for (int d = depth - 1; d >= 0; d--) {
        v = -gamma * v;                                  // flip + discount one ply
        Node *nd = &pl->nodes[path_node[d]];
        int act = path_act[d];
        nd->W[act] += v; nd->N[act] += 1; nd->visits += 1;
    }
    // Propagate solver proofs bottom-up: a freshly-proven leaf can prove its ancestors.
    for (int d = depth - 1; d >= 0; d--) propagate(pl, path_node[d]);
}

// Root score of action a. A PROVEN win outranks everything (shorter mate first); a
// PROVEN loss ranks below everything; otherwise the Gumbel score g(a)+log P(a)+sigma(q).
// The proof terms make forced mates seed-independent: once proven, the move is locked.
static float root_score(const Pool *pl, const Node *root, int a,
                        const float *score0, float sigma_scale) {
    int ci = root->child[a];
    if (ci >= 0) {
        int cp = pl->nodes[ci].proven, cd = pl->nodes[ci].pdist;
        if (cp == -1) return 1.0e6f - (float)(cd + 1);   // opponent loses => we win
        if (cp == +1) return -1.0e6f + score0[a];        // opponent wins => this move loses
    }
    float q = (root->N[a] > 0) ? root->W[a] / (float)root->N[a] : 0.0f;
    return score0[a] + sigma_scale * q;                  // score0[a] = log P(a) + g(a)
}

// Sort cand[0..ncand) by root_score descending (selection sort; ncand is small).
static void rank_candidates(const Pool *pl, const Node *root, int *cand, int ncand,
                            const float *score0, float c_visit, float c_scale) {
    int maxN = 0;
    for (int a = 0; a < root->n_children; a++) if (root->N[a] > maxN) maxN = root->N[a];
    float sigma_scale = (c_visit + (float)maxN) * c_scale;
    for (int i = 0; i < ncand; i++) {
        int best = i; float bv = root_score(pl, root, cand[i], score0, sigma_scale);
        for (int j = i + 1; j < ncand; j++) {
            float s = root_score(pl, root, cand[j], score0, sigma_scale);
            if (s > bv) { bv = s; best = j; }
        }
        int t = cand[i]; cand[i] = cand[best]; cand[best] = t;
    }
}

MctsConfig mcts_default_config(int num_simulations) {
    MctsConfig c;
    c.num_simulations = num_simulations;
    c.max_considered  = 16;
    c.c_visit = MCTS_C_VISIT;
    c.c_scale = MCTS_C_SCALE;
    c.gamma   = 1.0f;          // pure Gumbel-AZ; the solver proof handles mate distance
    c.root_dirichlet_alpha = 0.0f;  // OFF by default => G1 / equivalence gates bit-identical
    c.root_dirichlet_frac  = 0.0f;
    c.seed    = MCTS_DEFAULT_SEED;
    return c;
}

void mcts_search(const Position *root_pos, const ChessEvaluator *ev,
                 const MctsConfig *cfg, MctsResult *out) {
    memset(out, 0, sizeof(*out));
    out->c_visit = cfg->c_visit; out->c_scale = cfg->c_scale;
    out->best_move = MOVE_NONE;

    int N = cfg->num_simulations < 1 ? 1 : cfg->num_simulations;
    int cap = N + 2;
    Pool pl;
    pl.nodes = (Node*)malloc((size_t)cap * sizeof(Node));
    pl.n = 0; pl.cap = cap;
    int *path_node = (int*)malloc((size_t)cap * sizeof(int));
    int *path_act  = (int*)malloc((size_t)cap * sizeof(int));

    int ridx = pool_alloc(&pl);
    Node *root = &pl.nodes[ridx];
    root->pos = *root_pos;
    expand(root, ev, 0);                      // root: always expanded to choose a move
    out->root_value = root->value;
    out->n_legal = root->n_children;
    if (root->terminal) { out->nodes_used = pl.n; goto done; }

    int nleg = root->n_children;
    for (int a = 0; a < nleg; a++) { out->legal[a] = root->moves[a]; out->prior[a] = root->P[a]; }

    // Gumbel scores over ALL legal actions (same seeding/order as mcts_considered_set).
    float score0[MAX_MOVES];
    gumbel_scores(root->P, nleg, cfg->seed, score0);

    int m = cfg->max_considered; if (m > nleg) m = nleg; if (m < 1) m = 1;
    int cand[MAX_MOVES];
    int ncand = top_k(score0, nleg, m, cand);   // initial considered set (top-m by log P + g)

    int size[MCTS_MAX_PHASES], vis[MCTS_MAX_PHASES];
    int P = mcts_seq_halving(m, N, size, vis, MCTS_MAX_PHASES);

    int sims = 0;
    for (int ph = 0; ph < P && sims < N; ph++) {
        for (int r = 0; r < vis[ph] && sims < N; r++) {
            for (int i = 0; i < ncand && sims < N; i++) {
                simulate(&pl, ridx, cand[i], ev, cfg->c_visit, cfg->c_scale, cfg->gamma, path_node, path_act);
                sims++;
            }
        }
        rank_candidates(&pl, root, cand, ncand, score0, cfg->c_visit, cfg->c_scale);
        int keep = (ph + 1 < P) ? size[ph + 1] : 1;     // halve toward the single winner
        if (keep < 1) keep = 1; if (keep > ncand) keep = ncand;
        ncand = keep;
    }
    // Spend any remaining budget round-robin over the survivors -> EXACTLY N sims.
    while (sims < N) {
        for (int i = 0; i < ncand && sims < N; i++) {
            simulate(&pl, ridx, cand[i], ev, cfg->c_visit, cfg->c_scale, cfg->gamma, path_node, path_act);
            sims++;
        }
    }
    out->sims_done = sims;

    for (int a = 0; a < nleg; a++) {
        out->visits[a] = root->N[a];
        out->q[a] = root->N[a] > 0 ? root->W[a] / (float)root->N[a] : 0.0f;
    }
    rank_candidates(&pl, root, cand, ncand, score0, cfg->c_visit, cfg->c_scale);
    out->best_move = root->moves[cand[0]];
    out->nodes_used = pl.n;

done:
    free(pl.nodes); free(path_node); free(path_act);
}

// =============================================================================
// Improved-policy readouts (dense 4672-vectors over chess_move_to_index)
// =============================================================================

void mcts_visit_policy(const MctsResult *r, float out[CHESS_POLICY_SIZE]) {
    memset(out, 0, (size_t)CHESS_POLICY_SIZE * sizeof(float));
    long total = 0;
    for (int a = 0; a < r->n_legal; a++) total += r->visits[a];
    if (total <= 0) return;
    float inv = 1.0f / (float)total;
    for (int a = 0; a < r->n_legal; a++)
        out[chess_move_to_index(r->legal[a])] = (float)r->visits[a] * inv;
}

void mcts_improved_policy(const MctsResult *r, float out[CHESS_POLICY_SIZE]) {
    memset(out, 0, (size_t)CHESS_POLICY_SIZE * sizeof(float));
    int n = r->n_legal;
    if (n <= 0) return;
    int sumN = 0, maxN = 0;
    float sum_pi_vis = 0.0f, wq = 0.0f;
    for (int a = 0; a < n; a++) {
        sumN += r->visits[a];
        if (r->visits[a] > 0) {
            sum_pi_vis += r->prior[a];
            wq += r->prior[a] * r->q[a];
            if (r->visits[a] > maxN) maxN = r->visits[a];
        }
    }
    float v_mix = (sum_pi_vis > 0.0f)
        ? (r->root_value + (float)sumN * (wq / sum_pi_vis)) / (1.0f + (float)sumN)
        : r->root_value;
    float sigma_scale = (r->c_visit + (float)maxN) * r->c_scale;
    float il[MAX_MOVES], mx = -INFINITY;
    for (int a = 0; a < n; a++) {
        float q = (r->visits[a] > 0) ? r->q[a] : v_mix;
        il[a] = logf(r->prior[a] + 1e-9f) + sigma_scale * q;
        if (il[a] > mx) mx = il[a];
    }
    float Z = 0.0f, pi[MAX_MOVES];
    for (int a = 0; a < n; a++) { pi[a] = expf(il[a] - mx); Z += pi[a]; }
    float invZ = 1.0f / Z;
    for (int a = 0; a < n; a++)
        out[chess_move_to_index(r->legal[a])] = pi[a] * invZ;
}

// =============================================================================
// Vectorized batched-leaf search (build-step 4 / issue #18). See mcts.h.
//
// The driver runs B independent Gumbel-AZ searches as a steppable state machine, in
// lockstep, so every game's pending leaf eval (the root eval + each simulation's fresh
// leaf) is buffered and flushed as ONE batched forward. It calls the SAME static
// machinery mcts_search uses — expand_engine / nonroot_select / propagate /
// rank_candidates / gumbel_scores / top_k / mcts_seq_halving — so with a deterministic
// evaluator each game's outcome is bit-identical to mcts_search (test_mcts --batched).
// Only the sim *scheduling* and tree *descent* are re-expressed as resume points; that
// re-expression is exactly what the equivalence gate cross-checks against the untouched
// synchronous search above.
// =============================================================================

// The G1 oracle as a batched evaluator (per-position loop over the single oracle).
static void oracle_eval_batched(void *ctx, const Position *const *pos, int B,
                                const Move *const *legal, const int *n_legal,
                                float *const *priors, float *value) {
    (void)ctx; (void)legal;
    for (int b = 0; b < B; b++) {
        float u = 1.0f / (float)n_legal[b];
        for (int i = 0; i < n_legal[b]; i++) priors[b][i] = u;
        value[b] = 0.9f * tanhf((float)chess_material_diff(pos[b]) / 5.0f);
    }
}
BatchedChessEvaluator chess_oracle_batched_evaluator(void) {
    BatchedChessEvaluator e; e.ctx = NULL; e.evaluate = oracle_eval_batched; return e;
}

// Per-game resumable search state. Mirrors mcts_search's locals; the (ph,r,i,tail_i,sims)
// counters + phase_end_pending make its triple-loop + round-robin tail re-enterable one
// simulation at a time.
enum { CTX_ROOT, CTX_RUN, CTX_DONE };
typedef struct {
    Pool       pl;
    int       *path_node, *path_act;        // descent scratch (cap-long)
    int        ridx;                        // root node index
    MctsConfig cfg;
    int        N;                           // sim budget (>=1)
    float      score0[MAX_MOVES];           // log P(a) + Gumbel(a) over root legal actions
    int        cand[MAX_MOVES], ncand;      // current considered set
    int        size[MCTS_MAX_PHASES], vis[MCTS_MAX_PHASES], P;  // Sequential-Halving schedule
    int        ph, r, i, tail_i, sims;      // loop position
    int        phase_end_pending;           // owe a rank+shrink before the next action
    int        pending_depth, pending_leaf; // leaf awaiting a batched eval
    int        state;
    MctsResult *out;
} SearchCtx;

// Descend from the root via forced_action, then completed-Q non-root selection, to a
// leaf — exactly simulate()'s descent. Returns 1 if the leaf is a fresh non-terminal
// node needing a net eval (its idx in *leaf_idx, path saved in s); else sets *v_leaf
// (terminal/proven/existing-terminal) and returns 0. s->pending_depth records the path
// length so ctx_backup can run after the (possibly deferred) eval.
static int ctx_descend(SearchCtx *s, int forced_action, float *v_leaf, int *leaf_idx) {
    Pool *pl = &s->pl;
    int depth = 0, cur = s->ridx, a = forced_action;
    for (;;) {
        s->path_node[depth] = cur; s->path_act[depth] = a; depth++;
        Node *nd = &pl->nodes[cur];
        int cidx = nd->child[a];
        if (cidx < 0) {                                  // create + expand a new leaf
            int ni = pool_alloc(pl);
            Node *child = &pl->nodes[ni];
            child->pos = nd->pos;
            Undo u; chess_make(&child->pos, nd->moves[a], &u);
            nd->child[a] = ni;
            s->pending_depth = depth; *leaf_idx = ni;
            if (expand_engine(child, 1)) return 1;       // non-terminal: needs net (value+P)
            *v_leaf = child->value; return 0;            // terminal/proven: value already set
        }
        Node *child = &pl->nodes[cidx];
        if (child->terminal) { s->pending_depth = depth; *leaf_idx = cidx; *v_leaf = child->value; return 0; }
        cur = cidx; a = nonroot_select(child, s->cfg.c_visit, s->cfg.c_scale);
    }
}

// Negamax value backup + solver-proof propagation (== simulate()'s tail).
static void ctx_backup(SearchCtx *s, int depth, float v_leaf) {
    Pool *pl = &s->pl;
    float v = v_leaf;
    for (int d = depth - 1; d >= 0; d--) {
        v = -s->cfg.gamma * v;
        Node *nd = &pl->nodes[s->path_node[d]];
        int act = s->path_act[d];
        nd->W[act] += v; nd->N[act] += 1; nd->visits += 1;
    }
    for (int d = depth - 1; d >= 0; d--) propagate(pl, s->path_node[d]);
}

// The Sequential-Halving scheduler as a generator: returns the next forced ROOT action,
// or -1 when all N sims have been dispatched. Performs the per-phase rank+shrink lazily
// (after the phase's last sim has been backed up), exactly mirroring mcts_search's loop.
static int ctx_next_action(SearchCtx *s) {
    Pool *pl = &s->pl; Node *root = &pl->nodes[s->ridx];
    if (s->phase_end_pending) {
        rank_candidates(pl, root, s->cand, s->ncand, s->score0, s->cfg.c_visit, s->cfg.c_scale);
        int keep = (s->ph + 1 < s->P) ? s->size[s->ph + 1] : 1;
        if (keep < 1) keep = 1; if (keep > s->ncand) keep = s->ncand;
        s->ncand = keep;
        s->ph++; s->r = 0; s->i = 0;
        s->phase_end_pending = 0;
    }
    if (s->sims >= s->N) return -1;
    if (s->ph < s->P) {
        int a = s->cand[s->i];
        s->i++; s->sims++;
        if (s->i >= s->ncand) {
            s->i = 0; s->r++;
            if (s->r >= s->vis[s->ph]) s->phase_end_pending = 1;   // phase complete
        }
        return a;
    }
    int a = s->cand[s->tail_i];                          // round-robin tail over survivors
    s->tail_i++; if (s->tail_i >= s->ncand) s->tail_i = 0;
    s->sims++;
    return a;
}

// Gamma(alpha,1) via Marsaglia–Tsang (alpha<1 handled by the boost identity), using the
// search RNG. Used only for the Purist-Zero root Dirichlet noise (off by default).
static double rng_normal(Rng *r) {
    double u1 = rng_uniform(r), u2 = rng_uniform(r);
    return sqrt(-2.0*log(u1)) * cos(2.0*3.14159265358979323846*u2);
}
static double sample_gamma(Rng *r, double alpha) {
    if (alpha < 1.0) { double u = rng_uniform(r); return sample_gamma(r, alpha + 1.0) * pow(u, 1.0/alpha); }
    double d = alpha - 1.0/3.0, c = 1.0/sqrt(9.0*d);
    for (;;) {
        double x = rng_normal(r), v = 1.0 + c*x;
        if (v <= 0.0) continue;
        v = v*v*v; double u = rng_uniform(r);
        if (u < 1.0 - 0.0331*x*x*x*x) return d*v;
        if (log(u) < 0.5*x*x + d*(1.0 - v + log(v))) return d*v;
    }
}
// Purist-Zero root exploration: P(a) <- (1-frac)*P(a) + frac*Dir(alpha) over the legal
// actions (ADR 0005 decision 8). A distinct RNG stream from the Gumbel noise so the two
// don't correlate. No-op when alpha or frac is 0 (the default).
static void apply_root_dirichlet(float *P, int n, float alpha, float frac, uint64_t seed) {
    if (alpha <= 0.0f || frac <= 0.0f || n <= 0) return;
    Rng r; r.s = seed ^ 0xD1C5B17A1ull;
    double g[MAX_MOVES], sum = 0.0;
    for (int a = 0; a < n; a++) { g[a] = sample_gamma(&r, (double)alpha); sum += g[a]; }
    if (sum <= 0.0) return;
    for (int a = 0; a < n; a++) { float noise = (float)(g[a]/sum); P[a] = (1.0f-frac)*P[a] + frac*noise; }
}

// After the root has been evaluated (value + priors filled): set up Gumbel scores, the
// considered set, and the SH schedule — exactly mcts_search's pre-loop block.
static void ctx_setup_schedule(SearchCtx *s) {
    Node *root = &s->pl.nodes[s->ridx];
    int nleg = root->n_children;
    apply_root_dirichlet(root->P, nleg, s->cfg.root_dirichlet_alpha, s->cfg.root_dirichlet_frac, s->cfg.seed);
    s->out->root_value = root->value;
    for (int a = 0; a < nleg; a++) { s->out->legal[a] = root->moves[a]; s->out->prior[a] = root->P[a]; }
    gumbel_scores(root->P, nleg, s->cfg.seed, s->score0);
    int m = s->cfg.max_considered; if (m > nleg) m = nleg; if (m < 1) m = 1;
    s->ncand = top_k(s->score0, nleg, m, s->cand);
    s->P = mcts_seq_halving(m, s->N, s->size, s->vis, MCTS_MAX_PHASES);
    s->ph = 0; s->r = 0; s->i = 0; s->tail_i = 0; s->sims = 0; s->phase_end_pending = 0;
    s->state = CTX_RUN;
}

// Read out best_move + per-move stats — exactly mcts_search's post-loop block.
static void ctx_finalize(SearchCtx *s) {
    Node *root = &s->pl.nodes[s->ridx];
    int nleg = root->n_children;
    s->out->sims_done = s->sims;
    for (int a = 0; a < nleg; a++) {
        s->out->visits[a] = root->N[a];
        s->out->q[a] = root->N[a] > 0 ? root->W[a] / (float)root->N[a] : 0.0f;
    }
    rank_candidates(&s->pl, root, s->cand, s->ncand, s->score0, s->cfg.c_visit, s->cfg.c_scale);
    s->out->best_move = root->moves[s->cand[0]];
    s->out->nodes_used = s->pl.n;
    s->state = CTX_DONE;
}

void mcts_search_batched(const Position *roots, int B, const BatchedChessEvaluator *bev,
                         const MctsConfig *cfgs, MctsResult *outs) {
    if (B < 1) return;
    if (g_prof_enabled) g_prof.searches += B;
    double t_alloc = prof_start();
    SearchCtx *S = (SearchCtx*)calloc((size_t)B, sizeof(SearchCtx));
    // Batch scratch (one slot per game; the sim loop uses <= B slots).
    const Position **pos = (const Position**)malloc((size_t)B * sizeof(*pos));
    const Move    **leg  = (const Move**)   malloc((size_t)B * sizeof(*leg));
    int            *nleg = (int*)   malloc((size_t)B * sizeof(int));
    float         **pri  = (float**)malloc((size_t)B * sizeof(*pri));
    float          *val  = (float*) malloc((size_t)B * sizeof(float));
    int            *gidx = (int*)   malloc((size_t)B * sizeof(int));   // batch slot -> game
    prof_add(&g_prof.alloc_s, t_alloc);

    // ---- init each game; expand the root (engine part only) ----
    for (int b = 0; b < B; b++) {
        SearchCtx *s = &S[b];
        s->cfg = cfgs[b];
        s->N = s->cfg.num_simulations < 1 ? 1 : s->cfg.num_simulations;
        int cap = s->N + 2;
        t_alloc = prof_start();
        s->pl.nodes = (Node*)malloc((size_t)cap * sizeof(Node));
        s->pl.n = 0; s->pl.cap = cap;
        s->path_node = (int*)malloc((size_t)cap * sizeof(int));
        s->path_act  = (int*)malloc((size_t)cap * sizeof(int));
        prof_add(&g_prof.alloc_s, t_alloc);
        s->out = &outs[b];
        memset(s->out, 0, sizeof(MctsResult));
        s->out->c_visit = s->cfg.c_visit; s->out->c_scale = s->cfg.c_scale;
        s->out->best_move = MOVE_NONE;
        s->ridx = pool_alloc(&s->pl);
        Node *root = &s->pl.nodes[s->ridx];
        root->pos = roots[b];
        double t_root = prof_start();
        expand_engine(root, 0);                  // root: always expanded normally
        prof_add(&g_prof.root_expand_s, t_root);
        s->out->n_legal = root->n_children;
        if (root->n_children == 0) {             // terminal root: nothing to search
            s->out->root_value = root->value;
            s->out->nodes_used = s->pl.n;
            s->state = CTX_DONE;
        } else {
            s->state = CTX_ROOT;                 // root needs a net eval (value + priors)
            s->pending_leaf = s->ridx;
        }
    }

    // ---- batch the ROOT evals, then set up each game's schedule ----
    int ncoll = 0;
    for (int b = 0; b < B; b++) {
        if (S[b].state != CTX_ROOT) continue;
        Node *root = &S[b].pl.nodes[S[b].ridx];
        pos[ncoll] = &root->pos; leg[ncoll] = root->moves;
        nleg[ncoll] = root->n_children; pri[ncoll] = root->P; gidx[ncoll] = b; ncoll++;
    }
    if (ncoll > 0) {
        if (g_prof_enabled) { g_prof.eval_calls++; g_prof.eval_positions += ncoll; }
        double t_eval = prof_start();
        bev->evaluate(bev->ctx, pos, ncoll, leg, nleg, pri, val);
        prof_add(&g_prof.root_eval_s, t_eval);
        for (int k = 0; k < ncoll; k++) {
            SearchCtx *s = &S[gidx[k]];
            s->pl.nodes[s->ridx].value = val[k];
            ctx_setup_schedule(s);
        }
    }

    // ---- lockstep simulation loop: each tick advances every active game by one net
    //      eval (burning through any terminal-leaf sims for free), then flushes the
    //      collected leaves as ONE batched forward. ----
    for (;;) {
        ncoll = 0;
        double t_cpu = prof_start();
        for (int b = 0; b < B; b++) {
            SearchCtx *s = &S[b];
            if (s->state != CTX_RUN) continue;
            for (;;) {
                int a = ctx_next_action(s);
                if (a < 0) { ctx_finalize(s); break; }          // game complete
                float v; int leaf;
                if (ctx_descend(s, a, &v, &leaf)) {             // fresh net leaf -> batch it
                    s->pending_leaf = leaf;
                    Node *ln = &s->pl.nodes[leaf];
                    pos[ncoll] = &ln->pos; leg[ncoll] = ln->moves;
                    nleg[ncoll] = ln->n_children; pri[ncoll] = ln->P; gidx[ncoll] = b; ncoll++;
                    break;                                      // wait for the eval
                }
                ctx_backup(s, s->pending_depth, v);             // terminal leaf: free, keep going
            }
        }
        prof_add(&g_prof.sim_cpu_s, t_cpu);
        if (ncoll == 0) break;                                   // all games done
        if (g_prof_enabled) { g_prof.eval_calls++; g_prof.eval_positions += ncoll; }
        double t_eval = prof_start();
        bev->evaluate(bev->ctx, pos, ncoll, leg, nleg, pri, val);
        prof_add(&g_prof.leaf_eval_s, t_eval);
        for (int k = 0; k < ncoll; k++) {
            SearchCtx *s = &S[gidx[k]];
            s->pl.nodes[s->pending_leaf].value = val[k];
            ctx_backup(s, s->pending_depth, val[k]);
        }
    }

    if (g_prof_enabled) {
        for (int b = 0; b < B; b++) {
            g_prof.sims += S[b].out->sims_done;
            g_prof.nodes += S[b].out->nodes_used;
        }
    }
    for (int b = 0; b < B; b++) { free(S[b].pl.nodes); free(S[b].path_node); free(S[b].path_act); }
    free(S); free(pos); free(leg); free(nleg); free(pri); free(val); free(gidx);
}
