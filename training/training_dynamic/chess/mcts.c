// mcts.c — Gumbel-AlphaZero MCTS (Danihelka et al., 2022). See mcts.h for the contract.
// Pure C, zero external dependencies (math.h/stdlib.h/string.h only). All node-local
// stats are kept from the node's side-to-move perspective; the negamax sign flip on
// backup keeps that invariant, so v_mix / completed-Q / sigma compose consistently.

#include "mcts.h"
#include "chess.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

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

// Expand: legal moves + leaf value/priors. A node with no moves is a terminal (mate =
// proven loss for stm; stalemate = drawn). A non-root node with an immediate mate
// available is a proven win at distance 1, kept un-expanded (treat_proven_mate=1); the
// root is always expanded normally (=0).
static void expand(Node *nd, const ChessEvaluator *ev, int treat_proven_mate) {
    nd->n_children = chess_legal_moves(&nd->pos, nd->moves);
    nd->visits = 0;
    nd->proven = 0; nd->pdist = 0;
    if (nd->n_children == 0) {
        nd->terminal = 1;
        nd->value = chess_terminal_value(&nd->pos);
        if (nd->value < 0.0f) { nd->proven = -1; nd->pdist = 0; }   // checkmated
        return;                                                     // stalemate: drawn (proven=0)
    }
    if (treat_proven_mate && has_mate_in_1(&nd->pos)) {
        nd->terminal = 1;                 // proven win: side to move mates next ply
        nd->value = 1.0f;
        nd->proven = 1; nd->pdist = 1;
        return;
    }
    nd->terminal = 0;
    nd->value = ev->evaluate(ev->ctx, &nd->pos, nd->moves, nd->n_children, nd->P);
    for (int a = 0; a < nd->n_children; a++) { nd->child[a] = -1; nd->W[a] = 0.0f; nd->N[a] = 0; }
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
