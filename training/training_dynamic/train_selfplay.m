// train_selfplay.m — Build-step 4 of chess-RL-on-ANE (ADR 0005, issue #18): the full
// single-process, ONE-ANE-CLIENT self-play RL loop, and the G2 gate (the loop learns:
// win-rate climbs vs a fixed random-mover, then a weak fixed baseline).
//
// THE LOOP (all on one ANE client, time-shared by generation + learner — no second
// client; the AC and [[ane-resident-training-cpu-floor]] forbid it):
//   1. GENERATION — B games stepped in lockstep (selfplay.c play_selfplay_batch): every
//      game's MCTS leaf evals (root + per-sim) collapse into ONE batched ANE forward
//      (seq=B*SEQ; results/chess_throughput_probe.md). Purist-Zero cold start (random init,
//      Dirichlet root noise + move-temperature, no external labels — ADR 0005 decision 8).
//      Each searched position -> a replay sample: policy <- the (improved) MCTS visit
//      distribution, value <- the game outcome z (decision 13).
//   2. REPLAY — a sliding window of recent positions (decision 14: always-latest).
//   3. LEARNER — samples a minibatch and trains the shared trunk (chess_net.h) with the
//      AlphaZero loss + fp16 loss-scaling + grad-clip — the SAME net `make g0` verified.
//
// THE SPLIT: the net glue lives HERE (the batched forward behind mcts.h's evaluator
// contract — decision 2; the learner; the substrate self-check). The evaluator-AGNOSTIC
// orchestration (generation, eval ladder, z-labeling) lives in chess/selfplay.c and is
// oracle-tested without the ANE (chess/test_selfplay.c). "The thing you train is the thing
// you proved": B=1 here is byte-identical to G0's trunk; B>1 is checked vs B singles below.
//
// MEASURED GATE (the discipline): --g2 runs the loop and plays the net vs a random-mover
// and a 1-ply material-greedy baseline at checkpoints, printing the win-rate CURVE. G2 is
// green when the curve climbs and the net beats both opponents — never asserted from loss.
//
// Build: make train_selfplay   ·   Gate: make g2   ·   Self-check: ./train_selfplay --selfcheck

#include "mil_dynamic.h"          // pulls io.h + config.h (DIM/SEQ/... from chess_g0.h)
#include "cpu_ops.h"              // rmsnorm(_bwd), attn_cpu_*, adam_update, embed_*
#include "chess/chess.h"          // engine + codec (#15)
#include "chess/chess_heads.h"    // policy/value loss + posenc (#16)
#include "chess/chess_net.h"      // shared trunk: ANE matmul, batched fwd/bwd, readout, ckpt, optimizer
#include "chess/mcts.h"           // Gumbel-MCTS + the batched lockstep driver (#17/#18)
#include "chess/replay.h"         // sliding-window replay buffer
#include "chess/selfplay.h"       // generation + eval ladder + config (evaluator-agnostic)
#include <stdio.h>
#include <string.h>

static float g_res_alpha;   // 1/sqrt(2*NLAYERS), set in main

// ---- learner-internal sub-profiler (zero-overhead when g_learner_prof==0) ----
static int    g_learner_prof = 0;
static double g_lfwd_s = 0.0, g_lloss_s = 0.0, g_lbwd_s = 0.0, g_lemb_s = 0.0, g_lopt_s = 0.0, g_lfus_s = 0.0;

static inline int roundup32(int x) { return ((x + 31) / 32) * 32; }

// ============================================================================
// The net as a batched leaf evaluator (the decision-2 seam: #16 net behind mcts.h's
// ChessEvaluator contract, vectorized for #18). One seq=PAD*SEQ ANE forward over the
// (bucket-padded) batch, then per-position legal-masked policy softmax + WDL->W-L value.
// ============================================================================
typedef struct {
    ChessNet *net;
    int       maxB;            // scratch capacity in positions (a multiple of 32)
    uint16_t *tokens;          // [maxB*SEQ]
    float    *x_in, *x_pre, *x_final;   // [DIM*maxB*SEQ]
    CActs     acts[1];         // forward-only (save_acts=0)
    int       prof_enabled;
    long      prof_calls, prof_positions;
    double    prof_encode_s, prof_embed_s, prof_trunk_s, prof_readout_s;
} NetEvalCtx;

static void net_eval_ctx_alloc(NetEvalCtx *e, ChessNet *net, int maxB) {
    memset(e, 0, sizeof(*e));
    maxB = roundup32(maxB);
    e->net = net; e->maxB = maxB;
    e->tokens  = (uint16_t*)malloc((size_t)maxB*SEQ*sizeof(uint16_t));
    e->x_in    = fmalloc((size_t)DIM*maxB*SEQ);
    e->x_pre   = fmalloc((size_t)DIM*maxB*SEQ);
    e->x_final = fmalloc((size_t)DIM*maxB*SEQ);
    cacts_alloc(&e->acts[0], maxB*SEQ);
}

static void net_eval_profile_reset(NetEvalCtx *e) {
    e->prof_enabled = 1;
    e->prof_calls = e->prof_positions = 0;
    e->prof_encode_s = e->prof_embed_s = e->prof_trunk_s = e->prof_readout_s = 0.0;
    g_trunk_io_s = 0.0; g_trunk_ane_s = 0.0; g_trunk_attn_s = 0.0;
    g_trunk_rms_s = 0.0; g_trunk_silu_s = 0.0; g_trunk_softmax_s = 0.0; g_trunk_prof = 0;
}

static double mt_s(uint64_t t) { return tb_ms(t) * 1e-3; }

// Batched evaluator: B positions -> one ANE forward at packed width P (B rounded up to a
// multiple of 32, so seq=P*SEQ is always mult-of-32 AND only a handful of kernels compile
// as B shrinks when games finish). Pad slots are TOK_EMPTY; only [0,B) are read back.
static void net_eval_batched(void *ctx, const Position *const *pos, int B,
                             const Move *const *legal, const int *n_legal,
                             float *const *priors, float *value) {
    NetEvalCtx *e = (NetEvalCtx*)ctx;
    if (e->prof_enabled) { e->prof_calls++; e->prof_positions += B; }
    int P = roundup32(B); if (P > e->maxB) P = e->maxB;   // (maxB is mult-of-32 and >= B)
    uint64_t t0 = e->prof_enabled ? mach_absolute_time() : 0;
    for (int b = 0; b < P; b++) {
        uint16_t *tk = e->tokens + (size_t)b*SEQ;
        if (b < B) {
            int16_t t16[CHESS_NUM_TOKENS]; chess_encode(pos[b], t16);
            for (int t = 0; t < CHESS_NUM_TOKENS; t++) tk[t] = (uint16_t)t16[t];
            for (int t = CHESS_NUM_TOKENS; t < SEQ; t++) tk[t] = TOK_EMPTY;
        } else {
            for (int t = 0; t < SEQ; t++) tk[t] = TOK_EMPTY;
        }
    }
    if (e->prof_enabled) { e->prof_encode_s += mt_s(mach_absolute_time() - t0); t0 = mach_absolute_time(); }
    chess_embed_posenc_batched(e->x_in, P, e->tokens, e->net->tok_emb, e->net->rank_emb, e->net->file_emb, e->net->misc_emb);
    if (e->prof_enabled) { e->prof_embed_s += mt_s(mach_absolute_time() - t0); t0 = mach_absolute_time(); }
    chess_trunk_forward(e->net->W, e->acts, e->x_in, P, e->x_pre, e->x_final, e->net->rms_final, g_res_alpha, 0);
    if (e->prof_enabled) { e->prof_trunk_s += mt_s(mach_absolute_time() - t0); t0 = mach_absolute_time(); }
    // Readout is per-position independent (policy legal-masked softmax + WDL value); parallelize
    // over B. This is a SERIAL->PARALLEL win (full wall benefit, not /12 like the already-
    // parallel trunk sections). Each b writes value[b]/priors[b] (disjoint), reads x_final[b].
    // Bit-identical per b => checksum preserved. [iter 8]
    {
        int Bb = B; float *xfin = e->x_final; int Ps = P*SEQ;
        const float *Wp = e->net->W_pol, *Wv = e->net->W_val;
        const Move *const *lg = legal; const int *nl = n_legal; float *const *pr = priors; float *val = value;
        dispatch_queue_t dq = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
        dispatch_apply(Bb, dq, ^(size_t bb) {
            int b = (int)bb;
            val[b] = chess_policy_value_readout(xfin + (size_t)b*SEQ, Ps, Wp, Wv, lg[b], nl[b], pr[b]);
        });
    }
    if (e->prof_enabled) e->prof_readout_s += mt_s(mach_absolute_time() - t0);
}

// ============================================================================
// LEARNER: one minibatch step over the replay window (batched fwd+bwd, AZ loss).
// ============================================================================
typedef struct {
    ChessNet *net, *grads;
    CActs     acts[NLAYERS];                 // save_acts=1 (kept for backward), width K*SEQ
    float    *x_in, *x_pre, *x_final, *dx_final, *dy_in, *dxv;
    uint16_t *tokens;
    int       K;
} Learner;

static void learner_alloc(Learner *L, ChessNet *net, ChessNet *grads, int K) {
    L->net = net; L->grads = grads; L->K = K;
    for (int i = 0; i < NLAYERS; i++) cacts_alloc(&L->acts[i], K*SEQ);
    L->x_in=fmalloc((size_t)DIM*K*SEQ); L->x_pre=fmalloc((size_t)DIM*K*SEQ); L->x_final=fmalloc((size_t)DIM*K*SEQ);
    L->dx_final=fmalloc((size_t)DIM*K*SEQ); L->dy_in=fmalloc((size_t)DIM*K*SEQ); L->dxv=fmalloc((size_t)DIM*K*SEQ);
    L->tokens=(uint16_t*)malloc((size_t)K*SEQ*sizeof(uint16_t));
}

// Run one optimizer step on a uniform minibatch; reports mean policy/value CE via out.
static void learner_step(Learner *L, ReplayBuffer *rb, const SPConfig *cfg, int adam_t,
                         float *out_lp, float *out_lv) {
    int K = L->K, S = K*SEQ;
    uint64_t _t = g_learner_prof ? mach_absolute_time() : 0;
    const ReplaySample **batch = (const ReplaySample**)malloc((size_t)K*sizeof(*batch));
    replay_sample_batch(rb, batch, K);
    for (int k = 0; k < K; k++) memcpy(L->tokens + (size_t)k*SEQ, batch[k]->tokens, SEQ*sizeof(uint16_t));

    grads_zero();
    chess_embed_posenc_batched(L->x_in, K, L->tokens, L->net->tok_emb, L->net->rank_emb, L->net->file_emb, L->net->misc_emb);
    if (g_use_mps_graph && g_mtl_dev) {
        mg_ad_forward(K, L->x_in, L->net->W, L->net->rms_final, L->x_final);
    } else {
        chess_trunk_forward(L->net->W, L->acts, L->x_in, K, L->x_pre, L->x_final, L->net->rms_final, g_res_alpha, 1);
    }
    if (g_learner_prof) g_lfwd_s += mt_s(mach_absolute_time() - _t);

    _t = g_learner_prof ? mach_absolute_time() : 0;
    memset(L->dx_final, 0, (size_t)DIM*S*4);
    memset(L->dxv,      0, (size_t)DIM*S*4);
    double lp = 0, lv = 0;
    static uint8_t mask[POL]; static float tgt_pol[POL];
    for (int k = 0; k < K; k++) {
        // reconstruct the legal mask from the decoded position (mask can't drift from tokens)
        int16_t t16[CHESS_NUM_TOKENS]; for (int t = 0; t < CHESS_NUM_TOKENS; t++) t16[t] = (int16_t)batch[k]->tokens[t];
        Position p; chess_decode(&p, t16);
        Move legal[MAX_MOVES]; int nl = chess_legal_moves(&p, legal);
        chess_legal_mask(legal, nl, mask);
        // dense policy target from the sparse sample
        memset(tgt_pol, 0, sizeof tgt_pol);
        for (int e = 0; e < batch[k]->n_policy; e++) tgt_pol[batch[k]->policy_idx[e]] = batch[k]->policy_p[e];
        float v = batch[k]->z_nstep, tgt_val[NWDL];
        float tw = v > 0.0f ? v : 0.0f;
        float tl = v < 0.0f ? -v : 0.0f;
        tgt_val[0] = tw;
        tgt_val[2] = tl;
        tgt_val[1] = 1.0f - tw - tl;
        lp += chess_policy_loss(L->x_final + (size_t)k*SEQ, L->net->W_pol, DIM, S, NBOARD, PLANES,
                                mask, tgt_pol, L->dx_final + (size_t)k*SEQ, L->grads->W_pol);
        lv += chess_value_loss(L->x_final + (size_t)k*SEQ, L->net->W_val, DIM, S, NREAL, NWDL,
                               tgt_val, L->dxv + (size_t)k*SEQ, L->grads->W_val);
    }
    if (g_learner_prof) g_lloss_s += mt_s(mach_absolute_time() - _t);
    // total loss = policy + value_weight*value: blend value's dx + scale its head grad.
    float vw = cfg->value_weight;
    for (int i = 0; i < DIM*S; i++) L->dx_final[i] += vw * L->dxv[i];
    for (int i = 0; i < DIM*NWDL; i++) L->grads->W_val[i] *= vw;
    // fp16 loss-scaling: scale the grad entering the ANE backward by the FULL loss_scale
    // (underflow headroom — the same trick G0 uses at K=1), then the optimizer unscales by
    // 1/(loss_scale*K) so the AdamW step sees the MINIBATCH MEAN gradient.
    float ls = cfg->loss_scale;
    vDSP_vsmul(L->dx_final,     1, &ls, L->dx_final,     1, (vDSP_Length)(DIM*S));
    vDSP_vsmul(L->grads->W_pol, 1, &ls, L->grads->W_pol, 1, (vDSP_Length)(DIM*PLANES));
    vDSP_vsmul(L->grads->W_val, 1, &ls, L->grads->W_val, 1, (vDSP_Length)(DIM*NWDL));
    _t = g_learner_prof ? mach_absolute_time() : 0;
    if (g_use_mps_graph && g_mtl_dev) {
        mg_ad_backward(K, L->x_in, L->net->W, L->net->rms_final, L->dx_final, L->grads->W, L->grads->rms_final, L->dy_in);
    } else {
        chess_trunk_backward(L->net->W, L->grads->W, L->acts, L->dx_final, K, L->x_pre, L->net->rms_final, L->grads->rms_final, L->dy_in, g_res_alpha);
    }
    if (g_learner_prof) g_lbwd_s += mt_s(mach_absolute_time() - _t);
    _t = g_learner_prof ? mach_absolute_time() : 0;
    chess_embed_posenc_backward_batched(L->dy_in, K, L->grads->tok_emb, L->grads->rank_emb, L->grads->file_emb, L->grads->misc_emb, L->tokens);
    if (g_learner_prof) g_lemb_s += mt_s(mach_absolute_time() - _t);

    // NaN-gradient skip: if ANY gradient is non-finite, skip the optimizer step entirely
    // (zero the grads and return the last finite loss). This is a standard training technique
    // — a non-finite gradient would corrupt the AdamW momentum and poison all subsequent
    // steps. The MPSGraph fp32 generation produces slightly different replay samples than
    // the ANE fp16 path, and certain positions expose a backward numerical instability
    // (overflow in rmsnorm_bwd's reciprocal-RMS when activations are near-zero after a weight
    // update). Skipping the step lets the loop proceed; the next batch draws fresh samples.
    int grad_nan = grads_diagnose(adam_t, 1.0f/(ls*(float)K));
    if (grad_nan) {
        grads_zero();   // poison control: clear the NaN grads so AdamW momentum stays clean
        *out_lp = (float)(lp / K); *out_lv = (float)(lv / K);
        free(batch);
        return;
    }

    _t = g_learner_prof ? mach_absolute_time() : 0;
    optimizer_step(1.0f/(ls*(float)K), cfg->grad_clip, adam_t, cfg->lr, cfg->wd);
    if (g_learner_prof) g_lopt_s += mt_s(mach_absolute_time() - _t);
    _t = g_learner_prof ? mach_absolute_time() : 0;
    chess_net_build_fused(L->net);   // keep fused forward weights in sync with the canonical ones [iter 6]
    if (g_learner_prof) g_lfus_s += mt_s(mach_absolute_time() - _t);

    *out_lp = (float)(lp / K); *out_lv = (float)(lv / K);
    free(batch);
}

// ============================================================================
// SELF-CHECK: batched forward == B independent single forwards (cosine); the net
// evaluator readout matches a single-position readout; priors are a valid distribution and
// values are in [-1,1]. Proves the #18 batched ANE path before any training run (the G0
// discipline: never trust an unverified ANE path — silently-wrong gradients are the enemy).
// ============================================================================
static int selfcheck(ChessNet *net) {
    const int B = 24;
    int P = roundup32(B);   // packed width: exercises the TOK_EMPTY padding (24 real, 32 packed)
    printf("## [selfcheck] batched forward (B=%d, packed=%d) vs B single forwards (fp16 cos ~0.999+)\n", B, P);
    Position pos[64]; uint64_t rr = 12345;
    for (int b = 0; b < B; b++) {            // B distinct positions: short random self-walks
        chess_startpos(&pos[b]);
        int steps = b % 12;
        for (int s = 0; s < steps; s++) { Move mv[MAX_MOVES]; int n = chess_legal_moves(&pos[b], mv);
            if (n == 0) { chess_startpos(&pos[b]); break; } Undo u; chess_make(&pos[b], mv[sm_below(&rr, n)], &u); }
    }
    // batched evaluator (the production path)
    NetEvalCtx ev; net_eval_ctx_alloc(&ev, net, B);
    const Position *pp[64]; const Move *lp[64]; int nl[64]; float *prp[64]; float val[64];
    static float prbuf[64][MAX_MOVES]; static Move legbuf[64][MAX_MOVES];
    for (int b = 0; b < B; b++) { nl[b] = chess_legal_moves(&pos[b], legbuf[b]); pp[b] = &pos[b]; lp[b] = legbuf[b]; prp[b] = prbuf[b]; }
    net_eval_batched(&ev, pp, B, lp, nl, prp, val);

    // single-position reference forward + readout per position
    CActs sa[1]; cacts_alloc(&sa[0], SEQ);
    float *x_in=fmalloc(DIM*SEQ), *x_pre=fmalloc(DIM*SEQ), *x_single=fmalloc(DIM*SEQ);
    uint16_t tk[SEQ];
    double worst_cos = 1.0, worst_pri = 0.0, worst_val = 0.0;
    for (int b = 0; b < B; b++) {
        int16_t t16[CHESS_NUM_TOKENS]; chess_encode(&pos[b], t16);
        for (int t = 0; t < CHESS_NUM_TOKENS; t++) tk[t] = (uint16_t)t16[t];
        for (int t = CHESS_NUM_TOKENS; t < SEQ; t++) tk[t] = TOK_EMPTY;
        chess_embed_posenc_batched(x_in, 1, tk, net->tok_emb, net->rank_emb, net->file_emb, net->misc_emb);
        chess_trunk_forward(net->W, sa, x_in, 1, x_pre, x_single, net->rms_final, g_res_alpha, 0);
        double dot=0,na2=0,nc=0;        // cos(single, the b-th slice of the batched x_final)
        for (int d = 0; d < DIM; d++) for (int t = 0; t < SEQ; t++) {
            double a = x_single[d*SEQ+t], c = ev.x_final[(size_t)d*P*SEQ + (size_t)b*SEQ + t];
            dot += a*c; na2 += a*a; nc += c*c;
        }
        double cosv = dot/(sqrt(na2)*sqrt(nc)+1e-30); if (cosv < worst_cos) worst_cos = cosv;
        float pri_s[MAX_MOVES]; float val_s = chess_policy_value_readout(x_single, SEQ, net->W_pol, net->W_val, legbuf[b], nl[b], pri_s);
        double dv = fabs(val_s - val[b]); if (dv > worst_val) worst_val = dv;
        for (int a = 0; a < nl[b]; a++) { double dp = fabs(pri_s[a] - prbuf[b][a]); if (dp > worst_pri) worst_pri = dp; }
    }
    printf("   batched-vs-single trunk: min cos = %.6f\n", worst_cos);
    printf("   readout: max|prior diff| = %.2e   max|value diff| = %.2e\n", worst_pri, worst_val);
    double psum_err = 0;
    for (int b = 0; b < B; b++) { double s = 0; for (int a = 0; a < nl[b]; a++) s += prbuf[b][a]; double e = fabs(s-1.0); if (e>psum_err) psum_err = e; }
    int vrange = 1; for (int b = 0; b < B; b++) if (val[b] < -1.0001f || val[b] > 1.0001f) vrange = 0;
    printf("   priors sum-to-1 max err = %.2e ; values in [-1,1]: %s\n", psum_err, vrange ? "yes" : "NO");
    int ok = (worst_cos > 0.999) && (worst_pri < 5e-3) && (worst_val < 5e-3) && (psum_err < 1e-4) && vrange;
    printf("   => %s\n\n", ok ? "SELFCHECK OK (batched path == single, readout valid)" : "*** SELFCHECK FAIL ***");
    free(x_in); free(x_pre); free(x_single);
    return ok ? 0 : 1;
}

static void run_bench(NetEvalCtx *evctx, const BatchedChessEvaluator *bev, const SPConfig *cfg) {
    SPConfig bc = *cfg;
    ReplayBuffer rb;
    replay_init(&rb, 1, cfg->seed ^ 0xBEEFull);  // bench only needs generation side effects + checksum
    GenStats gs = {0};

    printf("\n## [bench] generation-only self-play throughput\n");
    printf("# target_games=%d B=%d sims=%d considered=%d max_plies=%d seed=%llu curriculum=%d adjudicate=%d\n",
           cfg->bench_games, cfg->B, cfg->sims, cfg->considered, cfg->max_plies,
           (unsigned long long)cfg->seed, cfg->curriculum, cfg->adjudicate);
    fflush(stdout);

    SPConfig warm = *cfg;
    warm.B = 1; warm.sims = 1; warm.considered = 1; warm.max_plies = 1; warm.bench_games = 1;
    GenStats warm_gs = {0};
    play_selfplay_batch(bev, NULL, &rb, &warm, cfg->seed ^ 0xA11CE5ull, &warm_gs);

    net_eval_profile_reset(evctx);
    mcts_profile_reset();
    g_trunk_prof = 1;   // enable the ane_matmul io/ane sub-profiler for the timed run
    uint64_t t0 = mach_absolute_time();
    int batch = 0;
    while (gs.games < cfg->bench_games) {
        int remain = cfg->bench_games - (int)gs.games;
        bc.B = remain < cfg->B ? remain : cfg->B;
        play_selfplay_batch(bev, NULL, &rb, &bc, cfg->seed + (uint64_t)batch*7919u, &gs);
        batch++;
    }
    double wall_s = tb_ms(mach_absolute_time() - t0) * 1e-3;
    MctsProfile mp = mcts_profile_snapshot();
    double prof_s = mp.alloc_s + mp.root_expand_s + mp.root_eval_s + mp.sim_cpu_s + mp.leaf_eval_s;

    printf("games=%ld plies=%ld sims=%ld nodes=%ld batches=%d wall_s=%.3f\n",
           gs.games, gs.plies, gs.sims, gs.nodes, batch, wall_s);
    printf("games/hour=%.1f positions/s=%.1f sims/s=%.1f nodes/s=%.1f checksum=0x%016llx\n",
           wall_s > 0 ? 3600.0*(double)gs.games/wall_s : 0.0,
           wall_s > 0 ? (double)gs.plies/wall_s : 0.0,
           wall_s > 0 ? (double)gs.sims/wall_s : 0.0,
           wall_s > 0 ? (double)gs.nodes/wall_s : 0.0,
           (unsigned long long)gs.checksum);
    printf("mcts_profile searches=%ld sims=%ld nodes=%ld eval_calls=%ld eval_positions=%ld avg_eval_batch=%.1f profiled_s=%.3f\n",
           mp.searches, mp.sims, mp.nodes, mp.eval_calls, mp.eval_positions,
           mp.eval_calls > 0 ? (double)mp.eval_positions/(double)mp.eval_calls : 0.0,
           prof_s);
    printf("cost_s alloc=%.3f root_expand=%.3f root_eval=%.3f sim_cpu=%.3f leaf_eval=%.3f\n",
           mp.alloc_s, mp.root_expand_s, mp.root_eval_s, mp.sim_cpu_s, mp.leaf_eval_s);
    printf("cost_pct alloc=%.1f root_expand=%.1f root_eval=%.1f sim_cpu=%.1f leaf_eval=%.1f unprofiled=%.1f\n",
           wall_s > 0 ? 100.0*mp.alloc_s/wall_s : 0.0,
           wall_s > 0 ? 100.0*mp.root_expand_s/wall_s : 0.0,
           wall_s > 0 ? 100.0*mp.root_eval_s/wall_s : 0.0,
           wall_s > 0 ? 100.0*mp.sim_cpu_s/wall_s : 0.0,
           wall_s > 0 ? 100.0*mp.leaf_eval_s/wall_s : 0.0,
           wall_s > 0 ? 100.0*(wall_s - prof_s)/wall_s : 0.0);
    double ev_s = evctx->prof_encode_s + evctx->prof_embed_s + evctx->prof_trunk_s + evctx->prof_readout_s;
    printf("eval_profile calls=%ld positions=%ld avg_batch=%.1f profiled_s=%.3f\n",
           evctx->prof_calls, evctx->prof_positions,
           evctx->prof_calls > 0 ? (double)evctx->prof_positions/(double)evctx->prof_calls : 0.0,
           ev_s);
    printf("eval_cost_s encode=%.3f embed=%.3f trunk=%.3f readout=%.3f\n",
           evctx->prof_encode_s, evctx->prof_embed_s, evctx->prof_trunk_s, evctx->prof_readout_s);
    printf("eval_cost_pct encode=%.1f embed=%.1f trunk=%.1f readout=%.1f\n",
            wall_s > 0 ? 100.0*evctx->prof_encode_s/wall_s : 0.0,
            wall_s > 0 ? 100.0*evctx->prof_embed_s/wall_s : 0.0,
            wall_s > 0 ? 100.0*evctx->prof_trunk_s/wall_s : 0.0,
            wall_s > 0 ? 100.0*evctx->prof_readout_s/wall_s : 0.0);
    // trunk sub-profile: io (fp16 convert + IOSurfaceLock) vs ane (ANE dispatch) vs cpu_ops
    double t_io = g_trunk_io_s, t_ane = g_trunk_ane_s;
    double t_attn = g_trunk_attn_s, t_softmax = g_trunk_softmax_s;
    double t_attn_compute = t_attn - t_softmax;
    double t_rms = g_trunk_rms_s, t_silu = g_trunk_silu_s;
    double t_cpuops = evctx->prof_trunk_s - (t_io + t_ane);
    double t_rest = t_cpuops - t_attn;   // rms + silu + residual + memcpy + dispatch overhead
    double t_resid = t_rest - t_rms - t_silu;   // residual + memcpy + GCD overhead
    if (t_io + t_ane > 0.0) {
        printf("trunk_split_s io=%.3f ane=%.3f attn_comp=%.3f softmax=%.3f rms=%.3f silu=%.3f resid=%.3f (trunk=%.3f)\n",
               t_io, t_ane, t_attn_compute, t_softmax, t_rms, t_silu, t_resid, evctx->prof_trunk_s);
        printf("trunk_split_pct_of_wall io=%.1f ane=%.1f attn_comp=%.1f softmax=%.1f rms=%.1f silu=%.1f resid=%.1f\n",
               wall_s > 0 ? 100.0*t_io/wall_s : 0.0,
               wall_s > 0 ? 100.0*t_ane/wall_s : 0.0,
               wall_s > 0 ? 100.0*t_attn_compute/wall_s : 0.0,
               wall_s > 0 ? 100.0*t_softmax/wall_s : 0.0,
               wall_s > 0 ? 100.0*t_rms/wall_s : 0.0,
               wall_s > 0 ? 100.0*t_silu/wall_s : 0.0,
               wall_s > 0 ? 100.0*t_resid/wall_s : 0.0);
    }
    replay_free(&rb);
}

// ============================================================================
// MAIN
// ============================================================================
int main(int argc, char **argv) {
    @autoreleasepool {
        ane_init();
        mach_timebase_info(&g_tb);
        chess_init();
        int mode; SPConfig cfg = sp_parse(argc, argv, &mode);
        g_res_alpha = 1.0f/sqrtf(2.0f*NLAYERS);
#if LILBRO_HAS_MPS
        // --mps-graph: the GPU iteration path (ADR 0006 build-step 2). Eval/generation
        // forward uses the whole-trunk MPSGraph (5.3-6.0x vs ANE+CPU); the LEARNER
        // forward+backward uses mg_ad_* (MPSGraph hybrid autodiff, fp32). The cblas
        // (g_cpu_mm=1) and ANE paths remain as the non-default fallback when --mps-graph
        // is off (AC #25 #4); the autodiff backward subsumes the CPU rmsnorm_bwd overflow
        // bug by construction (ADR 0006 dec 2). Optimizer + heads stay on the CPU floor.
        if (cfg.use_mps_graph) {
            mps_init();
            if (g_mtl_dev) {
                g_use_mps_graph = 1;
                g_cpu_mm = 1;  // cblas fp32 fallback for any non-autodiff ane_matmul call
                printf("# MPSGraph: device=%s ( GPU iteration path — trunk fwd+bwd hybrid autodiff )\n", [[g_mtl_dev name] UTF8String]);
                printf("# learner: MPSGraph autodiff fp32 ( mg_ad_forward/backward ); cblas retained as fallback\n");
            } else { printf("# MPSGraph: init FAILED — falling back to ANE\n"); }
        }
        if (cfg.use_mps && !cfg.use_mps_graph) {
            mps_init();
            if (g_mtl_dev) {
                g_use_mps = 1;
                printf("# MPS: device=%s ( fp32 MPS matmul backend ENABLED )\n", [[g_mtl_dev name] UTF8String]);
            } else { printf("# MPS: init FAILED — falling back to ANE\n"); }
        }
#endif

        printf("# chess self-play (%s) — DIM=%d HIDDEN=%d L=%d SEQ=%d  | B=%d sims=%d considered=%d\n",
               mode == 3 ? "bench" : (mode == 2 ? "selfcheck" : (mode == 1 ? "G2" : "smoke")),
               DIM, HIDDEN, NLAYERS, SEQ, cfg.B, cfg.sims, cfg.considered);
        // Cold-start mitigation (ADR 0005 dec 8, MEASURED-triggered): adjudicate capped games
        // by material so the value head gets a decisive signal when weak self-play never reaches
        // mate (the smoke runs showed 0/32 decisive games without it -> loss_val collapses to 0).
        // TRAINING-ONLY: the eval ladder always scores real outcomes (a capped eval game is a
        // draw). Default ON for smoke/G2; --no-adjudicate (via the flag) keeps the ablation path.
        if (cfg.adjudicate == 0 && (mode == 1 || mode == 0)) cfg.adjudicate = 1;
        printf("# replay=%d lbatch=%d lsteps=%d iters=%d lr=%g vw=%g dir(a=%g,f=%g) temp=%g/%d max_plies=%d seed=%llu\n",
                cfg.replay_cap, cfg.learner_batch, cfg.learner_steps, cfg.iters, cfg.lr, cfg.value_weight,
                cfg.dir_alpha, cfg.dir_frac, cfg.temp, cfg.temp_moves, cfg.max_plies, (unsigned long long)cfg.seed);
        printf("# warmup: iters=%d frac=%g  adjudicate=%d curriculum=%d  td_lambda=%g\n",
                cfg.warmup_iters, (double)cfg.warmup_frac, cfg.adjudicate, cfg.curriculum, (double)cfg.td_lambda);

        // ---- the ONE net + grads + AdamW registry (the single ANE client) ----
        ChessNet net, grads;
        chess_net_alloc(&net, 0); chess_net_alloc(&grads, 1);
        chess_net_init(&net, cfg.seed);
        chess_net_register(&net, &grads);
        // pre-size the cpu_ops RMSNorm scratch for the widest batch we will run (it is lazily
        // sized to the first S it sees; the batched path must pre-size before any rmsnorm).
        int maxB = roundup32(cfg.B > cfg.eval_games ? cfg.B : cfg.eval_games);
        int maxS = maxB*SEQ; if (cfg.learner_batch*SEQ > maxS) maxS = cfg.learner_batch*SEQ;
        chess_net_init_rmstmp(maxS);

        if (cfg.resume) {
            if (chess_net_load(&net, cfg.ckpt)) printf("# resumed from %s\n", cfg.ckpt);
            else printf("# --resume: no/!matching checkpoint at %s — starting from random init\n", cfg.ckpt);
        }
        // Build the fused forward-only weights (QKV, W1/W3) from the canonical ones. Must run
        // before ANY forward (selfcheck/bench/training) and is rebuilt after every optimizer
        // step in the learner. Forward-only; checkpoint + backward use the canonical weights.
        chess_net_build_fused(&net);

        if (mode == 2) return selfcheck(&net);   // --selfcheck

        // shared generation/eval evaluator + learner (both on the one ANE client)
        NetEvalCtx ev; net_eval_ctx_alloc(&ev, &net, maxB);
        BatchedChessEvaluator bev = { .ctx = &ev, .evaluate = net_eval_batched };
        if (mode == 3) { run_bench(&ev, &bev, &cfg); return 0; }

        Learner L; learner_alloc(&L, &net, &grads, cfg.learner_batch);
        ReplayBuffer rb; replay_init(&rb, cfg.replay_cap, cfg.seed ^ 0xBEEFull);

        int iters = (mode == 1) ? cfg.iters : 3;   // smoke = 3 iters
        printf("\n## [%s] self-play loop (%d iterations)\n", mode == 1 ? "G2" : "smoke", iters);
        printf("# %-4s %-9s %-9s %-8s | %-26s | %-26s\n", "iter", "loss_pol", "loss_val", "buf",
               "vs-random  (W/D/L  score)", "vs-greedy  (W/D/L  score)");

        double sc_rand[512], sc_greedy[512]; int npts = 0; int adam_t = 0; int diverged = 0;
        int prof = cfg.profile;
        if (prof) g_learner_prof = 1;
        double tot_gen = 0, tot_learn = 0, tot_evalr = 0, tot_evalg = 0, tot_ckpt = 0;
        for (int it = 0; it <= iters; it++) {
            float lp = 0, lv = 0; GenStats gs = {0,0,0,0,0};
            double it_gen = 0, it_learn = 0;
            g_lfwd_s = g_lloss_s = g_lbwd_s = g_lemb_s = g_lopt_s = g_lfus_s = 0.0;
            if (it > 0) {
                // Warmup value-prior: blend the net's leaf value with a material heuristic for
                // iter < warmup_iters, linearly decaying to 0 (purist-Zero resumes after warmup).
                // GENERATION ONLY: the eval ladder below uses the pure net (bev) so the win-rate
                // measures the net's TRUE strength, not the warmup-boosted search. A search prior,
                // not labels — the priors are the net's own. [dec 8 fallback, MEASURED-triggered]
                BatchedChessEvaluator gen_bev = bev;
                BatchedChessEvaluator warm = bev;
                float frac = 0.0f;
                if (cfg.warmup_iters > 0 && cfg.warmup_frac > 0.0f)
                    frac = cfg.warmup_frac * (float)(cfg.warmup_iters - it > 0 ? (cfg.warmup_iters - it) : 0) / (float)cfg.warmup_iters;
                if (frac > 0.0f) { warm = make_warmup_evaluator(&bev, frac); gen_bev = warm; }
                uint64_t _tg = prof ? mach_absolute_time() : 0;
                play_selfplay_batch(&gen_bev, frac > 0.0f ? &bev : NULL, &rb, &cfg, cfg.seed + (uint64_t)it*7919u, &gs);
                if (prof) { it_gen = mt_s(mach_absolute_time() - _tg); tot_gen += it_gen; }
                if (frac > 0.0f) warmup_evaluator_free(&warm);
                if (replay_count(&rb) >= cfg.learner_batch) {
                    uint64_t _tl = prof ? mach_absolute_time() : 0;
                    for (int s = 0; s < cfg.learner_steps; s++) { float a,b; learner_step(&L, &rb, &cfg, ++adam_t, &a, &b); lp = a; lv = b; }
                    if (prof) { it_learn = mt_s(mach_absolute_time() - _tl); tot_learn += it_learn; }
                }
                // fail-fast on fp16 gradient overflow: a NaN/inf loss means the trunk diverged
                // (lower --lr); abort now rather than spend the rest of the run spewing NaN.
                if (!isfinite(lp) || !isfinite(lv)) {
                    printf("  %-4d DIVERGED  loss_pol=%.4f loss_val=%.4f — fp16 overflow; lower --lr\n", it, lp, lv);
                    fflush(stdout); diverged = 1; break;
                }
            }
            // The eval (full games vs a non-resigning random-mover) is the loop's dominant cost,
            // so it runs only on eval-iters. Every OTHER iter still prints the loss + this-iter
            // generation W/D/L — a near-free learning signal (is loss_val falling off the ln3
            // uniform floor? is generation producing DECISIVE games for the value to learn from?).
            double it_evalr = 0, it_evalg = 0, it_ckpt = 0;
            if (it == 0 || it == iters || (cfg.eval_every > 0 && it % cfg.eval_every == 0)) {
                int w1,d1,l1,w2,d2,l2;
                uint64_t _te = prof ? mach_absolute_time() : 0;
                double sr = eval_vs_opponent(&bev, &cfg, opp_random, cfg.eval_games, cfg.seed ^ (0x1111ull*(uint64_t)(it+1)), &w1,&d1,&l1);
                if (prof) { it_evalr = mt_s(mach_absolute_time() - _te); tot_evalr += it_evalr; }
                _te = prof ? mach_absolute_time() : 0;
                double sg = eval_vs_opponent(&bev, &cfg, opp_greedy, cfg.eval_games, cfg.seed ^ (0x2222ull*(uint64_t)(it+1)), &w2,&d2,&l2);
                if (prof) { it_evalg = mt_s(mach_absolute_time() - _te); tot_evalg += it_evalg; }
                if (npts < (int)(sizeof(sc_rand)/sizeof(sc_rand[0]))) { sc_rand[npts] = sr; sc_greedy[npts] = sg; npts++; }
                printf("  %-4d %-9.4f %-9.4f %-8d | %3d/%3d/%-3d   %.3f       | %3d/%3d/%-3d   %.3f\n",
                       it, lp, lv, replay_count(&rb), w1,d1,l1, sr, w2,d2,l2, sg);
                if (cfg.ckpt && cfg.ckpt[0]) {
                    uint64_t _tc = prof ? mach_absolute_time() : 0;
                    chess_net_save(&net, cfg.ckpt);
                    if (prof) { it_ckpt = mt_s(mach_absolute_time() - _tc); tot_ckpt += it_ckpt; }
                }
            } else {
                printf("  %-4d %-9.4f %-9.4f %-8d | gen W/D/L %ld/%ld/%ld  plies %ld\n",
                       it, lp, lv, replay_count(&rb), gs.wins_w, gs.draws, gs.wins_b, gs.plies);
            }
            if (prof && it > 0) {
                printf("    t_gen=%.2fs t_learn=%.2fs [fwd=%.2f loss=%.2f bwd=%.2f emb=%.2f opt=%.2f fus=%.2f] t_evalr=%.2fs t_evalg=%.2fs t_ckpt=%.2fs\n",
                       it_gen, it_learn, g_lfwd_s, g_lloss_s, g_lbwd_s, g_lemb_s, g_lopt_s, g_lfus_s, it_evalr, it_evalg, it_ckpt);
            }
            fflush(stdout);   // progress visibility for long background runs (output is else fully buffered)
        }
        if (prof) {
            double tot = tot_gen + tot_learn + tot_evalr + tot_evalg + tot_ckpt;
            printf("# profile: gen=%.1fs (%.0f%%) learn=%.1fs (%.0f%%) eval_r=%.1fs (%.0f%%) eval_g=%.1fs (%.0f%%) ckpt=%.1fs (%.0f%%) total=%.1fs\n",
                   tot_gen, 100*tot_gen/tot, tot_learn, 100*tot_learn/tot, tot_evalr, 100*tot_evalr/tot,
                   tot_evalg, 100*tot_evalg/tot, tot_ckpt, 100*tot_ckpt/tot, tot);
        }

        if (diverged) {
            printf("\n## [G2] verdict: ABORTED — training diverged (fp16 overflow). NOT green; lower --lr and rerun.\n");
            return 2;
        }
        if (mode == 1) {
            // ---- G2 verdict (MEASURED): the curve must climb and beat both opponents ----
            double r0 = sc_rand[0], rN = sc_rand[npts-1], g0 = sc_greedy[0], gN = sc_greedy[npts-1];
            double rMax = r0, gMax = g0; for (int i = 0; i < npts; i++) { if (sc_rand[i] > rMax) rMax = sc_rand[i]; if (sc_greedy[i] > gMax) gMax = sc_greedy[i]; }
            int climb_rand   = (rN >= r0 + 0.10) || (r0 >= 0.85 && rN >= 0.85);
            int beat_rand    = (rN >= 0.85);
            int climb_greedy = (gN >= g0 + 0.05) || (gMax >= g0 + 0.05);
            int beat_greedy  = (gN > 0.50);
            int green = climb_rand && beat_rand && climb_greedy && beat_greedy;
            printf("\n## [G2] verdict (MEASURED win-rate curve, not loss):\n");
            printf("   vs random : start %.3f -> end %.3f  (max %.3f)  climb=%s beats-random[>=0.85]=%s\n",
                   r0, rN, rMax, climb_rand?"yes":"NO", beat_rand?"yes":"NO");
            printf("   vs greedy : start %.3f -> end %.3f  (max %.3f)  climb=%s beats-greedy[>0.50]=%s\n",
                   g0, gN, gMax, climb_greedy?"yes":"NO", beat_greedy?"yes":"NO");
            printf("   => %s\n", green ? "G2-GREEN (the loop learns)" : "G2 NOT YET (see curve; consider more iters / curriculum)");
            if (cfg.ckpt && cfg.ckpt[0]) { chess_net_save(&net, cfg.ckpt); printf("   checkpoint: %s\n", cfg.ckpt); }
            return green ? 0 : 1;
        }
        printf("\n# smoke run complete (use --g2 for the gate).\n");
        return 0;
    }
}
