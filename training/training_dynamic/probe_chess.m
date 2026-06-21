// probe_chess.m — Build-step 0: the THROUGHPUT PROBE for chess RL self-play.
//
// The single measurement that gates the whole chess-on-ANE project (ADR 0005):
// can purist-Zero search-guided self-play climb on ONE Mac in DAYS vs NEVER?
//
// It measures, then combines, two costs of the v1 self-play loop:
//   (A) ANE batched forward-only eval at chess shapes — the leaf evaluator.
//   (B) CPU self-play orchestration — stub movegen + stub Gumbel-MCTS bookkeeping.
// → games/day = f(ms/eval@B, n sims/move, ~plies/game, B, max(ANE,CPU)/step).
//
// DISCIPLINE (ADR 0005 / CLAUDE.md): measure don't guess; ONE ANE client; ANE is
// dispatch-bound (batch big, few evals); flag every estimate; cross-check
// "ANE-bound" with powermetrics (run alongside the --sustain phase).
//
// Model-agnostic like probe_dispatch.m: it ignores the compiled MODEL's dims and
// drives CHESS shapes at runtime through gen_dyn_matmul_mil(ic,oc,seq). The
// dynamic trainer's forward is 3 ANE evals/layer (sdpaFwd = QKV-proj+attn,
// woFwd = output matmul, ffnFused = W1/W3 + SiLU + W2); RMSNorm/embed/head are
// CPU (verified in train.m). We reconstruct that 3-dispatch/layer forward from
// measured runtime matmuls so we can sweep batch B without recompiling the
// monolithic kernels (which need on-disk RoPE/mask blobs).
//
// LOCAL non-aborting eval (probe sizes the ANE may reject; we report rejections).
#include "mil_dynamic.h"   // pulls io.h + config.h
#include <stdio.h>
#include <string.h>

// ----------------------------------------------------------------------------
// ANE timing harness (one client; same pattern as probe_dispatch.m)
// ----------------------------------------------------------------------------
static BOOL try_eval(Kern *k) {
    id mdl = (__bridge id)k->model; id req = (__bridge id)k->request; NSError *e = nil;
    return ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
}
static double time_kern(Kern *k, int iters) {
    for (int i = 0; i < 20; i++) if (!try_eval(k)) return -1;   // warmup + validity
    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++) try_eval(k);
    return tb_ms(mach_absolute_time() - t0) / iters;
}

// One forward matmul as ONE ANE eval: y[oc,seq] = W[ic,oc]^T @ x[ic,seq].
// seq = total tokens = B * SEQ_padded (B positions packed into the spatial dim,
// the only way to batch this no-batch-dim pipeline for token-wise ops). Returns
// per-eval ms, or <0 if the ANE rejects the shape (too big / not mult-of-32).
static double matmul_eval_ms(int ic, int oc, int seq, int iters) {
    @autoreleasepool {
        Kern *k = compile_kern_mil_w(gen_dyn_matmul_mil(ic, oc, seq), @{},
                                     ic*(seq+oc)*2, oc*seq*2);
        if (!k) return -2;                       // compile rejected
        float *inbuf = (float*)calloc((size_t)ic*(seq+oc), 4);
        io_write_fp16(k->ioIn, inbuf, ic, seq+oc);
        double ms = time_kern(k, iters);         // <0 if eval rejected
        free(inbuf); free_kern(k);
        return ms;
    }
}

// ----------------------------------------------------------------------------
// Chess net configs (ADR 0005 #12: r2_small/r2_mid shape, d=256-384, 6-8L, ~13-30M)
// Both are MHA at HD=32 (HEADS==KV_HEADS) so Q_DIM == KV_DIM == DIM.
// ----------------------------------------------------------------------------
typedef struct {
    const char *name;
    int dim, heads, hd, hidden, layers;
    long params_M_x10;   // approx params in millions ×10 (trunk only), for reporting
} ChessCfg;

// Approx trunk params (no embed/heads): per layer = Wqkv(3*dim*dim) + Wo(dim*dim)
// + FFN(3*dim*hidden) + 2 norms(2*dim). ×layers.
static double trunk_params_M(const ChessCfg *c) {
    double per = 3.0*c->dim*c->dim + (double)c->dim*c->dim
               + 3.0*c->dim*c->hidden + 2.0*c->dim;
    return per * c->layers / 1e6;
}

// SEQ padded to a multiple of 32 (ANE packs spatial dims to mult-of-32; chess is
// ~77 real tokens = 64 squares + state). 96 = smallest mult-of-32 >= 77.
#define SEQ_REAL 77
#define SEQ_PAD  96

// The four distinct forward matmul shapes per layer (ic,oc), MHA chess:
//   qkv : dim -> 3*dim         (inside sdpaFwd eval)
//   wo  : dim -> dim           (woFwd eval)
//   fup : dim -> 2*hidden      (W1+W3, inside ffnFused eval)
//   fdn : hidden -> dim        (W2, inside ffnFused eval)
// Measured per-eval ms at seq = B*SEQ_PAD. Composed into the real 3-dispatch
// forward: sdpa_eval = qkv + attn_compute; wo_eval = wo; ffn_eval = fup + fdn.
// So t_ane(B) = layers * (qkv + wo + fup + fdn) + layers*attn  with 3 dispatch
// floors/layer already inside qkv/wo/fup (fdn shares ffn's floor → subtract 1).

typedef struct { double qkv, wo, fup, fdn, floor; BOOL ok; } FwdMs;

static FwdMs measure_forward(const ChessCfg *c, int B, int seqpad, int iters) {
    FwdMs r = {0,0,0,0,0,YES};
    int seq = B * seqpad;
    r.qkv = matmul_eval_ms(c->dim,    3*c->dim,    seq, iters);
    r.wo  = matmul_eval_ms(c->dim,    c->dim,      seq, iters);
    r.fup = matmul_eval_ms(c->dim,    2*c->hidden, seq, iters);
    r.fdn = matmul_eval_ms(c->hidden, c->dim,      seq, iters);
    if (r.qkv < 0 || r.wo < 0 || r.fup < 0 || r.fdn < 0) r.ok = NO;
    return r;
}

// Analytic attention compute per layer at batch B (1 eval, fused in sdpaFwd):
//   scores = Q@K^T  : B*heads*seq^2*hd MACs ; out = P@V : same  → 2 matmuls.
// Estimated from the GFLOP/s the projection matmuls achieve at this B (so it
// rides the same measured throughput). FLAGGED as an estimate.
static double attn_ms_estimate(const ChessCfg *c, int B, double gflops_measured) {
    double s = SEQ_PAD;
    double macs = 2.0 * (double)B * c->heads * s * s * c->hd; // QK^T + P@V
    double flops = 2.0 * macs;                                // MAC = 2 FLOP
    if (gflops_measured <= 0) return 0;
    return (flops / (gflops_measured * 1e9)) * 1e3;           // ms
}

// ============================================================================
// (B) CPU self-play orchestration stub — stub movegen + stub Gumbel-MCTS.
// Pure C, no ANE. The HANDOFF's prime suspect for the real bottleneck.
// ============================================================================
//
// Stub movegen: bitboard-flavoured work that produces ~MAXMOVES pseudo-moves —
// occupancy masks, per-piece sliding rays via shifts + popcount. NOT real rules;
// a defensible proxy for the *cost shape* of a bitboard generator. Real movegen
// speed is the C-engine/perft gate (build-step 1); flagged in the writeup.
#define MAXMOVES 40
static volatile uint64_t g_sink;   // defeat dead-code elimination

static int movegen_stub(uint64_t occ, uint64_t own, int *moves_out) {
    int nm = 0;
    uint64_t pieces = own;
    // ~16 pieces; each scans rays in 4-8 directions with masked shifts.
    for (int p = 0; p < 16 && pieces; p++) {
        int from = __builtin_ctzll(pieces | 1ull);
        pieces &= pieces - 1;
        uint64_t attacks = 0;
        // 8 sliding directions, walk until blocked (bitboard ray style).
        const int shifts[8] = {1, -1, 8, -8, 9, -9, 7, -7};
        for (int d = 0; d < 8; d++) {
            int sh = shifts[d];
            uint64_t ray = 1ull << from;
            for (int step = 0; step < 7; step++) {
                ray = (sh > 0) ? (ray << sh) : (ray >> (-sh));
                ray &= ~own;                 // can't land on own
                attacks |= ray;
                if (ray & occ) break;        // blocked by any piece
                if (!ray) break;
            }
        }
        // emit up to a few moves per piece
        uint64_t targets = attacks & ~own;
        while (targets && nm < MAXMOVES) {
            int to = __builtin_ctzll(targets);
            targets &= targets - 1;
            moves_out[nm++] = (from << 6) | to;
        }
    }
    return nm ? nm : 1;
}

// Stub Gumbel-MCTS node + one simulation's CPU bookkeeping (select→expand→backup),
// engine-independent: the tree math AlphaZero pays regardless of movegen impl.
typedef struct MctsNode {
    float Q[MAXMOVES], P[MAXMOVES];
    int   N[MAXMOVES], nchild;
    int   visits;
} MctsNode;

// One simulation for ONE game: descend `depth` nodes (PUCT argmax over children),
// expand a leaf (movegen + prior init from a stub policy vector), backup `depth`.
static double g_value_sink;
static void mcts_one_sim(MctsNode *scratch, int depth, const float *policy_vec,
                         uint64_t occ, uint64_t own, int *movebuf) {
    // ---- SELECT: descend `depth` plies, PUCT argmax each step ----
    double cpuct = 1.25;
    for (int d = 0; d < depth; d++) {
        MctsNode *nd = &scratch[d];
        double sqrtN = sqrt((double)(nd->visits + 1));
        int best = 0; double bestU = -1e30;
        for (int a = 0; a < nd->nchild; a++) {
            double u = nd->Q[a] + cpuct * nd->P[a] * sqrtN / (1.0 + nd->N[a]);
            if (u > bestU) { bestU = u; best = a; }
        }
        nd->N[best]++; nd->visits++;
        g_value_sink += bestU;
    }
    // ---- EXPAND leaf: movegen + softmax-ish prior init from policy logits ----
    MctsNode *leaf = &scratch[depth];
    int nm = movegen_stub(occ, own, movebuf);
    if (nm > MAXMOVES) nm = MAXMOVES;
    leaf->nchild = nm;
    double Z = 0;
    for (int a = 0; a < nm; a++) { double e = exp(policy_vec[a % 64]); leaf->P[a] = (float)e; Z += e; }
    for (int a = 0; a < nm; a++) { leaf->P[a] /= (float)Z; leaf->Q[a] = 0; leaf->N[a] = 0; }
    leaf->visits = 0;
    g_sink ^= (uint64_t)movebuf[nm-1];
    // ---- BACKUP: propagate a value up the descended path ----
    double v = (double)policy_vec[0];
    for (int d = depth - 1; d >= 0; d--) {
        MctsNode *nd = &scratch[d];
        int a = 0; // arbitrary; cost is the update, not the index
        nd->Q[a] += (float)((v - nd->Q[a]) / (nd->N[a] + 1));
        v = -v;    // alternate perspective
    }
}

// Time the CPU work of ONE simulation-step across B parallel games (B sims, one
// per game — what backs one batched ANE forward). Returns ms for the B-wide step.
static double cpu_simstep_ms(int B, int depth, int reps) {
    MctsNode *scratch = (MctsNode*)calloc(depth + 1, sizeof(MctsNode));
    float policy_vec[64];
    for (int i = 0; i < 64; i++) policy_vec[i] = (float)(0.5 - (i % 7) * 0.1);
    int movebuf[MAXMOVES];
    // init the descend path with plausible child counts
    for (int d = 0; d <= depth; d++) { scratch[d].nchild = MAXMOVES; scratch[d].visits = d + 1;
        for (int a = 0; a < MAXMOVES; a++) { scratch[d].P[a] = 1.0f/MAXMOVES; scratch[d].N[a] = 1; } }
    uint64_t occ = 0xFFFF00000000FFFFull, own = 0x000000000000FFFFull;
    // warmup
    for (int b = 0; b < B; b++) mcts_one_sim(scratch, depth, policy_vec, occ, own, movebuf);
    uint64_t t0 = mach_absolute_time();
    for (int r = 0; r < reps; r++)
        for (int b = 0; b < B; b++)
            mcts_one_sim(scratch, depth, policy_vec, occ ^ ((uint64_t)b<<13), own, movebuf);
    double ms = tb_ms(mach_absolute_time() - t0) / reps;
    free(scratch);
    return ms;
}

// Warm the ANE (power-up): the FIRST compiled kernel pays a cold-start tax that
// inflates an isolated floor measurement. Hammer a throwaway kernel so all timed
// evals below run on a warm engine.
static void ane_warmup(void) {
    @autoreleasepool {
        Kern *k = compile_kern_mil_w(gen_dyn_matmul_mil(256, 256, 256), @{}, 256*512*2, 256*256*2);
        if (!k) return;
        float *in = (float*)calloc(256*512, 4);
        io_write_fp16(k->ioIn, in, 256, 512);
        for (int i = 0; i < 80; i++) try_eval(k);
        free(in); free_kern(k);
    }
}

// Find the ANE's max spatial (token) dim: the largest `seq` for which a chess
// matmul still compiles+evals. Caps the batch B (B <= seq_max / SEQ_PAD).
static int find_seq_ceiling(int ic, int oc) {
    int probe[] = {96, 1536, 6144, 12288, 12384, 16320, 16384, 16416, 24576, 32768};
    int n = (int)(sizeof(probe)/sizeof(int)), best = 0;
    for (int i = 0; i < n; i++) {
        int seq = probe[i];
        @autoreleasepool {
            Kern *k = compile_kern_mil_w(gen_dyn_matmul_mil(ic, oc, seq), @{}, ic*(seq+oc)*2, oc*seq*2);
            BOOL ok = NO;
            if (k) {
                float *in = (float*)calloc((size_t)ic*(seq+oc), 4);
                io_write_fp16(k->ioIn, in, ic, seq+oc);
                ok = try_eval(k);
                free(in); free_kern(k);
            }
            printf("   seq=%-6d (B~%-4d) : %s\n", seq, seq/SEQ_PAD, ok ? "OK" : "REJECTED");
            if (ok) best = seq;
        }
    }
    return best;
}

// ============================================================================
// MAIN
// ============================================================================
int main(int argc, char **argv) {
    @autoreleasepool {
        ane_init();
        mach_timebase_info(&g_tb);

        int sustain = 0, sustain_B = 128, sustain_sec = 15;
        for (int i = 1; i < argc; i++) {
            if (!strcmp(argv[i], "--sustain")) sustain = 1;
            else if (!strncmp(argv[i], "--B=", 4)) sustain_B = atoi(argv[i]+4);
            else if (!strncmp(argv[i], "--sec=", 6)) sustain_sec = atoi(argv[i]+6);
        }

        ChessCfg cfgs[] = {
            {"r2_small", 256,  8, 32,  768, 6, 0},
            {"r2_mid",   512, 16, 32, 1408, 8, 0},
        };
        int NCFG = (int)(sizeof(cfgs)/sizeof(cfgs[0]));
        int Bs[] = {1, 16, 64, 128, 170};   // 170*96=16320 just under the 16384 ceiling
        int NB = (int)(sizeof(Bs)/sizeof(int));
        int ITERS = 120;
        int PLIES = 100;            // ~plies/game (ADR; flagged estimate)
        int Ns[] = {8, 16, 32, 64, 128};    // Gumbel sims/move to report (AlphaZero=800; Gumbel=low)
        int NN = (int)(sizeof(Ns)/sizeof(int));
        int MCTS_DEPTH = 12;        // mean tree-descent depth/sim (flagged estimate)

        // ---- SUSTAIN mode: hammer one batched forward so powermetrics can read
        //      ANE power (the two-signal rule). Run: sudo powermetrics --samplers
        //      ane_power -i 1000  in another shell while this runs. ----
        if (sustain) {
            ChessCfg *c = &cfgs[0];
            int seq = sustain_B * SEQ_PAD;
            printf("[SUSTAIN] %s  B=%d  seq=%d  hammering ffn-down matmul for %ds — sample powermetrics now.\n",
                   c->name, sustain_B, seq, sustain_sec);
            Kern *k = compile_kern_mil_w(gen_dyn_matmul_mil(c->hidden, c->dim, seq), @{},
                                         c->hidden*(seq+c->dim)*2, c->dim*seq*2);
            if (!k) { printf("compile/eval REJECTED at B=%d\n", sustain_B); return 1; }
            float *inbuf = (float*)calloc((size_t)c->hidden*(seq+c->dim), 4);
            io_write_fp16(k->ioIn, inbuf, c->hidden, seq+c->dim);
            for (int i = 0; i < 20; i++) if (!try_eval(k)) { printf("eval REJECTED\n"); return 1; }
            uint64_t t0 = mach_absolute_time(); long ev = 0;
            while (tb_ms(mach_absolute_time() - t0) < sustain_sec*1000.0) { try_eval(k); ev++; }
            double ms = tb_ms(mach_absolute_time() - t0);
            printf("[SUSTAIN] done: %ld evals in %.1f s = %.4f ms/eval\n", ev, ms/1000.0, ms/ev);
            free(inbuf); free_kern(k);
            return 0;
        }

        printf("# ============================================================\n");
        printf("# CHESS RL SELF-PLAY THROUGHPUT PROBE (build-step 0, ADR 0005)\n");
        printf("# ============================================================\n");
        printf("# Machine: M-series ANE.  ONE ANE client.  fp16.  LOCAL eval.\n");
        printf("# Chess position = %d real tokens -> padded to %d (mult-of-32).\n", SEQ_REAL, SEQ_PAD);
        printf("# Forward = 3 ANE evals/layer (sdpaFwd, woFwd, ffnFused); RMSNorm/embed/head on CPU.\n");
        printf("# Batch B positions packed into the spatial (token) dim: seq = B*%d.\n\n", SEQ_PAD);

        // -------- (0a) Warm the ANE so the floor isn't cold-start-inflated --------
        printf("## [0a] Warming the ANE (power-up; first kernel pays a cold-start tax)...\n");
        ane_warmup();

        // -------- (0b) Dispatch floor on THIS machine, NOW (warm; re-measure, don't trust priors) --------
        printf("## [0b] Per-eval DISPATCH FLOOR (min over tiny warm matmuls):\n");
        double f1 = matmul_eval_ms(64, 32, 32, 400);
        double f2 = matmul_eval_ms(256, 96, 96, 400);
        double f3 = matmul_eval_ms(128, 64, 64, 400);
        double floor_ms = f1; if (f2 > 0 && f2 < floor_ms) floor_ms = f2; if (f3 > 0 && f3 < floor_ms) floor_ms = f3;
        printf("   floors: 64x32@32=%.4f  256x96@96=%.4f  128x64@64=%.4f  ->  floor ~= %.4f ms/eval\n", f1, f2, f3, floor_ms);
        printf("   (prior/memory: ~0.12 ms; this is the live warm number on this box)\n\n");

        // -------- (0c) ANE max spatial (token) dim -> the batch ceiling --------
        printf("## [0c] ANE max spatial dim (caps batch B; B_max = seq_max/%d):\n", SEQ_PAD);
        int seq_ceiling = find_seq_ceiling(256, 768);
        printf("   => seq_max ~= %d  => B_max ~= %d positions/forward\n\n", seq_ceiling, seq_ceiling/SEQ_PAD);

        // -------- (A) ANE batched forward sweep --------
        // store t_ane(B) per cfg for the combine step
        double t_ane[2][8]; int seq_ok[2][8];
        for (int ci = 0; ci < NCFG; ci++) {
            ChessCfg *c = &cfgs[ci];
            c->params_M_x10 = (long)(trunk_params_M(c) * 10);
            printf("## [A] ANE batched forward — %s  (DIM=%d HEADS=%d HD=%d HIDDEN=%d L=%d, ~%.1fM trunk params)\n",
                   c->name, c->dim, c->heads, c->hd, c->hidden, c->layers, trunk_params_M(c));
            printf("   %-6s %-9s %-9s %-9s %-9s %-9s | %-10s %-11s %-12s\n",
                   "B", "qkv ms", "wo ms", "ffup ms", "ffdn ms", "attn~ms", "fwd ms", "fwd/pos ms", "pos/s");
            for (int bi = 0; bi < NB; bi++) {
                int B = Bs[bi];
                FwdMs f = measure_forward(c, B, SEQ_PAD, ITERS);
                if (!f.ok) {
                    printf("   %-6d  REJECTED (ANE rejected a shape at this B — see qkv/wo/fup/fdn)\n", B);
                    t_ane[ci][bi] = -1; seq_ok[ci][bi] = 0;
                    continue;
                }
                // GFLOP/s from the biggest matmul (ffn-up) to price attention.
                int seq = B * SEQ_PAD;
                double fup_flops = 2.0*(double)c->dim*(2*c->hidden)*seq;
                double gflops = (fup_flops / (f.fup/1e3)) / 1e9;
                double attn = attn_ms_estimate(c, B, gflops);
                // Real forward = 3 dispatches/layer (sdpaFwd, woFwd, ffnFused). Model
                // each layer as 3 dispatch floors + compute-above-floor of its matmuls
                // (qkv+attn in sdpa; wo; fup+fdn in ffn). Robust: no negative terms.
                double cqkv = f.qkv > floor_ms ? f.qkv - floor_ms : 0;
                double cwo  = f.wo  > floor_ms ? f.wo  - floor_ms : 0;
                double cfup = f.fup > floor_ms ? f.fup - floor_ms : 0;
                double cfdn = f.fdn > floor_ms ? f.fdn - floor_ms : 0;
                double per_layer = 3*floor_ms + cqkv + cwo + cfup + cfdn + attn;
                double fwd = c->layers * per_layer;
                double fwd_per_pos = fwd / B;
                t_ane[ci][bi] = fwd; seq_ok[ci][bi] = 1;
                printf("   %-6d %-9.4f %-9.4f %-9.4f %-9.4f %-9.4f | %-10.3f %-11.4f %-12.0f\n",
                       B, f.qkv, f.wo, f.fup, f.fdn, attn, fwd, fwd_per_pos, 1000.0*B/fwd);
            }
            printf("\n");
        }

        // -------- (B) CPU self-play orchestration stub --------
        printf("## [B] CPU stub: movegen + Gumbel-MCTS bookkeeping (no ANE). depth=%d, MAXMOVES=%d\n", MCTS_DEPTH, MAXMOVES);
        printf("   %-6s %-14s %-16s\n", "B", "ms/sim-step", "ms/sim/game(us)");
        double t_cpu[8];
        for (int bi = 0; bi < NB; bi++) {
            int B = Bs[bi];
            int reps = 40000 / B; if (reps < 200) reps = 200;
            double ms = cpu_simstep_ms(B, MCTS_DEPTH, reps);
            t_cpu[bi] = ms;
            printf("   %-6d %-14.5f %-16.4f\n", B, ms, ms*1000.0/B);
        }
        printf("   (movegen is a STUB — real bitboard movegen speed is the C-engine/perft gate, build-step 1. FLAGGED.)\n");
        printf("   BREAK-EVEN: CPU flips to the bottleneck only if real (movegen+tree)/sim/game exceeds\n");
        printf("   t_ane(B)/B = the 'fwd/pos ms' above (e.g. ~0.14 ms = 140 us/sim at r2_small B=64).\n");
        printf("   The stub is ~0.4 us/sim/game, so movegen must be ~300x slower than the stub to flip it.\n\n");

        // -------- (C) COMBINE → games/day --------
        printf("## [C] COMBINE -> games/day  (plies/game=%d [est], pipelined = max(ANE,CPU)/sim-step)\n", PLIES);
        printf("# games/day = B * 86.4e6 / (plies * n * max(t_ane(B), t_cpu(B)))\n");
        printf("# Also report SERIAL (t_ane+t_cpu) as the pessimistic bound.\n\n");
        for (int ci = 0; ci < NCFG; ci++) {
            ChessCfg *c = &cfgs[ci];
            printf("### %s\n", c->name);
            printf("   %-6s %-5s %-10s %-10s %-7s %-13s %-14s %-12s\n",
                   "B", "n", "t_ane ms", "t_cpu ms", "bound", "ms/move(max)", "games/day(max)", "games/day(serial)");
            for (int bi = 0; bi < NB; bi++) {
                int B = Bs[bi];
                if (!seq_ok[ci][bi]) { printf("   %-6d  (forward REJECTED at this B)\n", B); continue; }
                double ta = t_ane[ci][bi], tc = t_cpu[bi];
                const char *bound = ta >= tc ? "ANE" : "CPU";
                for (int ni = 0; ni < NN; ni++) {
                    int n = Ns[ni];
                    double t_move_max    = n * (ta >= tc ? ta : tc);
                    double t_move_serial = n * (ta + tc);
                    double gd_max    = (double)B * 86.4e6 / (PLIES * t_move_max);
                    double gd_serial = (double)B * 86.4e6 / (PLIES * t_move_serial);
                    printf("   %-6d %-5d %-10.3f %-10.3f %-7s %-13.2f %-14.0f %-12.0f\n",
                           B, n, ta, tc, bound, t_move_max, gd_max, gd_serial);
                }
            }
            printf("\n");
        }
        printf("# NOTE: t_ane batches B leaves into 1 forward/sim; t_cpu is B games' tree work/sim.\n");
        printf("# Cross-check the ANE-bound rows with: sudo powermetrics --samplers ane_power -i 1000\n");
        printf("# during  ./probe_chess --sustain --B=128 --sec=20\n");
    }
    return 0;
}
