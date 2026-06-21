// train_chess.m — Build-step 2 of chess-RL-on-ANE (ADR 0005, issue #16):
// policy (8x8x73, legal-masked) + WDL value heads + 2D rank+file posenc (RoPE off)
// + the AlphaZero loss, trained ON THE ANE, with the G0 overfit gate.
//
// WHAT RUNS WHERE (the CPU/ANE split, ADR 0004 / [[ane-resident-training-cpu-floor]]):
//   ANE (fp16): every trunk matmul — QKV/Wo/W1/W3/W2 forward AND the dx backward
//     matmuls — via gen_dyn_matmul_mil(ic,oc,seq). That generic kernel IS the exact
//     primitive train.m's woFwd / qBwd / wotBwd / ffnBwd kernels are built from
//     (mil_dynamic.h lines 218/676/681/425), so this is the proven substrate,
//     un-fused. G0 thus measures the real fp16 forward+backward path.
//   CPU (fp32): RMSNorm, attention softmax (attn_cpu_*, RoPE OFF), SiLU, dW (cblas),
//     the embedding + 2D posenc, the policy/value heads, the AZ loss, AdamW — the
//     irreducible CPU floor. Heads on CPU mirrors the LM classifier head (decision 5)
//     and sidesteps the WDL=3-not-mult-of-32 ANE constraint.
//
// Trunk decomposition note (scope): v1/G0 uses separate matmul evals + CPU attention
// (RoPE off by construction — the chess path never calls the RoPE kernel) for
// correctness and per-kernel verifiability; the fused sdpaFwd/ffnFused kernels are a
// later throughput step (G2+). Attention is causal here (reusing the FD-verified
// attn_cpu core); bidirectional/2D-RoPE attention is a deferred ablation (ADR 0005
// "Open"). Neither affects the G0 correctness gate.
//
// G0 GATE (the discipline — a MEASURED gate): `./train_chess --overfit` pins ONE
// chess position -> (one-hot target policy, one-hot target value) and trains until
// BOTH cross-entropies collapse to ~0. If they do, the new heads + their backward +
// the fp16 trunk path are correct end-to-end. Exit 0 on pass, 1 on fail. Never
// assert a backward is right because it "looks" right.
//
// Build: make train_chess   ·   Gate: make g0

#include "mil_dynamic.h"          // pulls io.h + config.h (DIM/SEQ/... from chess_g0.h)
#include "cpu_ops.h"              // rmsnorm(_bwd), attn_cpu_*, adam_update, embed_*
#include "chess/chess.h"          // engine + codec (#15): encode, legal moves/mask, move index
#include "chess/chess_heads.h"    // NEW heads: posenc, policy, value, L2 (FD-gated)
#include <stdio.h>
#include <string.h>

// ---- chess shapes (from chess.h) -------------------------------------------
#define NBOARD 64
#define PLANES 73
#define POL    CHESS_POLICY_SIZE  // 4672
#define NWDL   3
#define NREAL  CHESS_NUM_TOKENS   // 77
#define NMISC  (SEQ - NBOARD)     // 32 (state tokens + padding)

// ============================================================================
// ANE matmul primitive: y[oc,seq] = W[ic,oc]^T @ x[ic,seq]  (W stored [IN,OUT]).
// Cached per (ic,oc,seq) shape — chess uses only a few. A global CPU fallback
// (cblas) lets --selfcheck run the identical trunk on CPU and compare cos.
// ============================================================================
typedef struct { int ic, oc, seq; Kern *k; } MMEntry;
static MMEntry g_mm[16]; static int g_nmm = 0;
static int g_cpu_mm = 0;   // 1 = use cblas instead of the ANE (selfcheck reference)

static Kern *mm_kernel(int ic, int oc, int seq) {
    for (int i = 0; i < g_nmm; i++)
        if (g_mm[i].ic==ic && g_mm[i].oc==oc && g_mm[i].seq==seq) return g_mm[i].k;
    Kern *k = compile_kern_mil_w(gen_dyn_matmul_mil(ic, oc, seq), @{}, ic*(seq+oc)*2, oc*seq*2);
    if (!k) { fprintf(stderr, "[chess] matmul compile FAILED ic=%d oc=%d seq=%d\n", ic, oc, seq); abort(); }
    g_mm[g_nmm++] = (MMEntry){ic, oc, seq, k};
    return k;
}
static void ane_matmul(int ic, int oc, int seq, const float *x, const float *W, float *y) {
    if (g_cpu_mm) {  // y[oc,seq] = W^T @ x = sum_i W[i,o] x[i,s]
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, oc, seq, ic,
                    1.0f, W, oc, x, seq, 0.0f, y, seq);
        return;
    }
    Kern *k = mm_kernel(ic, oc, seq);
    io_write_dyn(k->ioIn, x, ic, seq, W, oc);
    ane_eval(k);
    io_read_dyn(k->ioOut, y, oc, seq);
}

// transpose src[rows,cols] -> dst[cols,rows]
static void transpose2d(float *dst, const float *src, int rows, int cols) {
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++) dst[(size_t)c*rows + r] = src[(size_t)r*cols + c];
}
// gW[IN,O] += x[IN,seq] @ dy[O,seq]^T   (gW[i,o] += sum_s x[i,s] dy[o,s])
static void dW_acc(float *gW, const float *x, const float *dy, int IN, int O, int seq) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, IN, O, seq,
                1.0f, x, seq, dy, seq, 1.0f, gW, O);
}

// ============================================================================
// Net + per-layer activations / gradients (weights [IN,OUT] row-major).
// ============================================================================
typedef struct { float *Wq,*Wk,*Wv,*Wo,*W1,*W2,*W3,*rms_att,*rms_ffn; } CLayer;
typedef struct { float *layer_in,*xnorm,*Q,*K,*V,*attn,*x2,*x2norm,*h1,*h3,*gate; } CActs;

static float *fmalloc(size_t n) { return (float*)malloc(n*4); }
static float *fcalloc(size_t n) { return (float*)calloc(n,4); }

static void clayer_alloc(CLayer *w) {
    w->Wq=fmalloc(DIM*Q_DIM); w->Wk=fmalloc(DIM*KV_DIM); w->Wv=fmalloc(DIM*KV_DIM);
    w->Wo=fmalloc(Q_DIM*DIM); w->W1=fmalloc(DIM*HIDDEN); w->W2=fmalloc(HIDDEN*DIM);
    w->W3=fmalloc(DIM*HIDDEN); w->rms_att=fmalloc(DIM); w->rms_ffn=fmalloc(DIM);
}
static void clayer_calloc(CLayer *g) {
    g->Wq=fcalloc(DIM*Q_DIM); g->Wk=fcalloc(DIM*KV_DIM); g->Wv=fcalloc(DIM*KV_DIM);
    g->Wo=fcalloc(Q_DIM*DIM); g->W1=fcalloc(DIM*HIDDEN); g->W2=fcalloc(HIDDEN*DIM);
    g->W3=fcalloc(DIM*HIDDEN); g->rms_att=fcalloc(DIM); g->rms_ffn=fcalloc(DIM);
}
static void cacts_alloc(CActs *a) {
    a->layer_in=fmalloc(DIM*SEQ); a->xnorm=fmalloc(DIM*SEQ);
    a->Q=fmalloc(Q_DIM*SEQ); a->K=fmalloc(KV_DIM*SEQ); a->V=fmalloc(KV_DIM*SEQ);
    a->attn=fmalloc(Q_DIM*SEQ); a->x2=fmalloc(DIM*SEQ); a->x2norm=fmalloc(DIM*SEQ);
    a->h1=fmalloc(HIDDEN*SEQ); a->h3=fmalloc(HIDDEN*SEQ); a->gate=fmalloc(HIDDEN*SEQ);
}

// ---- forward: x_in[DIM,SEQ] (embed+posenc already summed) -> x_final[DIM,SEQ] ----
static void chess_forward(CLayer *W, CActs *acts, const float *x_in,
                          float *x_pre_final, float *x_final, const float *rms_final, float res_alpha) {
    float *x = fmalloc(DIM*SEQ); memcpy(x, x_in, DIM*SEQ*4);
    float *o = fmalloc(DIM*SEQ), *ffn = fmalloc(DIM*SEQ);
    for (int L = 0; L < NLAYERS; L++) {
        CActs *ac = &acts[L]; CLayer *w = &W[L];
        memcpy(ac->layer_in, x, DIM*SEQ*4);
        rmsnorm(ac->xnorm, x, w->rms_att, DIM, SEQ);
        ane_matmul(DIM, Q_DIM, SEQ, ac->xnorm, w->Wq, ac->Q);
        ane_matmul(DIM, KV_DIM, SEQ, ac->xnorm, w->Wk, ac->K);
        ane_matmul(DIM, KV_DIM, SEQ, ac->xnorm, w->Wv, ac->V);
        // RoPE OFF (2D posenc replaces it): attention straight on Q/K/V (causal CPU core).
        attn_cpu_forward(ac->attn, ac->Q, ac->K, ac->V, NULL, NULL, NULL, SEQ);
        ane_matmul(Q_DIM, DIM, SEQ, ac->attn, w->Wo, o);
        for (int i = 0; i < DIM*SEQ; i++) ac->x2[i] = x[i] + res_alpha*o[i];
        rmsnorm(ac->x2norm, ac->x2, w->rms_ffn, DIM, SEQ);
        ane_matmul(DIM, HIDDEN, SEQ, ac->x2norm, w->W1, ac->h1);
        ane_matmul(DIM, HIDDEN, SEQ, ac->x2norm, w->W3, ac->h3);
        for (int i = 0; i < HIDDEN*SEQ; i++) {
            float sig = 1.0f/(1.0f+expf(-ac->h1[i]));
            ac->gate[i] = (ac->h1[i]*sig) * ac->h3[i];   // silu(h1)*h3
        }
        ane_matmul(HIDDEN, DIM, SEQ, ac->gate, w->W2, ffn);
        for (int i = 0; i < DIM*SEQ; i++) x[i] = ac->x2[i] + res_alpha*ffn[i];
    }
    memcpy(x_pre_final, x, DIM*SEQ*4);
    rmsnorm(x_final, x, rms_final, DIM, SEQ);
    free(x); free(o); free(ffn);
}

// ---- backward: dx_final[DIM,SEQ] -> grads (accumulated), returns dy into embed/posenc ----
static void chess_backward(CLayer *W, CLayer *G, CActs *acts, const float *dx_final,
                           const float *x_pre_final, const float *rms_final, float *grms_final,
                           float *dy_out, float res_alpha) {
    float *dy = fmalloc(DIM*SEQ);
    rmsnorm_bwd(dy, grms_final, dx_final, x_pre_final, rms_final, DIM, SEQ);  // grad wrt pre-final hidden

    float *dx2=fmalloc(DIM*SEQ), *dffn=fmalloc(DIM*SEQ), *dgate=fmalloc(HIDDEN*SEQ);
    float *dh1=fmalloc(HIDDEN*SEQ), *dh3=fmalloc(HIDDEN*SEQ), *dx2norm=fmalloc(DIM*SEQ);
    float *tmp=fmalloc(DIM*SEQ), *tmpd=fmalloc(DIM*SEQ), *da=fmalloc(DIM*SEQ);
    float *dop=fmalloc(DIM*SEQ), *dattn=fmalloc(Q_DIM*SEQ);
    float *dQ=fmalloc(Q_DIM*SEQ), *dK=fmalloc(KV_DIM*SEQ), *dV=fmalloc(KV_DIM*SEQ), *dxn=fmalloc(DIM*SEQ);
    // backward-weight transposes (recomputed each step; weights change each step)
    float *W2t=fmalloc(DIM*HIDDEN), *W1t=fmalloc(HIDDEN*DIM), *W3t=fmalloc(HIDDEN*DIM);
    float *Wot=fmalloc(DIM*Q_DIM), *Wqt=fmalloc(Q_DIM*DIM), *Wkt=fmalloc(KV_DIM*DIM), *Wvt=fmalloc(KV_DIM*DIM);

    for (int L = NLAYERS-1; L >= 0; L--) {
        CActs *ac = &acts[L]; CLayer *w = &W[L]; CLayer *g = &G[L];
        // out = x2 + res_alpha*ffn  ->  dx2 = dy (so far), dffn = res_alpha*dy
        memcpy(dx2, dy, DIM*SEQ*4);
        for (int i = 0; i < DIM*SEQ; i++) dffn[i] = res_alpha*dy[i];
        // FFN down: ffn = W2^T @ gate
        transpose2d(W2t, w->W2, HIDDEN, DIM);                 // [DIM,HIDDEN]
        ane_matmul(DIM, HIDDEN, SEQ, dffn, W2t, dgate);       // dgate = W2 @ dffn
        dW_acc(g->W2, ac->gate, dffn, HIDDEN, DIM, SEQ);
        // SiLU backward: gate = silu(h1)*h3
        for (int i = 0; i < HIDDEN*SEQ; i++) {
            float sig = 1.0f/(1.0f+expf(-ac->h1[i]));
            float siluprime = sig*(1.0f + ac->h1[i]*(1.0f - sig));
            dh3[i] = dgate[i]*(ac->h1[i]*sig);
            dh1[i] = (dgate[i]*ac->h3[i])*siluprime;
        }
        // FFN up: h1=W1^T@x2norm, h3=W3^T@x2norm -> dx2norm = W1@dh1 + W3@dh3
        transpose2d(W1t, w->W1, DIM, HIDDEN);                 // [HIDDEN,DIM]
        transpose2d(W3t, w->W3, DIM, HIDDEN);
        ane_matmul(HIDDEN, DIM, SEQ, dh1, W1t, dx2norm);
        ane_matmul(HIDDEN, DIM, SEQ, dh3, W3t, tmp);
        for (int i = 0; i < DIM*SEQ; i++) dx2norm[i] += tmp[i];
        dW_acc(g->W1, ac->x2norm, dh1, DIM, HIDDEN, SEQ);
        dW_acc(g->W3, ac->x2norm, dh3, DIM, HIDDEN, SEQ);
        // RMSNorm2 backward, accumulate into dx2 (which already holds the residual dy)
        rmsnorm_bwd(tmpd, g->rms_ffn, dx2norm, ac->x2, w->rms_ffn, DIM, SEQ);
        for (int i = 0; i < DIM*SEQ; i++) dx2[i] += tmpd[i];
        // x2 = layer_in + res_alpha*o -> da = dx2, do = res_alpha*dx2
        memcpy(da, dx2, DIM*SEQ*4);
        for (int i = 0; i < DIM*SEQ; i++) dop[i] = res_alpha*dx2[i];
        // Wo: o = Wo^T @ attn -> dattn = Wo @ do
        transpose2d(Wot, w->Wo, Q_DIM, DIM);                  // [DIM,Q_DIM]
        ane_matmul(DIM, Q_DIM, SEQ, dop, Wot, dattn);
        dW_acc(g->Wo, ac->attn, dop, Q_DIM, DIM, SEQ);
        // attention backward (RoPE off -> no rope_backward) -> dQ,dK,dV
        attn_cpu_backward(dattn, ac->Q, ac->K, ac->V, NULL, NULL, NULL,
                          dQ, dK, dV, NULL, NULL, NULL, SEQ);
        // QKV proj: Q=Wq^T@xnorm ... -> dxnorm = Wq@dQ + Wk@dK + Wv@dV
        transpose2d(Wqt, w->Wq, DIM, Q_DIM);                  // [Q_DIM,DIM]
        transpose2d(Wkt, w->Wk, DIM, KV_DIM);
        transpose2d(Wvt, w->Wv, DIM, KV_DIM);
        ane_matmul(Q_DIM, DIM, SEQ, dQ, Wqt, dxn);
        ane_matmul(KV_DIM, DIM, SEQ, dK, Wkt, tmp); for (int i=0;i<DIM*SEQ;i++) dxn[i]+=tmp[i];
        ane_matmul(KV_DIM, DIM, SEQ, dV, Wvt, tmp); for (int i=0;i<DIM*SEQ;i++) dxn[i]+=tmp[i];
        dW_acc(g->Wq, ac->xnorm, dQ, DIM, Q_DIM, SEQ);
        dW_acc(g->Wk, ac->xnorm, dK, DIM, KV_DIM, SEQ);
        dW_acc(g->Wv, ac->xnorm, dV, DIM, KV_DIM, SEQ);
        // RMSNorm1 backward, accumulate into da (the +layer_in residual)
        rmsnorm_bwd(tmpd, g->rms_att, dxn, ac->layer_in, w->rms_att, DIM, SEQ);
        for (int i = 0; i < DIM*SEQ; i++) da[i] += tmpd[i];
        memcpy(dy, da, DIM*SEQ*4);   // grad wrt this layer's input -> next (earlier) layer
    }
    memcpy(dy_out, dy, DIM*SEQ*4);
    free(dy);free(dx2);free(dffn);free(dgate);free(dh1);free(dh3);free(dx2norm);
    free(tmp);free(tmpd);free(da);free(dop);free(dattn);free(dQ);free(dK);free(dV);free(dxn);
    free(W2t);free(W1t);free(W3t);free(Wot);free(Wqt);free(Wkt);free(Wvt);
}

// ============================================================================
// Param registry for the global grad-norm + clip + AdamW step.
// ============================================================================
typedef struct { float *w, *g; AdamState a; int n; } Param;
static Param g_params[256]; static int g_nparams = 0;
static void reg(float *w, float *g, int n) {
    g_params[g_nparams].w=w; g_params[g_nparams].g=g;
    g_params[g_nparams].a=adam_alloc(n); g_params[g_nparams].n=n; g_nparams++;
}
static void grads_zero(void) {
    for (int i = 0; i < g_nparams; i++) memset(g_params[i].g, 0, (size_t)g_params[i].n*4);
}
// scale by gsc (unscale loss_scale), global-clip to `clip`, AdamW. Mirrors train.m.
static void optimizer_step(float gsc, float clip, int t, float lr, float wd) {
    for (int i = 0; i < g_nparams; i++)
        vDSP_vsmul(g_params[i].g, 1, &gsc, g_params[i].g, 1, (vDSP_Length)g_params[i].n);
    float nsq = 0;
    for (int i = 0; i < g_nparams; i++) { float s; vDSP_dotpr(g_params[i].g,1,g_params[i].g,1,&s,(vDSP_Length)g_params[i].n); nsq += s; }
    float norm = sqrtf(nsq);
    if (clip > 0 && norm > clip) {
        float cs = clip/norm;
        for (int i = 0; i < g_nparams; i++) vDSP_vsmul(g_params[i].g,1,&cs,g_params[i].g,1,(vDSP_Length)g_params[i].n);
    }
    for (int i = 0; i < g_nparams; i++)
        adam_update(g_params[i].w, g_params[i].g, &g_params[i].a, t, lr, 0.9f, 0.999f, 1e-8f, wd);
}

static float frand(void) { return (float)(2*drand48()-1); }

int main(int argc, char *argv[]) {
    @autoreleasepool {
        ane_init();
        mach_timebase_info(&g_tb);

        int do_overfit = 0, do_selfcheck = 0, steps = 600;
        float lr = 1e-3f, thresh = 0.05f, loss_scale = 256.0f, grad_clip = 1.0f, wd = 0.0f;
        float vw = 1.0f, l2 = 0.0f;   // value-loss weight; L2 (0 for the gate so CEs -> 0)
        for (int i = 1; i < argc; i++) {
            if      (!strcmp(argv[i], "--overfit"))   do_overfit = 1;
            else if (!strcmp(argv[i], "--selfcheck")) do_selfcheck = 1;
            else if (!strcmp(argv[i], "--steps") && i+1<argc) steps = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--lr")    && i+1<argc) lr = atof(argv[++i]);
            else if (!strcmp(argv[i], "--thresh")&& i+1<argc) thresh = atof(argv[++i]);
            else if (!strcmp(argv[i], "--clip")  && i+1<argc) grad_clip = atof(argv[++i]);
            else if (!strcmp(argv[i], "--l2")    && i+1<argc) l2 = atof(argv[++i]);
        }
        if (do_overfit) do_selfcheck = 1;  // the gate always proves the substrate is live first
        if (!do_overfit && !do_selfcheck) do_selfcheck = 1;

        printf("# chess G0 trainer — DIM=%d HEADS=%d HD=%d HIDDEN=%d L=%d SEQ=%d VOCAB=%d\n",
               DIM, HEADS, HD, HIDDEN, NLAYERS, SEQ, VOCAB);
        printf("# trunk matmuls on ANE (fp16) via gen_dyn_matmul_mil; heads/loss/norm/attn on CPU (fp32)\n\n");

        float res_alpha = 1.0f/sqrtf(2.0f*NLAYERS);

        // ---- allocate net + grads + acts ----
        CLayer W[NLAYERS], G[NLAYERS]; CActs acts[NLAYERS];
        for (int L = 0; L < NLAYERS; L++) { clayer_alloc(&W[L]); clayer_calloc(&G[L]); cacts_alloc(&acts[L]); }
        float *rms_final=fmalloc(DIM), *grms_final=fcalloc(DIM);
        float *tok_emb=fmalloc((size_t)VOCAB*DIM), *g_tok=fcalloc((size_t)VOCAB*DIM);
        float *rank_emb=fmalloc(8*DIM), *g_rank=fcalloc(8*DIM);
        float *file_emb=fmalloc(8*DIM), *g_file=fcalloc(8*DIM);
        float *misc_emb=fmalloc((size_t)NMISC*DIM), *g_misc=fcalloc((size_t)NMISC*DIM);
        float *W_pol=fmalloc((size_t)DIM*PLANES), *g_pol=fcalloc((size_t)DIM*PLANES);
        float *W_val=fmalloc((size_t)DIM*NWDL), *g_val=fcalloc((size_t)DIM*NWDL);

        // ---- init (mirror train.m scales; heads/embeds small like the LM head) ----
        srand48(42);
        float sd=1.0f/sqrtf(DIM), sq=1.0f/sqrtf(Q_DIM), sh=1.0f/sqrtf(HIDDEN), rs=res_alpha, e=0.02f;
        for (int L=0;L<NLAYERS;L++) {
            for (int i=0;i<DIM*Q_DIM;i++) W[L].Wq[i]=sd*frand();
            for (int i=0;i<DIM*KV_DIM;i++){ W[L].Wk[i]=sd*frand(); W[L].Wv[i]=sd*frand(); }
            for (int i=0;i<Q_DIM*DIM;i++) W[L].Wo[i]=sq*rs*frand();
            for (int i=0;i<DIM*HIDDEN;i++){ W[L].W1[i]=sh*frand(); W[L].W3[i]=sh*frand(); }
            for (int i=0;i<HIDDEN*DIM;i++) W[L].W2[i]=sd*rs*frand();
            for (int i=0;i<DIM;i++){ W[L].rms_att[i]=1.0f; W[L].rms_ffn[i]=1.0f; }
        }
        for (int i=0;i<DIM;i++) rms_final[i]=1.0f;
        for (int i=0;i<(int)(VOCAB*DIM);i++) tok_emb[i]=e*frand();
        for (int i=0;i<8*DIM;i++){ rank_emb[i]=e*frand(); file_emb[i]=e*frand(); }
        for (int i=0;i<NMISC*DIM;i++) misc_emb[i]=e*frand();
        for (int i=0;i<DIM*PLANES;i++) W_pol[i]=e*frand();
        for (int i=0;i<DIM*NWDL;i++)  W_val[i]=e*frand();

        // ---- register params for the optimizer (order is irrelevant) ----
        for (int L=0;L<NLAYERS;L++) {
            reg(W[L].Wq,G[L].Wq,DIM*Q_DIM); reg(W[L].Wk,G[L].Wk,DIM*KV_DIM); reg(W[L].Wv,G[L].Wv,DIM*KV_DIM);
            reg(W[L].Wo,G[L].Wo,Q_DIM*DIM); reg(W[L].W1,G[L].W1,DIM*HIDDEN); reg(W[L].W2,G[L].W2,HIDDEN*DIM);
            reg(W[L].W3,G[L].W3,DIM*HIDDEN); reg(W[L].rms_att,G[L].rms_att,DIM); reg(W[L].rms_ffn,G[L].rms_ffn,DIM);
        }
        reg(rms_final,grms_final,DIM); reg(tok_emb,g_tok,VOCAB*DIM);
        reg(rank_emb,g_rank,8*DIM); reg(file_emb,g_file,8*DIM); reg(misc_emb,g_misc,NMISC*DIM);
        reg(W_pol,g_pol,DIM*PLANES); reg(W_val,g_val,DIM*NWDL);

        // ============================================================
        // Build the fixed G0 batch from the engine (ONE position).
        // ============================================================
        chess_init();
        Position pos; chess_startpos(&pos);
        Move legal[MAX_MOVES]; int nlegal = chess_legal_moves(&pos, legal);
        int16_t toks16[CHESS_NUM_TOKENS]; chess_encode(&pos, toks16);
        uint16_t tokens[SEQ];
        for (int i=0;i<CHESS_NUM_TOKENS;i++) tokens[i]=(uint16_t)toks16[i];
        for (int i=CHESS_NUM_TOKENS;i<SEQ;i++) tokens[i]=TOK_EMPTY;
        // target policy: one-hot on a fixed legal move (=> CE floor 0). target value: one-hot Win.
        static uint8_t legal_mask[POL]; chess_legal_mask(legal, nlegal, legal_mask);
        static float tgt_pol[POL]; memset(tgt_pol,0,sizeof tgt_pol);
        Move tgt_move = legal[0];
        int tgt_idx = chess_move_to_index(tgt_move);
        tgt_pol[tgt_idx] = 1.0f;
        float tgt_val[NWDL] = {1.0f, 0.0f, 0.0f};  // Win (one-hot)
        { char u[6]; chess_move_to_uci(tgt_move,u);
          printf("# G0 batch: startpos, %d legal moves; target move=%s -> policy idx %d; target value=Win\n\n",
                 nlegal, u, tgt_idx); }

        // input embedding + 2D posenc (recomputed each step from the live tables)
        float *x_in=fmalloc(DIM*SEQ), *x_pre=fmalloc(DIM*SEQ), *x_final=fmalloc(DIM*SEQ);
        float *dx_final=fmalloc(DIM*SEQ), *dy_in=fmalloc(DIM*SEQ);

        // ------------------------------------------------------------
        // SELF-CHECK: ANE matmul vs cblas, and ANE-trunk vs CPU-trunk.
        // ------------------------------------------------------------
        if (do_selfcheck) {
            // ASSERTED substrate gate: a fp16 ANE path that disagreed with the CPU
            // reference (or silently ran on stale/zero surfaces) would make any G0
            // pass meaningless, so we track the worst cos and abort if it degrades.
            const double COS_MIN = 0.99;
            double worst_cos = 1.0, cosv;
            printf("## [selfcheck] ANE matmul vs cblas (cos; fp16 expected ~0.999+)\n");
            int shapes[][2] = {{DIM,DIM},{DIM,HIDDEN},{HIDDEN,DIM}};
            for (int s=0;s<3;s++) {
                int ic=shapes[s][0], oc=shapes[s][1];
                float *xx=fmalloc((size_t)ic*SEQ), *ww=fmalloc((size_t)ic*oc);
                float *ya=fmalloc((size_t)oc*SEQ), *yc=fmalloc((size_t)oc*SEQ);
                for (int i=0;i<ic*SEQ;i++) xx[i]=0.1f*frand();
                for (int i=0;i<ic*oc;i++)  ww[i]=0.1f*frand();
                g_cpu_mm=0; ane_matmul(ic,oc,SEQ,xx,ww,ya);
                g_cpu_mm=1; ane_matmul(ic,oc,SEQ,xx,ww,yc); g_cpu_mm=0;
                double dot=0,na=0,nc=0;
                for (int i=0;i<oc*SEQ;i++){ dot+=(double)ya[i]*yc[i]; na+=(double)ya[i]*ya[i]; nc+=(double)yc[i]*yc[i]; }
                cosv = dot/(sqrt(na)*sqrt(nc)+1e-30); if (cosv<worst_cos) worst_cos=cosv;
                printf("   matmul ic=%-4d oc=%-4d : cos=%.6f\n", ic, oc, cosv);
                free(xx);free(ww);free(ya);free(yc);
            }
            // full trunk: ANE vs CPU forward (cos of x_final)
            embed_lookup(x_in, tok_emb, tokens, DIM, SEQ);
            chess_posenc_forward(x_in, rank_emb, file_emb, misc_emb, DIM, SEQ, NBOARD);
            float *xf_cpu=fmalloc(DIM*SEQ);
            g_cpu_mm=1; chess_forward(W, acts, x_in, x_pre, xf_cpu, rms_final, res_alpha);
            g_cpu_mm=0; chess_forward(W, acts, x_in, x_pre, x_final, rms_final, res_alpha);
            double dot=0,na=0,nc=0;
            for (int i=0;i<DIM*SEQ;i++){ dot+=(double)x_final[i]*xf_cpu[i]; na+=(double)x_final[i]*x_final[i]; nc+=(double)xf_cpu[i]*xf_cpu[i]; }
            cosv = dot/(sqrt(na)*sqrt(nc)+1e-30); if (cosv<worst_cos) worst_cos=cosv;
            printf("   full %d-layer trunk fwd: cos(ANE, CPU)=%.6f\n", NLAYERS, cosv);
            free(xf_cpu);
            // BACKWARD substrate check: the dx-matmuls (the fp16 path G0 ultimately
            // trusts) must match CPU too — never assume a backward is right because
            // the forward is. Same saved acts (from the ANE forward above); a random
            // upstream grad; compare the gradient that flows into the embedding.
            float *dxr=fmalloc(DIM*SEQ), *dyA=fmalloc(DIM*SEQ), *dyC=fmalloc(DIM*SEQ);
            for (int i=0;i<DIM*SEQ;i++) dxr[i]=0.1f*frand();
            grads_zero(); g_cpu_mm=1; chess_backward(W,G,acts,dxr,x_pre,rms_final,grms_final,dyC,res_alpha);
            grads_zero(); g_cpu_mm=0; chess_backward(W,G,acts,dxr,x_pre,rms_final,grms_final,dyA,res_alpha);
            grads_zero();   // leave grads clean for training
            dot=0;na=0;nc=0;
            for (int i=0;i<DIM*SEQ;i++){ dot+=(double)dyA[i]*dyC[i]; na+=(double)dyA[i]*dyA[i]; nc+=(double)dyC[i]*dyC[i]; }
            cosv = dot/(sqrt(na)*sqrt(nc)+1e-30); if (cosv<worst_cos) worst_cos=cosv;
            printf("   full %d-layer trunk bwd: cos(ANE, CPU)=%.6f\n", NLAYERS, cosv);
            free(dxr);free(dyA);free(dyC);
            if (worst_cos < COS_MIN) {
                printf("   [selfcheck] FAIL: worst cos %.6f < %.2f — the ANE path is broken/stale\n\n", worst_cos, COS_MIN);
                return 1;
            }
            printf("   [selfcheck] substrate OK (min cos %.6f)\n\n", worst_cos);
            if (!do_overfit) return 0;
        }

        // ------------------------------------------------------------
        // G0 OVERFIT GATE: drive BOTH cross-entropies to ~0 on the ANE.
        // ------------------------------------------------------------
        printf("## [G0] overfit one batch (steps=%d lr=%g clip=%g loss_scale=%g thresh=%g)\n",
               steps, lr, grad_clip, loss_scale, thresh);
        float lp=0, lv=0; int adam_t=0;
        for (int step=0; step<steps; step++) {
            grads_zero();
            // forward
            embed_lookup(x_in, tok_emb, tokens, DIM, SEQ);
            chess_posenc_forward(x_in, rank_emb, file_emb, misc_emb, DIM, SEQ, NBOARD);
            chess_forward(W, acts, x_in, x_pre, x_final, rms_final, res_alpha);
            // heads + AZ loss (CPU); dx_final accumulates policy + value gradients
            memset(dx_final, 0, DIM*SEQ*4);
            lp = chess_policy_loss(x_final, W_pol, DIM, SEQ, NBOARD, PLANES, legal_mask, tgt_pol, dx_final, g_pol);
            float *dxv=fcalloc(DIM*SEQ);
            lv = chess_value_loss(x_final, W_val, DIM, SEQ, NREAL, NWDL, tgt_val, dxv, g_val);
            for (int i=0;i<DIM*SEQ;i++) dx_final[i] += vw*dxv[i];
            // scale g_val by the value weight (policy weight = 1)
            for (int i=0;i<DIM*NWDL;i++) g_val[i] *= vw;
            free(dxv);
            // optional L2 on the trunk weight matrices (the standard AZ L2 target;
            // norms/embeds/posenc/heads excluded, like AdamW no-decay). Reported
            // separately (default 0 so the CEs reach ~0). The grad is added in
            // loss_scale units (l2s) so the uniform 1/loss_scale unscale in the
            // optimizer leaves the true 2*l2*w — same fp16 256x scaling the CE path
            // gets; the returned penalty is divided back out for true-units reporting.
            float l2pen = 0, l2s = l2*loss_scale;
            if (l2 > 0) {
                for (int L=0;L<NLAYERS;L++){
                    l2pen += chess_l2_penalty(W[L].Wq,DIM*Q_DIM,l2s,G[L].Wq);
                    l2pen += chess_l2_penalty(W[L].Wk,DIM*KV_DIM,l2s,G[L].Wk);
                    l2pen += chess_l2_penalty(W[L].Wv,DIM*KV_DIM,l2s,G[L].Wv);
                    l2pen += chess_l2_penalty(W[L].Wo,Q_DIM*DIM,l2s,G[L].Wo);
                    l2pen += chess_l2_penalty(W[L].W1,DIM*HIDDEN,l2s,G[L].W1);
                    l2pen += chess_l2_penalty(W[L].W2,HIDDEN*DIM,l2s,G[L].W2);
                    l2pen += chess_l2_penalty(W[L].W3,DIM*HIDDEN,l2s,G[L].W3);
                }
                l2pen /= loss_scale;  // report the true penalty (grad stayed scaled)
            }
            // loss-scaling: scale the grad entering the trunk + the head weight grads
            vDSP_vsmul(dx_final,1,&loss_scale,dx_final,1,(vDSP_Length)(DIM*SEQ));
            vDSP_vsmul(g_pol,1,&loss_scale,g_pol,1,(vDSP_Length)(DIM*PLANES));
            vDSP_vsmul(g_val,1,&loss_scale,g_val,1,(vDSP_Length)(DIM*NWDL));
            // trunk backward (ANE dx matmuls + CPU dW), then posenc + embed backward
            chess_backward(W, G, acts, dx_final, x_pre, rms_final, grms_final, dy_in, res_alpha);
            chess_posenc_backward(dy_in, g_rank, g_file, g_misc, DIM, SEQ, NBOARD);
            embed_backward(g_tok, dy_in, tokens, DIM, SEQ);
            // optimizer: unscale (1/loss_scale), global-clip, AdamW
            adam_t++;
            optimizer_step(1.0f/loss_scale, grad_clip, adam_t, lr, wd);

            if (step % 50 == 0 || step == steps-1) {
                if (l2 > 0) printf("   step %-4d  loss_pol=%.5f  loss_val=%.5f  l2=%.5f\n", step, lp, lv, l2pen);
                else        printf("   step %-4d  loss_pol=%.5f  loss_val=%.5f\n", step, lp, lv);
            }
        }

        int pass = (lp < thresh) && (lv < thresh);
        printf("\n## [G0] final: loss_pol=%.5f loss_val=%.5f  (thresh %.3f)  =>  %s\n",
               lp, lv, thresh, pass ? "PASS (G0-green)" : "FAIL");
        return pass ? 0 : 1;
    }
}
