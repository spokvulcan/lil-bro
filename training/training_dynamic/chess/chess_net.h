// chess_net.h — the shared chess transformer trunk: the ANE matmul primitive, the
// forward + backward (single-position AND vectorized B-position), batched attention,
// the policy/value inference readout, a whole-net weights container with checkpoint
// I/O, and the AdamW param registry.
//
// This is the SINGLE source of truth for the trunk that build-step 2's G0 gate
// (train_chess.m) verifies and build-step 4's self-play learner (train_selfplay.m)
// trains — "the thing you train is the thing you proved." The B=1 path is byte-identical
// to the pre-#18 inline trunk (re-verified by `make g0`); the B>1 path packs B positions
// along the spatial dim so every trunk matmul is ONE ANE dispatch at seq=B*SEQ
// (results/chess_throughput_probe.md: the never->days pivot, B<=170 under the 16384 wall)
// while attention / RMSNorm / SiLU stay per-column. The B>1 path is verified against B
// independent B=1 forwards by cosine in train_selfplay --selfcheck.
//
// WHAT RUNS WHERE (ADR 0004 / [[ane-resident-training-cpu-floor]]): trunk matmuls on the
// ANE in fp16 via gen_dyn_matmul_mil; RMSNorm / attention softmax / SiLU / dW / embed /
// 2D posenc / heads / loss / AdamW on the CPU in fp32. Include AFTER mil_dynamic.h,
// cpu_ops.h, chess/chess.h, chess/chess_heads.h. Pure C/Obj-C, zero new deps.
#ifndef LILBRO_CHESS_NET_H
#define LILBRO_CHESS_NET_H

#include <stdio.h>
#include <string.h>
#include <math.h>

// ---- chess shapes (from chess.h / the model header) ------------------------
#define NBOARD 64
#define PLANES 73
#define POL    CHESS_POLICY_SIZE   // 4672
#define NWDL   3
#define NREAL  CHESS_NUM_TOKENS    // 77
#define NMISC  (SEQ - NBOARD)      // state tokens + padding

// ============================================================================
// ANE matmul primitive: y[oc,seq] = W[ic,oc]^T @ x[ic,seq]  (W stored [IN,OUT]).
// Cached per (ic,oc,seq) shape. g_cpu_mm=1 forces the cblas reference (selfcheck).
// seq = B*SEQ when batching B positions: each output column y[o, b*SEQ+p] depends
// only on input column x[:, b*SEQ+p], so packing B positions is exact (no cross-
// position mixing in a matmul) and costs ONE dispatch instead of B.
// ============================================================================
typedef struct { int ic, oc, seq; Kern *k; } MMEntry;
static MMEntry g_mm[256]; static int g_nmm = 0;   // many (ic,oc,seq) shapes: B is bucketed
static int g_cpu_mm = 0;

static Kern *mm_kernel(int ic, int oc, int seq) {
    for (int i = 0; i < g_nmm; i++)
        if (g_mm[i].ic==ic && g_mm[i].oc==oc && g_mm[i].seq==seq) return g_mm[i].k;
    if (g_nmm >= (int)(sizeof(g_mm)/sizeof(g_mm[0]))) {
        fprintf(stderr, "[chess_net] matmul kernel cache overflow (%d shapes)\n", g_nmm); abort();
    }
    Kern *k = compile_kern_mil_w(gen_dyn_matmul_mil(ic, oc, seq), @{}, ic*(seq+oc)*2, oc*seq*2);
    if (!k) { fprintf(stderr, "[chess_net] matmul compile FAILED ic=%d oc=%d seq=%d\n", ic, oc, seq); abort(); }
    g_mm[g_nmm++] = (MMEntry){ic, oc, seq, k};
    return k;
}
static void ane_matmul(int ic, int oc, int seq, const float *x, const float *W, float *y) {
    if (g_cpu_mm) {
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
// gW[IN,O] += x[IN,seq] @ dy[O,seq]^T   (gW[i,o] += sum_s x[i,s] dy[o,s])  — for a batch
// of B positions packed along seq this sums per-position gradients = the minibatch grad.
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
// Activations sized for up to `maxS` packed tokens (= maxB*SEQ). B=1 path uses maxS=SEQ.
static void cacts_alloc(CActs *a, int maxS) {
    a->layer_in=fmalloc((size_t)DIM*maxS); a->xnorm=fmalloc((size_t)DIM*maxS);
    a->Q=fmalloc((size_t)Q_DIM*maxS); a->K=fmalloc((size_t)KV_DIM*maxS); a->V=fmalloc((size_t)KV_DIM*maxS);
    a->attn=fmalloc((size_t)Q_DIM*maxS); a->x2=fmalloc((size_t)DIM*maxS); a->x2norm=fmalloc((size_t)DIM*maxS);
    a->h1=fmalloc((size_t)HIDDEN*maxS); a->h3=fmalloc((size_t)HIDDEN*maxS); a->gate=fmalloc((size_t)HIDDEN*maxS);
}

// Ensure the cpu_ops RMSNorm scratch is sized for `maxS` columns (it is lazily sized to
// the FIRST S it sees; the batched path must pre-size it before any rmsnorm call).
static void chess_net_init_rmstmp(int maxS) {
    if (g_rms_tmp) { free(g_rms_tmp); }
    g_rms_tmp = (float*)malloc((size_t)maxS*4);
}

// ============================================================================
// Batched causal attention (per-position, channel stride B*seqp). Chess v1 has no
// attention-sink / qk-norm, so this is the plain causal core of attn_cpu_forward; for
// B=1 it is byte-identical to attn_cpu_forward(...,NULL,NULL,NULL,SEQ). GQA-aware
// (kv-head = h % KV_HEADS); chess_g0 is MHA so kv-head == head.
// ============================================================================
static void attn_cpu_forward_batched(float *attn_out, const float *Q, const float *K,
                                     const float *V, int B, int seqp) {
    int S = B*seqp;
    float scale = 1.0f/sqrtf((float)HD);
    for (int b = 0; b < B; b++) {
        int base = b*seqp;
        for (int h = 0; h < HEADS; h++) {
            int kvh = h % KV_HEADS;
            for (int q = 0; q < seqp; q++) {
                float sc[SEQ]; float m = -1e30f;
                for (int j = 0; j <= q; j++) {
                    float dot = 0; for (int d = 0; d < HD; d++) dot += Q[(h*HD+d)*S+base+q]*K[(kvh*HD+d)*S+base+j];
                    sc[j] = dot*scale; if (sc[j] > m) m = sc[j];
                }
                float Z = 0; for (int j = 0; j <= q; j++) { sc[j] = expf(sc[j]-m); Z += sc[j]; }
                float inv = 1.0f/Z;
                for (int d = 0; d < HD; d++) {
                    float acc = 0; for (int j = 0; j <= q; j++) acc += sc[j]*V[(kvh*HD+d)*S+base+j];
                    attn_out[(h*HD+d)*S+base+q] = acc*inv;
                }
            }
        }
    }
}
// Backward of attn_cpu_forward_batched. da[Q_DIM,S] -> dQ[Q_DIM,S], dK/dV[KV_DIM,S]
// (accumulated, GQA-reduced). Recomputes the softmax. B=1 == attn_cpu_backward (no knobs).
static void attn_cpu_backward_batched(const float *da, const float *Q, const float *K,
                                      const float *V, int B, int seqp,
                                      float *dQ, float *dK, float *dV) {
    int S = B*seqp;
    float scale = 1.0f/sqrtf((float)HD);
    memset(dQ, 0, (size_t)Q_DIM*S*4); memset(dK, 0, (size_t)KV_DIM*S*4); memset(dV, 0, (size_t)KV_DIM*S*4);
    for (int b = 0; b < B; b++) {
        int base = b*seqp;
        for (int h = 0; h < HEADS; h++) {
            int kvh = h % KV_HEADS;
            for (int q = 0; q < seqp; q++) {
                float sc[SEQ]; float m = -1e30f;
                for (int j = 0; j <= q; j++) {
                    float dot = 0; for (int d = 0; d < HD; d++) dot += Q[(h*HD+d)*S+base+q]*K[(kvh*HD+d)*S+base+j];
                    sc[j] = dot*scale; if (sc[j] > m) m = sc[j];
                }
                float Z = 0; for (int j = 0; j <= q; j++) { sc[j] = expf(sc[j]-m); Z += sc[j]; }
                float inv = 1.0f/Z; float p[SEQ]; for (int j = 0; j <= q; j++) p[j] = sc[j]*inv;
                float dp[SEQ];
                for (int j = 0; j <= q; j++) {
                    float acc = 0;
                    for (int d = 0; d < HD; d++) { float dad = da[(h*HD+d)*S+base+q];
                        acc += dad*V[(kvh*HD+d)*S+base+j]; dV[(kvh*HD+d)*S+base+j] += p[j]*dad; }
                    dp[j] = acc;
                }
                float g = 0; for (int j = 0; j <= q; j++) g += p[j]*dp[j];
                for (int j = 0; j <= q; j++) {
                    float dscore = p[j]*(dp[j]-g)*scale;
                    for (int d = 0; d < HD; d++) {
                        dQ[(h*HD+d)*S+base+q] += dscore*K[(kvh*HD+d)*S+base+j];
                        dK[(kvh*HD+d)*S+base+j] += dscore*Q[(h*HD+d)*S+base+q];
                    }
                }
            }
        }
    }
}

// ============================================================================
// Trunk forward: x_in[DIM, B*SEQ] (embed+posenc already summed) -> x_final[DIM, B*SEQ].
// save_acts=1 -> acts is NLAYERS-long (kept for backward); =0 -> acts[0] reused (eval).
// B=1 is byte-identical to the pre-#18 chess_forward.
// ============================================================================
static void chess_trunk_forward(CLayer *W, CActs *acts, const float *x_in, int B,
                                float *x_pre_final, float *x_final, const float *rms_final,
                                float res_alpha, int save_acts) {
    int S = B*SEQ;
    float *x = fmalloc((size_t)DIM*S); memcpy(x, x_in, (size_t)DIM*S*4);
    float *o = fmalloc((size_t)DIM*S), *ffn = fmalloc((size_t)DIM*S);
    for (int L = 0; L < NLAYERS; L++) {
        CActs *ac = save_acts ? &acts[L] : &acts[0];
        CLayer *w = &W[L];
        memcpy(ac->layer_in, x, (size_t)DIM*S*4);
        rmsnorm(ac->xnorm, x, w->rms_att, DIM, S);
        ane_matmul(DIM, Q_DIM, S, ac->xnorm, w->Wq, ac->Q);
        ane_matmul(DIM, KV_DIM, S, ac->xnorm, w->Wk, ac->K);
        ane_matmul(DIM, KV_DIM, S, ac->xnorm, w->Wv, ac->V);
        attn_cpu_forward_batched(ac->attn, ac->Q, ac->K, ac->V, B, SEQ);
        ane_matmul(Q_DIM, DIM, S, ac->attn, w->Wo, o);
        for (int i = 0; i < DIM*S; i++) ac->x2[i] = x[i] + res_alpha*o[i];
        rmsnorm(ac->x2norm, ac->x2, w->rms_ffn, DIM, S);
        ane_matmul(DIM, HIDDEN, S, ac->x2norm, w->W1, ac->h1);
        ane_matmul(DIM, HIDDEN, S, ac->x2norm, w->W3, ac->h3);
        for (int i = 0; i < HIDDEN*S; i++) {
            float sig = 1.0f/(1.0f+expf(-ac->h1[i]));
            ac->gate[i] = (ac->h1[i]*sig) * ac->h3[i];
        }
        ane_matmul(HIDDEN, DIM, S, ac->gate, w->W2, ffn);
        for (int i = 0; i < DIM*S; i++) x[i] = ac->x2[i] + res_alpha*ffn[i];
    }
    memcpy(x_pre_final, x, (size_t)DIM*S*4);
    rmsnorm(x_final, x, rms_final, DIM, S);
    free(x); free(o); free(ffn);
}

// Trunk backward: dx_final[DIM, B*SEQ] -> grads (accumulated), returns dy into embed/posenc.
// B=1 is byte-identical to the pre-#18 chess_backward.
static void chess_trunk_backward(CLayer *W, CLayer *G, CActs *acts, const float *dx_final, int B,
                                 const float *x_pre_final, const float *rms_final, float *grms_final,
                                 float *dy_out, float res_alpha) {
    int S = B*SEQ;
    float *dy = fmalloc((size_t)DIM*S);
    rmsnorm_bwd(dy, grms_final, dx_final, x_pre_final, rms_final, DIM, S);

    float *dx2=fmalloc((size_t)DIM*S), *dffn=fmalloc((size_t)DIM*S), *dgate=fmalloc((size_t)HIDDEN*S);
    float *dh1=fmalloc((size_t)HIDDEN*S), *dh3=fmalloc((size_t)HIDDEN*S), *dx2norm=fmalloc((size_t)DIM*S);
    float *tmp=fmalloc((size_t)DIM*S), *tmpd=fmalloc((size_t)DIM*S), *da=fmalloc((size_t)DIM*S);
    float *dop=fmalloc((size_t)DIM*S), *dattn=fmalloc((size_t)Q_DIM*S);
    float *dQ=fmalloc((size_t)Q_DIM*S), *dK=fmalloc((size_t)KV_DIM*S), *dV=fmalloc((size_t)KV_DIM*S), *dxn=fmalloc((size_t)DIM*S);
    float *W2t=fmalloc(DIM*HIDDEN), *W1t=fmalloc(HIDDEN*DIM), *W3t=fmalloc(HIDDEN*DIM);
    float *Wot=fmalloc(DIM*Q_DIM), *Wqt=fmalloc(Q_DIM*DIM), *Wkt=fmalloc(KV_DIM*DIM), *Wvt=fmalloc(KV_DIM*DIM);

    for (int L = NLAYERS-1; L >= 0; L--) {
        CActs *ac = &acts[L]; CLayer *w = &W[L]; CLayer *g = &G[L];
        memcpy(dx2, dy, (size_t)DIM*S*4);
        for (int i = 0; i < DIM*S; i++) dffn[i] = res_alpha*dy[i];
        transpose2d(W2t, w->W2, HIDDEN, DIM);
        ane_matmul(DIM, HIDDEN, S, dffn, W2t, dgate);
        dW_acc(g->W2, ac->gate, dffn, HIDDEN, DIM, S);
        for (int i = 0; i < HIDDEN*S; i++) {
            float sig = 1.0f/(1.0f+expf(-ac->h1[i]));
            float siluprime = sig*(1.0f + ac->h1[i]*(1.0f - sig));
            dh3[i] = dgate[i]*(ac->h1[i]*sig);
            dh1[i] = (dgate[i]*ac->h3[i])*siluprime;
        }
        transpose2d(W1t, w->W1, DIM, HIDDEN);
        transpose2d(W3t, w->W3, DIM, HIDDEN);
        ane_matmul(HIDDEN, DIM, S, dh1, W1t, dx2norm);
        ane_matmul(HIDDEN, DIM, S, dh3, W3t, tmp);
        for (int i = 0; i < DIM*S; i++) dx2norm[i] += tmp[i];
        dW_acc(g->W1, ac->x2norm, dh1, DIM, HIDDEN, S);
        dW_acc(g->W3, ac->x2norm, dh3, DIM, HIDDEN, S);
        rmsnorm_bwd(tmpd, g->rms_ffn, dx2norm, ac->x2, w->rms_ffn, DIM, S);
        for (int i = 0; i < DIM*S; i++) dx2[i] += tmpd[i];
        memcpy(da, dx2, (size_t)DIM*S*4);
        for (int i = 0; i < DIM*S; i++) dop[i] = res_alpha*dx2[i];
        transpose2d(Wot, w->Wo, Q_DIM, DIM);
        ane_matmul(DIM, Q_DIM, S, dop, Wot, dattn);
        dW_acc(g->Wo, ac->attn, dop, Q_DIM, DIM, S);
        attn_cpu_backward_batched(dattn, ac->Q, ac->K, ac->V, B, SEQ, dQ, dK, dV);
        transpose2d(Wqt, w->Wq, DIM, Q_DIM);
        transpose2d(Wkt, w->Wk, DIM, KV_DIM);
        transpose2d(Wvt, w->Wv, DIM, KV_DIM);
        ane_matmul(Q_DIM, DIM, S, dQ, Wqt, dxn);
        ane_matmul(KV_DIM, DIM, S, dK, Wkt, tmp); for (int i=0;i<DIM*S;i++) dxn[i]+=tmp[i];
        ane_matmul(KV_DIM, DIM, S, dV, Wvt, tmp); for (int i=0;i<DIM*S;i++) dxn[i]+=tmp[i];
        dW_acc(g->Wq, ac->xnorm, dQ, DIM, Q_DIM, S);
        dW_acc(g->Wk, ac->xnorm, dK, DIM, KV_DIM, S);
        dW_acc(g->Wv, ac->xnorm, dV, DIM, KV_DIM, S);
        rmsnorm_bwd(tmpd, g->rms_att, dxn, ac->layer_in, w->rms_att, DIM, S);
        for (int i = 0; i < DIM*S; i++) da[i] += tmpd[i];
        memcpy(dy, da, (size_t)DIM*S*4);
    }
    memcpy(dy_out, dy, (size_t)DIM*S*4);
    free(dy);free(dx2);free(dffn);free(dgate);free(dh1);free(dh3);free(dx2norm);
    free(tmp);free(tmpd);free(da);free(dop);free(dattn);free(dQ);free(dK);free(dV);free(dxn);
    free(W2t);free(W1t);free(W3t);free(Wot);free(Wqt);free(Wkt);free(Wvt);
}

// ============================================================================
// Batched embedding + 2D posenc input builder, and its backward (channel stride B*SEQ).
// For B=1 these match embed_lookup + chess_posenc_forward / their backward.
// tokens is B*SEQ uint16 (position b at [b*SEQ, (b+1)*SEQ)).
// ============================================================================
static void chess_embed_posenc_batched(float *x_in, int B, const uint16_t *tokens,
                                       const float *tok_emb, const float *rank_emb,
                                       const float *file_emb, const float *misc_emb) {
    int S = B*SEQ;
    for (int b = 0; b < B; b++) {
        const uint16_t *tk = tokens + b*SEQ;
        for (int t = 0; t < SEQ; t++) {
            int tok = tk[t], col = b*SEQ + t;
            for (int d = 0; d < DIM; d++) x_in[d*S+col] = tok_emb[tok*DIM+d];
            if (t < NBOARD) { int rk = t>>3, fl = t&7;
                for (int d = 0; d < DIM; d++) x_in[d*S+col] += rank_emb[rk*DIM+d] + file_emb[fl*DIM+d]; }
            else { int mi = t - NBOARD;
                for (int d = 0; d < DIM; d++) x_in[d*S+col] += misc_emb[mi*DIM+d]; }
        }
    }
}
static void chess_embed_posenc_backward_batched(const float *dx, int B, float *d_tok,
                                                float *d_rank, float *d_file, float *d_misc,
                                                const uint16_t *tokens) {
    int S = B*SEQ;
    for (int b = 0; b < B; b++) {
        const uint16_t *tk = tokens + b*SEQ;
        for (int t = 0; t < SEQ; t++) {
            int tok = tk[t], col = b*SEQ + t;
            for (int d = 0; d < DIM; d++) d_tok[tok*DIM+d] += dx[d*S+col];
            if (t < NBOARD) { int rk = t>>3, fl = t&7;
                for (int d = 0; d < DIM; d++) { d_rank[rk*DIM+d] += dx[d*S+col]; d_file[fl*DIM+d] += dx[d*S+col]; } }
            else { int mi = t - NBOARD;
                for (int d = 0; d < DIM; d++) d_misc[mi*DIM+d] += dx[d*S+col]; }
        }
    }
}

// ============================================================================
// Policy/value INFERENCE readout (the ChessEvaluator seam, decision 2): forward-only
// twin of chess_policy_loss/chess_value_loss. For ONE position whose final hidden lives
// at x[d*stride + tok] (stride = the channel stride = B*SEQ in a batch, SEQ if single):
//   priors[i] = softmax over legal moves of logit(legal[i]); legal[i]'s logit reads its
//              policy index idx=chess_move_to_index -> (square=idx/PLANES, plane=idx%PLANES).
//   value     = q[Win] - q[Loss] in [-1,1], q = softmax(WDL logits from the mean-pooled
//              first NREAL real tokens). Exactly what mcts.h's evaluator contract wants.
// ============================================================================
static float chess_policy_value_readout(const float *x, int stride, const float *W_pol,
                                        const float *W_val, const Move *legal, int n_legal,
                                        float *priors) {
    // --- policy: legal-masked softmax directly over the n_legal moves ---
    float mx = -INFINITY;
    for (int i = 0; i < n_legal; i++) {
        int idx = chess_move_to_index(legal[i]);
        int sq = idx / PLANES, pl = idx % PLANES;
        float acc = 0; for (int d = 0; d < DIM; d++) acc += W_pol[d*PLANES+pl] * x[d*stride+sq];
        priors[i] = acc; if (acc > mx) mx = acc;
    }
    float Z = 0; for (int i = 0; i < n_legal; i++) { priors[i] = expf(priors[i]-mx); Z += priors[i]; }
    float invZ = 1.0f/Z; for (int i = 0; i < n_legal; i++) priors[i] *= invZ;
    // --- value: WDL softmax from mean-pooled real tokens -> W - L ---
    float pooled[DIM]; float invn = 1.0f/(float)NREAL;
    for (int d = 0; d < DIM; d++) { float acc = 0; for (int p = 0; p < NREAL; p++) acc += x[d*stride+p]; pooled[d] = acc*invn; }
    float vlogit[NWDL], vm = -INFINITY;
    for (int k = 0; k < NWDL; k++) { float acc = 0; for (int d = 0; d < DIM; d++) acc += W_val[d*NWDL+k]*pooled[d]; vlogit[k] = acc; if (acc > vm) vm = acc; }
    float vZ = 0, q[NWDL]; for (int k = 0; k < NWDL; k++) { q[k] = expf(vlogit[k]-vm); vZ += q[k]; }
    float invvZ = 1.0f/vZ; for (int k = 0; k < NWDL; k++) q[k] *= invvZ;
    return q[0] - q[2];   // WDL order {Win, Draw, Loss}; value in [-1,1], stm perspective
}

// ============================================================================
// Whole-net weights container + checkpoint I/O + AdamW param registry.
// ============================================================================
typedef struct {
    CLayer W[NLAYERS];
    float *rms_final, *tok_emb, *rank_emb, *file_emb, *misc_emb, *W_pol, *W_val;
} ChessNet;

typedef struct { float *p; int n; } ParamRef;
// Enumerate every trainable tensor of *n in a FIXED order (used by register/save/load).
static int chess_net_params(ChessNet *n, ParamRef *out) {
    int k = 0;
    for (int L = 0; L < NLAYERS; L++) {
        out[k++]=(ParamRef){n->W[L].Wq, DIM*Q_DIM};  out[k++]=(ParamRef){n->W[L].Wk, DIM*KV_DIM};
        out[k++]=(ParamRef){n->W[L].Wv, DIM*KV_DIM}; out[k++]=(ParamRef){n->W[L].Wo, Q_DIM*DIM};
        out[k++]=(ParamRef){n->W[L].W1, DIM*HIDDEN}; out[k++]=(ParamRef){n->W[L].W2, HIDDEN*DIM};
        out[k++]=(ParamRef){n->W[L].W3, DIM*HIDDEN}; out[k++]=(ParamRef){n->W[L].rms_att, DIM};
        out[k++]=(ParamRef){n->W[L].rms_ffn, DIM};
    }
    out[k++]=(ParamRef){n->rms_final, DIM};       out[k++]=(ParamRef){n->tok_emb, VOCAB*DIM};
    out[k++]=(ParamRef){n->rank_emb, 8*DIM};      out[k++]=(ParamRef){n->file_emb, 8*DIM};
    out[k++]=(ParamRef){n->misc_emb, NMISC*DIM};  out[k++]=(ParamRef){n->W_pol, DIM*PLANES};
    out[k++]=(ParamRef){n->W_val, DIM*NWDL};
    return k;
}

static void chess_net_alloc(ChessNet *n, int zero) {
    for (int L = 0; L < NLAYERS; L++) { if (zero) clayer_calloc(&n->W[L]); else clayer_alloc(&n->W[L]); }
    n->rms_final = zero?fcalloc(DIM):fmalloc(DIM);
    n->tok_emb   = zero?fcalloc((size_t)VOCAB*DIM):fmalloc((size_t)VOCAB*DIM);
    n->rank_emb  = zero?fcalloc(8*DIM):fmalloc(8*DIM);
    n->file_emb  = zero?fcalloc(8*DIM):fmalloc(8*DIM);
    n->misc_emb  = zero?fcalloc((size_t)NMISC*DIM):fmalloc((size_t)NMISC*DIM);
    n->W_pol     = zero?fcalloc((size_t)DIM*PLANES):fmalloc((size_t)DIM*PLANES);
    n->W_val     = zero?fcalloc((size_t)DIM*NWDL):fmalloc((size_t)DIM*NWDL);
}

// Random init — mirrors train_chess.m's G0 init EXACTLY (same scales + drand48 order) so
// a given seed reproduces the same starting net (the project's determinism discipline).
static void chess_net_init(ChessNet *n, uint64_t seed) {
    srand48((long)seed);
    float res_alpha = 1.0f/sqrtf(2.0f*NLAYERS);
    float sd=1.0f/sqrtf(DIM), sq=1.0f/sqrtf(Q_DIM), sh=1.0f/sqrtf(HIDDEN), rs=res_alpha, e=0.02f;
    #define FR (float)(2*drand48()-1)
    for (int L=0;L<NLAYERS;L++) {
        for (int i=0;i<DIM*Q_DIM;i++) n->W[L].Wq[i]=sd*FR;
        for (int i=0;i<DIM*KV_DIM;i++){ n->W[L].Wk[i]=sd*FR; n->W[L].Wv[i]=sd*FR; }
        for (int i=0;i<Q_DIM*DIM;i++) n->W[L].Wo[i]=sq*rs*FR;
        for (int i=0;i<DIM*HIDDEN;i++){ n->W[L].W1[i]=sh*FR; n->W[L].W3[i]=sh*FR; }
        for (int i=0;i<HIDDEN*DIM;i++) n->W[L].W2[i]=sd*rs*FR;
        for (int i=0;i<DIM;i++){ n->W[L].rms_att[i]=1.0f; n->W[L].rms_ffn[i]=1.0f; }
    }
    for (int i=0;i<DIM;i++) n->rms_final[i]=1.0f;
    for (int i=0;i<(int)(VOCAB*DIM);i++) n->tok_emb[i]=e*FR;
    for (int i=0;i<8*DIM;i++){ n->rank_emb[i]=e*FR; n->file_emb[i]=e*FR; }
    for (int i=0;i<NMISC*DIM;i++) n->misc_emb[i]=e*FR;
    for (int i=0;i<DIM*PLANES;i++) n->W_pol[i]=e*FR;
    for (int i=0;i<DIM*NWDL;i++)  n->W_val[i]=e*FR;
    #undef FR
}

#define CHESS_CKPT_MAGIC 0x43484e31u   // "CHN1"
// Save/load a checkpoint: a small header (magic + shape) for sanity, then every tensor
// fp32 in chess_net_params order. Returns 1 on success.
static int chess_net_save(ChessNet *n, const char *path) {
    FILE *f = fopen(path, "wb"); if (!f) return 0;
    uint32_t hdr[7] = { CHESS_CKPT_MAGIC, DIM, HIDDEN, NLAYERS, VOCAB, SEQ, (uint32_t)(HEADS) };
    fwrite(hdr, sizeof(uint32_t), 7, f);
    ParamRef pr[256]; int k = chess_net_params(n, pr);
    for (int i = 0; i < k; i++) fwrite(pr[i].p, 4, (size_t)pr[i].n, f);
    fclose(f); return 1;
}
static int chess_net_load(ChessNet *n, const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) return 0;
    uint32_t hdr[7];
    if (fread(hdr, sizeof(uint32_t), 7, f) != 7) { fclose(f); return 0; }
    if (hdr[0]!=CHESS_CKPT_MAGIC || hdr[1]!=DIM || hdr[2]!=HIDDEN || hdr[3]!=NLAYERS ||
        hdr[4]!=VOCAB || hdr[5]!=SEQ) { fclose(f); return 0; }   // shape mismatch
    ParamRef pr[256]; int k = chess_net_params(n, pr);
    for (int i = 0; i < k; i++)
        if (fread(pr[i].p, 4, (size_t)pr[i].n, f) != (size_t)pr[i].n) { fclose(f); return 0; }
    fclose(f); return 1;
}

// AdamW param registry: register every (weight, grad) pair, zero grads, and run a global
// grad-norm + clip + AdamW step. Mirrors train_chess.m's optimizer (same fp16-scaled
// pattern). gsc unscales the loss-scale; clip>0 applies global-norm clipping.
typedef struct { float *w, *g; AdamState a; int n; } NetParam;
static NetParam g_params[256]; static int g_nparams = 0;
static void reg(float *w, float *g, int n) {
    g_params[g_nparams].w=w; g_params[g_nparams].g=g;
    g_params[g_nparams].a=adam_alloc(n); g_params[g_nparams].n=n; g_nparams++;
}
static void chess_net_register(ChessNet *W, ChessNet *G) {
    ParamRef wp[256], gp[256];
    int k = chess_net_params(W, wp); chess_net_params(G, gp);
    for (int i = 0; i < k; i++) reg(wp[i].p, gp[i].p, wp[i].n);
}
static void grads_zero(void) {
    for (int i = 0; i < g_nparams; i++) memset(g_params[i].g, 0, (size_t)g_params[i].n*4);
}
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

#endif  // LILBRO_CHESS_NET_H
