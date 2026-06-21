// cpu_ops.h — CPU operations: RMSNorm, cross-entropy, Adam, embedding
#pragma once
#include "config.h"

static float *g_rms_tmp = NULL;

static void rmsnorm(float *out, const float *x, const float *w, int d, int S) {
    if (!g_rms_tmp) g_rms_tmp = (float*)malloc(S*4);
    float ss[S];
    memset(ss, 0, (size_t)S*4);
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, x+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vadd(g_rms_tmp, 1, ss, 1, ss, 1, (vDSP_Length)S);
    }
    float invd = 1.0f/d, eps=1e-5f;
    vDSP_vsmsa(ss, 1, &invd, &eps, ss, 1, (vDSP_Length)S);
    int n = S; vvrsqrtf(ss, ss, &n);
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, ss, 1, out+i*S, 1, (vDSP_Length)S);
        vDSP_vsmul(out+i*S, 1, &w[i], out+i*S, 1, (vDSP_Length)S);
    }
}

static void rmsnorm_bwd(float *dx, float *dw, const float *dy, const float *x, const float *w, int d, int S) {
    if (!g_rms_tmp) g_rms_tmp = (float*)malloc(S*4);
    float ss[S];
    memset(ss, 0, (size_t)S*4);
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, x+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vadd(g_rms_tmp, 1, ss, 1, ss, 1, (vDSP_Length)S);
    }
    float invd = 1.0f/d, eps=1e-5f;
    vDSP_vsmsa(ss, 1, &invd, &eps, ss, 1, (vDSP_Length)S);
    float rrms[S];
    int n = S; vvrsqrtf(rrms, ss, &n);
    float dot[S];
    memset(dot, 0, (size_t)S*4);
    for (int i=0; i<d; i++) {
        vDSP_vmul(dy+i*S, 1, x+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vsma(g_rms_tmp, 1, &w[i], dot, 1, dot, 1, (vDSP_Length)S);
    }
    vDSP_vmul(rrms, 1, rrms, 1, ss, 1, (vDSP_Length)S);
    vDSP_vsmul(ss, 1, &invd, ss, 1, (vDSP_Length)S);
    vDSP_vmul(dot, 1, ss, 1, dot, 1, (vDSP_Length)S);
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, dot, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vsub(g_rms_tmp, 1, dy+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vmul(g_rms_tmp, 1, rrms, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vsmul(g_rms_tmp, 1, &w[i], dx+i*S, 1, (vDSP_Length)S);
        vDSP_vmul(dy+i*S, 1, x+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vmul(g_rms_tmp, 1, rrms, 1, g_rms_tmp, 1, (vDSP_Length)S);
        float s; vDSP_sve(g_rms_tmp, 1, &s, (vDSP_Length)S);
        dw[i] += s;
    }
}

static void adam_update(float *w, const float *g, AdamState *s, int t, float lr, float b1, float b2, float eps, float wd) {
    float bc1 = 1.0f - powf(b1, t), bc2 = 1.0f - powf(b2, t);
    for (size_t i=0; i<s->n; i++) {
        s->m[i] = b1*s->m[i] + (1-b1)*g[i];
        s->v[i] = b2*s->v[i] + (1-b2)*g[i]*g[i];
        float mh = s->m[i]/bc1, vh = s->v[i]/bc2;
        w[i] -= lr * (mh / (sqrtf(vh) + eps) + wd * w[i]);
    }
}

// ===== Muon optimizer (CPU-side) =====
// Newton-Schulz orthogonalization in float64 (the only fp32 step is the final
// write-back into the fp32 weight). G is [rows,cols] row-major; the result
// overwrites Xout [rows,cols]. Internally transposes so the iterated matrix has
// rows<=cols.
//
// Each iteration is the quintic (DeepSeek-V4 §2.4 eq 28):
//   M_k = a·M + b·(M Mᵀ)M + c·(M Mᵀ)²M
// Two variants of the coefficient schedule:
//   - prior (Keller Jordan): all `steps` iterations on (3.4445,-4.7750,2.0315).
//   - v4 (DeepSeek-V4 hybrid, MUON_V4_STEPS=10): first 8 on that same triple for
//     rapid convergence, final 2 on (2,-1.5,0.5) to settle the singular values
//     exactly at 1. Selected by `v4_hybrid` (which forces the last-2-step stage).
#define MUON_V4_STEPS    10     // V4 hybrid Newton-Schulz iteration count (§2.4)
#define MUON_V4_RMS      0.18   // V4-Flash target update RMS γ (§4.2.1)
static void newton_schulz_f64(double *Xout, const double *G, int rows, int cols,
                              int steps, int v4_hybrid) {
    int transposed = rows > cols;
    int r = transposed ? cols : rows;   // iterated matrix is [r, c], r<=c
    int c = transposed ? rows : cols;
    double *X  = (double*)malloc((size_t)r*c*sizeof(double));
    if (!transposed)
        for (size_t i=0;i<(size_t)rows*cols;i++) X[i] = G[i];
    else
        for (int i=0;i<rows;i++) for (int j=0;j<cols;j++) X[j*rows + i] = G[i*cols + j];

    double nrm = 0; for (size_t i=0;i<(size_t)r*c;i++) nrm += X[i]*X[i];
    nrm = sqrt(nrm) + 1e-7;
    for (size_t i=0;i<(size_t)r*c;i++) X[i] /= nrm;

    double *A  = (double*)malloc((size_t)r*r*sizeof(double));
    double *AA = (double*)malloc((size_t)r*r*sizeof(double));
    double *B  = (double*)malloc((size_t)r*r*sizeof(double));
    double *BX = (double*)malloc((size_t)r*c*sizeof(double));
    for (int s=0;s<steps;s++) {
        // V4 hybrid: the final 2 of 10 steps switch to the stabilizing triple.
        double a, b, cc;
        if (v4_hybrid && s >= steps - 2) { a = 2.0; b = -1.5; cc = 0.5; }
        else                             { a = 3.4445; b = -4.7750; cc = 2.0315; }
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,   r, r, c, 1.0, X, c, X, c, 0.0, A,  r);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, r, r, r, 1.0, A, r, A, r, 0.0, AA, r);
        for (size_t i=0;i<(size_t)r*r;i++) B[i] = b*A[i] + cc*AA[i];
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, r, c, r, 1.0, B, r, X, c, 0.0, BX, c);
        for (size_t i=0;i<(size_t)r*c;i++) X[i] = a*X[i] + BX[i];
    }

    if (!transposed)
        for (size_t i=0;i<(size_t)rows*cols;i++) Xout[i] = X[i];
    else
        for (int i=0;i<rows;i++) for (int j=0;j<cols;j++) Xout[i*cols + j] = X[j*rows + i];
    free(X); free(A); free(AA); free(B); free(BX);
}

// One Muon update for a 2D weight w[rows,cols] (row-major). `buf` is the
// persistent SGD-momentum buffer. Both variants share the Nesterov momentum:
//   M = mu*buf + g ;  d = g + mu*M (nesterov) ;  u = NewtonSchulz(d)
// then differ only in the update rescale and weight decay (DeepSeek-V4 Alg. 1):
//   - v4=1:  scale = sqrt(max(n,m))·γ (γ=MUON_V4_RMS) ; decoupled wd:
//            w = w·(1 - lr·wd) - lr·u·scale          [§2.4 / §4.2.1]
//   - v4=0:  scale = max(1, n/m)^0.5 ; no wd (the prior Keller-Jordan path).
// This is the only difference between the prior and V4 Muon, so --muon-variant
// is a clean one-variable comparison.
static void muon_update(float *w, const float *g, float *buf, int rows, int cols,
                        float lr, float mu, int nesterov, int v4, float wd) {
    size_t n = (size_t)rows*cols;
    int steps = v4 ? MUON_V4_STEPS : 5;
    double *d = (double*)malloc(n*sizeof(double));
    double *u = (double*)malloc(n*sizeof(double));
    for (size_t i=0;i<n;i++) {
        buf[i] = mu*buf[i] + g[i];
        d[i] = nesterov ? ((double)g[i] + (double)mu*buf[i]) : (double)buf[i];
    }
    newton_schulz_f64(u, d, rows, cols, steps, v4);
    double scale = v4 ? sqrt((double)(rows > cols ? rows : cols)) * MUON_V4_RMS
                      : sqrt(fmax(1.0, (double)rows/(double)cols));
    if (v4 && wd > 0.0f)
        for (size_t i=0;i<n;i++)
            w[i] = (float)((double)w[i]*(1.0 - (double)lr*wd) - (double)lr*u[i]*scale);
    else
        for (size_t i=0;i<n;i++) w[i] -= (float)((double)lr * u[i] * scale);
    free(d); free(u);
}

// Cross-entropy loss: operates on logits[V, S] column-major (each column = one token)
// Avoids transposing by using a per-token temp buffer
static float cross_entropy_loss(float *dlogits, const float *logits, const uint16_t *targets, int V, int S) {
    float *col = (float*)malloc(V * 4);  // single column buffer
    float total_loss = 0;
    float invS = 1.0f / S;
    for (int t = 0; t < S; t++) {
        // Gather column t: logits[v, t] = logits[v*S + t], stride=S
        cblas_scopy(V, logits + t, S, col, 1);
        // Softmax
        float maxv; vDSP_maxv(col, 1, &maxv, (vDSP_Length)V);
        float neg_max = -maxv;
        vDSP_vsadd(col, 1, &neg_max, col, 1, (vDSP_Length)V);
        int n = V; vvexpf(col, col, &n);
        float sum; vDSP_sve(col, 1, &sum, (vDSP_Length)V);
        float inv_sum = 1.0f / sum;
        vDSP_vsmul(col, 1, &inv_sum, col, 1, (vDSP_Length)V);
        // Loss + gradient
        int tgt = targets[t];
        total_loss -= logf(col[tgt] + 1e-10f);
        col[tgt] -= 1.0f;
        vDSP_vsmul(col, 1, &invS, col, 1, (vDSP_Length)V);
        // Scatter back: dlogits[v*S + t] = col[v]
        cblas_scopy(V, col, 1, dlogits + t, S);
    }
    free(col);
    return total_loss / S;
}

// Cross-entropy loss WITHOUT the gradient (validation). Same softmax/CE as
// cross_entropy_loss, but: (a) no dlogits scatter, (b) targets are signed compact
// ids and any target < 0 (a token absent from the train-built compact vocab, so
// it has no LM-head row) is skipped and excluded from the denominator. Returns
// the mean loss over the *scoreable* tokens and reports that count via n_valid.
static float cross_entropy_loss_only(const float *logits, const int *targets,
                                     int V, int S, int *n_valid) {
    float *col = (float*)malloc(V * 4);
    double total_loss = 0; int nv = 0;
    for (int t = 0; t < S; t++) {
        int tgt = targets[t];
        if (tgt < 0) continue;                 // unscoreable token — skip
        cblas_scopy(V, logits + t, S, col, 1);
        float maxv; vDSP_maxv(col, 1, &maxv, (vDSP_Length)V);
        float neg_max = -maxv;
        vDSP_vsadd(col, 1, &neg_max, col, 1, (vDSP_Length)V);
        int n = V; vvexpf(col, col, &n);
        float sum; vDSP_sve(col, 1, &sum, (vDSP_Length)V);
        total_loss -= logf(col[tgt] / sum + 1e-10f);
        nv++;
    }
    free(col);
    if (n_valid) *n_valid = nv;
    return nv > 0 ? (float)(total_loss / nv) : 0.0f;
}

// Vocab compaction: build mapping from full 32K vocab to compact vocab
typedef struct {
    int compact_vocab;          // number of active tokens
    int *full_to_compact;       // [VOCAB] → compact id (-1 if unused)
    int *compact_to_full;       // [compact_vocab] → full vocab id
} VocabMap;

static VocabMap vocab_map_build(const uint16_t *data, size_t n_tokens, int full_vocab) {
    VocabMap vm;
    vm.full_to_compact = (int*)malloc(full_vocab * sizeof(int));
    memset(vm.full_to_compact, -1, full_vocab * sizeof(int));
    // Scan for used tokens
    for (size_t i = 0; i < n_tokens; i++) {
        vm.full_to_compact[data[i]] = 0;  // mark as used
    }
    // Assign compact IDs
    int cid = 0;
    for (int v = 0; v < full_vocab; v++) {
        if (vm.full_to_compact[v] == 0)
            vm.full_to_compact[v] = cid++;
        else
            vm.full_to_compact[v] = -1;
    }
    vm.compact_vocab = cid;
    vm.compact_to_full = (int*)malloc(cid * sizeof(int));
    for (int v = 0; v < full_vocab; v++) {
        if (vm.full_to_compact[v] >= 0)
            vm.compact_to_full[vm.full_to_compact[v]] = v;
    }
    return vm;
}

// Create compact embedding from full embedding
static float *vocab_compact_embed(const float *full_embed, const VocabMap *vm, int dim) {
    float *ce = (float*)malloc((size_t)vm->compact_vocab * dim * 4);
    for (int c = 0; c < vm->compact_vocab; c++)
        memcpy(ce + c*dim, full_embed + vm->compact_to_full[c]*dim, dim*4);
    return ce;
}

// Scatter compact embed gradients back to full embed
static void vocab_scatter_grads(float *full_gembed, const float *compact_gembed, const VocabMap *vm, int dim) {
    for (int c = 0; c < vm->compact_vocab; c++) {
        int fv = vm->compact_to_full[c];
        for (int d = 0; d < dim; d++)
            full_gembed[fv*dim + d] += compact_gembed[c*dim + d];
    }
}

// Update full embed from compact embed (after adam)
static void vocab_update_full(float *full_embed, const float *compact_embed, const VocabMap *vm, int dim) {
    for (int c = 0; c < vm->compact_vocab; c++)
        memcpy(full_embed + vm->compact_to_full[c]*dim, compact_embed + c*dim, dim*4);
}

static void vocab_compact_embed_into(float *compact_embed, const float *full_embed, const VocabMap *vm, int dim) {
    for (int c = 0; c < vm->compact_vocab; c++)
        memcpy(compact_embed + c*dim, full_embed + vm->compact_to_full[c]*dim, dim*4);
}

static void vocab_scatter_scale_active_grads(float *full_gembed, const float *compact_gembed,
                                             const VocabMap *vm, int dim, float scale) {
    for (int c = 0; c < vm->compact_vocab; c++) {
        float *fg = full_gembed + (size_t)vm->compact_to_full[c]*dim;
        const float *cg = compact_gembed + (size_t)c*dim;
        vDSP_vadd(fg, 1, cg, 1, fg, 1, (vDSP_Length)dim);
        vDSP_vsmul(fg, 1, &scale, fg, 1, (vDSP_Length)dim);
    }
}

static float vocab_active_dotpr(const float *full_gembed, const VocabMap *vm, int dim) {
    float acc = 0, s;
    for (int c = 0; c < vm->compact_vocab; c++) {
        const float *fg = full_gembed + (size_t)vm->compact_to_full[c]*dim;
        vDSP_dotpr(fg, 1, fg, 1, &s, (vDSP_Length)dim);
        acc += s;
    }
    return acc;
}

static void vocab_active_vsmul(float *full_gembed, const VocabMap *vm, int dim, float scale) {
    for (int c = 0; c < vm->compact_vocab; c++) {
        float *fg = full_gembed + (size_t)vm->compact_to_full[c]*dim;
        vDSP_vsmul(fg, 1, &scale, fg, 1, (vDSP_Length)dim);
    }
}

static void vocab_zero_active_grads(float *full_gembed, float *compact_gembed, const VocabMap *vm, int dim) {
    for (int c = 0; c < vm->compact_vocab; c++)
        memset(full_gembed + (size_t)vm->compact_to_full[c]*dim, 0, (size_t)dim*4);
    memset(compact_gembed, 0, (size_t)vm->compact_vocab*dim*4);
}

static void adam_update_vocab_active(float *full_embed, float *compact_embed, const float *full_gembed,
                                     AdamState *aembed, const VocabMap *vm, int dim, int t,
                                     float lr, float b1, float b2, float eps, float wd) {
    float bc1 = 1.0f - powf(b1, t), bc2 = 1.0f - powf(b2, t);
    for (int c = 0; c < vm->compact_vocab; c++) {
        int fv = vm->compact_to_full[c];
        float *w = full_embed + (size_t)fv*dim;
        const float *g = full_gembed + (size_t)fv*dim;
        float *m = aembed->m + (size_t)fv*dim;
        float *v = aembed->v + (size_t)fv*dim;
        float *cw = compact_embed + (size_t)c*dim;
        for (int d = 0; d < dim; d++) {
            m[d] = b1*m[d] + (1-b1)*g[d];
            v[d] = b2*v[d] + (1-b2)*g[d]*g[d];
            float mh = m[d]/bc1, vh = v[d]/bc2;
            w[d] -= lr * (mh / (sqrtf(vh) + eps) + wd * w[d]);
            cw[d] = w[d];
        }
    }
}

static void embed_lookup(float *x, const float *embed, const uint16_t *tokens, int dim, int seq) {
    for (int t = 0; t < seq; t++) {
        int tok = tokens[t];
        for (int d = 0; d < dim; d++)
            x[d*seq + t] = embed[tok*dim + d];
    }
}

static void embed_backward(float *d_embed, const float *dx, const uint16_t *tokens, int dim, int seq) {
    for (int t = 0; t < seq; t++) {
        int tok = tokens[t];
        for (int d = 0; d < dim; d++)
            d_embed[tok*dim + d] += dx[d*seq + t];
    }
}

// ===== Attention sink (issue #8): learnable per-head sink logit in the softmax
// denominator. When ATTN_SINK is on, the whole attention core (scores, softmax,
// scores@V) runs on CPU here — Q/K/V are the RoPE'd entries from the SDPA kernel.
// This lands the softmax on the CPU (the documented mask+softmax decomposition)
// and absorbs the sink's effect on the softmax normalizer, which the ANE backward
// kernels cannot. GQA: q-head h reads kv-head (h % KV_HEADS); causal.
// FD-verified fwd+bwd (training/training_dynamic/test_attn_sink derivation). =====
// Unified CPU attention core for the V4 knobs that the ANE softmax kernels can't
// express (issue #8 sink + issue #7 QK-norm). Runs the score/softmax/·V on CPU
// from the kernel's own (post-RoPE) Q/K/V. All knobs optional via NULL pointers:
//   sink_h[HEADS]   — per-head learnable sink logit in the softmax denominator
//   gq[HD], gk[HD]  — per-dim RMSNorm gains applied to each (head,pos) Q/K vector
//                     over head_dim, just before the scores (post-RoPE). NULL = off.
// With every pointer NULL this is exactly the baseline causal attention.
static void attn_cpu_forward(float *attn_out, const float *Q, const float *K, const float *V,
                             const float *sink_h, const float *gq, const float *gk, int seq) {
    float scale = 1.0f/sqrtf((float)HD);
    for (int h=0; h<HEADS; h++) {
        int kvh = h % KV_HEADS;
        int have_sink = (sink_h != NULL);
        float s = have_sink ? sink_h[h] : 0.0f;
        for (int q=0; q<seq; q++) {
            float qn[HD];
            if (gq) { float ms=0; for(int d=0;d<HD;d++){float v=Q[(h*HD+d)*seq+q]; ms+=v*v;}
                      float r=sqrtf(ms/HD+NORM_EPS); for(int d=0;d<HD;d++) qn[d]=Q[(h*HD+d)*seq+q]/r*gq[d]; }
            else    { for(int d=0;d<HD;d++) qn[d]=Q[(h*HD+d)*seq+q]; }
            float sc[SEQ]; float m = have_sink ? s : -1e30f;   // sink logit joins the max
            for (int j=0; j<=q; j++) {
                float dot=0;
                if (gk) { float ms=0; for(int d=0;d<HD;d++){float v=K[(kvh*HD+d)*seq+j]; ms+=v*v;}
                          float r=sqrtf(ms/HD+NORM_EPS); for(int d=0;d<HD;d++) dot+=qn[d]*(K[(kvh*HD+d)*seq+j]/r*gk[d]); }
                else    { for(int d=0;d<HD;d++) dot+=qn[d]*K[(kvh*HD+d)*seq+j]; }
                sc[j]=dot*scale; if (sc[j]>m) m=sc[j];
            }
            float Z=0; for (int j=0;j<=q;j++){ sc[j]=expf(sc[j]-m); Z+=sc[j]; }
            float inv = 1.0f/(Z + (have_sink?expf(s-m):0.0f));  // +exp(sink) in the denominator
            for (int d=0; d<HD; d++) {
                float acc=0;
                for (int j=0;j<=q;j++) acc += sc[j]*V[(kvh*HD+d)*seq+j];
                attn_out[(h*HD+d)*seq+q] = acc*inv;
            }
        }
    }
}
// Backward for attn_cpu_forward. da[Q_DIM,seq]=dL/dattn_out -> dQ[Q_DIM], dK[KV_DIM],
// dV[KV_DIM] (GQA-reduced), and (when enabled) dsink[HEADS], dgq[HD], dgk[HD] — all
// accumulated. dQ/dK are w.r.t. the kernel's post-RoPE Q/K, so the caller's
// rope_backward_inplace still applies unchanged. Recomputes the softmax (no saved
// probs). The RMSNorm VJP: dL/dx_i = g_i·dy_i/r − x_i·c/(HD·r³), c=Σ_d g_d·dy_d·x_d;
// dL/dg_d += dy_d·x_d/r (dy = grad w.r.t. the normalized vector).
static void attn_cpu_backward(const float *da, const float *Q, const float *K, const float *V,
                              const float *sink_h, const float *gq, const float *gk,
                              float *dQ, float *dK, float *dV, float *dsink,
                              float *dgq, float *dgk, int seq) {
    float scale=1.0f/sqrtf((float)HD);
    memset(dQ,0,Q_DIM*seq*4); memset(dK,0,KV_DIM*seq*4); memset(dV,0,KV_DIM*seq*4);
    float *Qn=(float*)malloc(Q_DIM*seq*4), *Kn=(float*)malloc(KV_DIM*seq*4);
    float *rq=(float*)malloc(HEADS*seq*4), *rk=(float*)malloc(KV_HEADS*seq*4);
    float *dQn=(float*)calloc(Q_DIM*seq,4), *dKn=(float*)calloc(KV_DIM*seq,4);
    // Precompute normalized Q,K (or alias) plus the per-(head,pos) rms for the VJP.
    for (int h=0;h<HEADS;h++) for(int q=0;q<seq;q++){
        if (gq){ float ms=0; for(int d=0;d<HD;d++){float v=Q[(h*HD+d)*seq+q]; ms+=v*v;}
                 float r=sqrtf(ms/HD+NORM_EPS); rq[h*seq+q]=r;
                 for(int d=0;d<HD;d++) Qn[(h*HD+d)*seq+q]=Q[(h*HD+d)*seq+q]/r*gq[d]; }
        else { for(int d=0;d<HD;d++) Qn[(h*HD+d)*seq+q]=Q[(h*HD+d)*seq+q]; }
    }
    for (int kvh=0;kvh<KV_HEADS;kvh++) for(int j=0;j<seq;j++){
        if (gk){ float ms=0; for(int d=0;d<HD;d++){float v=K[(kvh*HD+d)*seq+j]; ms+=v*v;}
                 float r=sqrtf(ms/HD+NORM_EPS); rk[kvh*seq+j]=r;
                 for(int d=0;d<HD;d++) Kn[(kvh*HD+d)*seq+j]=K[(kvh*HD+d)*seq+j]/r*gk[d]; }
        else { for(int d=0;d<HD;d++) Kn[(kvh*HD+d)*seq+j]=K[(kvh*HD+d)*seq+j]; }
    }
    for (int h=0;h<HEADS;h++) {
        int kvh=h%KV_HEADS; int have_sink=(sink_h!=NULL); float s=have_sink?sink_h[h]:0.0f; float dsh=0;
        for (int q=0;q<seq;q++){
            float sc[SEQ]; float m=have_sink?s:-1e30f;
            for(int j=0;j<=q;j++){ float dot=0; for(int d=0;d<HD;d++) dot+=Qn[(h*HD+d)*seq+q]*Kn[(kvh*HD+d)*seq+j];
                                   sc[j]=dot*scale; if(sc[j]>m)m=sc[j]; }
            float Z=0; for(int j=0;j<=q;j++){ sc[j]=expf(sc[j]-m); Z+=sc[j]; }
            float esink=have_sink?expf(s-m):0.0f, inv=1.0f/(Z+esink);
            float p[SEQ]; for(int j=0;j<=q;j++) p[j]=sc[j]*inv;
            float psink=esink*inv;
            float dp[SEQ];
            for(int j=0;j<=q;j++){ float acc=0; for(int d=0;d<HD;d++){ float dad=da[(h*HD+d)*seq+q];
                acc+=dad*V[(kvh*HD+d)*seq+j]; dV[(kvh*HD+d)*seq+j]+=p[j]*dad; } dp[j]=acc; }
            float g=0; for(int j=0;j<=q;j++) g+=p[j]*dp[j];
            if(have_sink) dsh += -psink*g;
            for(int j=0;j<=q;j++){ float dscore=p[j]*(dp[j]-g)*scale;
                for(int d=0;d<HD;d++){ dQn[(h*HD+d)*seq+q]+=dscore*Kn[(kvh*HD+d)*seq+j];
                                       dKn[(kvh*HD+d)*seq+j]+=dscore*Qn[(h*HD+d)*seq+q]; } }
        }
        if(have_sink) dsink[h]+=dsh;
    }
    // RMSNorm VJP back to the post-RoPE Q/K (or straight copy when QK-norm is off).
    if (gq){ for(int h=0;h<HEADS;h++) for(int q=0;q<seq;q++){ float r=rq[h*seq+q]; float c=0;
        for(int d=0;d<HD;d++) c+=gq[d]*dQn[(h*HD+d)*seq+q]*Q[(h*HD+d)*seq+q];
        for(int d=0;d<HD;d++){ float xd=Q[(h*HD+d)*seq+q], dy=dQn[(h*HD+d)*seq+q];
            dQ[(h*HD+d)*seq+q]= gq[d]*dy/r - xd*c/(HD*r*r*r); dgq[d]+= dy*xd/r; } } }
    else memcpy(dQ,dQn,Q_DIM*seq*4);
    if (gk){ for(int kvh=0;kvh<KV_HEADS;kvh++) for(int j=0;j<seq;j++){ float r=rk[kvh*seq+j]; float c=0;
        for(int d=0;d<HD;d++) c+=gk[d]*dKn[(kvh*HD+d)*seq+j]*K[(kvh*HD+d)*seq+j];
        for(int d=0;d<HD;d++){ float xd=K[(kvh*HD+d)*seq+j], dy=dKn[(kvh*HD+d)*seq+j];
            dK[(kvh*HD+d)*seq+j]= gk[d]*dy/r - xd*c/(HD*r*r*r); dgk[d]+= dy*xd/r; } } }
    else memcpy(dK,dKn,KV_DIM*seq*4);
    free(Qn);free(Kn);free(rq);free(rk);free(dQn);free(dKn);
}

#if MTP_DEPTH > 0
// ===== Multi-Token Prediction (MTP) — CPU-first (issue #6) =====
// A faithful CPU implementation of an extra transformer block per MTP depth plus
// the V4 MTP glue (rms_h/rms_e, concat, proj, per-depth head). CPU-first per ADR
// 0001 (lowest new-kernel risk to reach the overfit-green gate; the Sk<SEQ
// truncation fights the fixed-shape MIL kernels). Matmuls use cblas; attention
// reuses attn_cpu_forward/backward (no sink/qk-norm). All [C,S] channel-major.
// FD-verified end-to-end (test_mtp.c): grads of the combined MTP loss w.r.t. all
// MTP params + the shared trunk/embed match central differences to ~8e-5.

static void mtp_mm(float *o, const float *W, const float *x, int O, int IN, int S) {     // o = W@x
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,O,S,IN,1.0f,W,IN,x,S,0.0f,o,S); }
static void mtp_mmWT(float *o, const float *W, const float *dy, int O, int IN, int S) {   // o = Wᵀ@dy
    cblas_sgemm(CblasRowMajor,CblasTrans,CblasNoTrans,IN,S,O,1.0f,W,IN,dy,S,0.0f,o,S); }
static void mtp_dWacc(float *dW, const float *dy, const float *x, int O, int IN, int S) { // dW += dy@xᵀ
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,O,IN,S,1.0f,dy,S,x,S,1.0f,dW,IN); }

// Interleaved RoPE on [C,S] (C=nheads*HD), partial-aware (matches the ANE kernel
// and rope_backward_inplace). inv=0 forward rotation, inv=1 inverse (transpose).
static void rope_apply(float *x, int C, int S, int inv) {
    int nh = C/HD, norot = HD - ROPE_ROTARY_EFF;
    for (int h=0; h<nh; h++) for (int i=0; i<HD/2; i++) {
        if (2*i < norot) continue;
        float freq = 1.0f/powf(10000.0f, 2.0f*i/(float)HD);
        for (int p=0; p<S; p++) {
            float th=p*freq, c=cosf(th), s=sinf(th);
            int a=(h*HD+2*i)*S+p, b=(h*HD+2*i+1)*S+p; float v0=x[a], v1=x[b];
            if (!inv) { x[a]=v0*c - v1*s; x[b]=v0*s + v1*c; }
            else      { x[a]=v0*c + v1*s; x[b]=-v0*s + v1*c; }
        }
    }
}

// One transformer block, identical to the main block (mlx_ref _block): pre-norm
// attention with residual scale res_alpha, then pre-norm SwiGLU FFN.
typedef struct { float Wq[Q_DIM*DIM],Wk[KV_DIM*DIM],Wv[KV_DIM*DIM],Wo[DIM*Q_DIM];
    float W1[HIDDEN*DIM],W2[DIM*HIDDEN],W3[HIDDEN*DIM],rms_att[DIM],rms_ffn[DIM]; } MtpBlock;
typedef struct { float xin[DIM*SEQ],h1[DIM*SEQ],Q[Q_DIM*SEQ],K[KV_DIM*SEQ],V[KV_DIM*SEQ],
    attn[Q_DIM*SEQ],x2[DIM*SEQ],h2[DIM*SEQ],g1[HIDDEN*SEQ],g3[HIDDEN*SEQ],
    silu[HIDDEN*SEQ],gate[HIDDEN*SEQ]; } MtpBlockAct;

static void mtp_block_fwd(float *out, const float *x, const MtpBlock *b, MtpBlockAct *a,
                          int S, float res_alpha) {
    memcpy(a->xin, x, DIM*S*4);
    rmsnorm(a->h1, x, b->rms_att, DIM, S);
    mtp_mm(a->Q,b->Wq,a->h1,Q_DIM,DIM,S); mtp_mm(a->K,b->Wk,a->h1,KV_DIM,DIM,S); mtp_mm(a->V,b->Wv,a->h1,KV_DIM,DIM,S);
    rope_apply(a->Q,Q_DIM,S,0); rope_apply(a->K,KV_DIM,S,0);
    attn_cpu_forward(a->attn, a->Q, a->K, a->V, NULL, NULL, NULL, S);
    float *o = (float*)malloc(DIM*S*4);
    mtp_mm(o,b->Wo,a->attn,DIM,Q_DIM,S);
    for (int i=0;i<DIM*S;i++) a->x2[i]=x[i]+res_alpha*o[i];
    rmsnorm(a->h2, a->x2, b->rms_ffn, DIM, S);
    mtp_mm(a->g1,b->W1,a->h2,HIDDEN,DIM,S); mtp_mm(a->g3,b->W3,a->h2,HIDDEN,DIM,S);
    for (int i=0;i<HIDDEN*S;i++){ float s=1.0f/(1.0f+expf(-a->g1[i])); a->silu[i]=a->g1[i]*s; a->gate[i]=a->silu[i]*a->g3[i]; }
    float *ff = (float*)malloc(DIM*S*4);
    mtp_mm(ff,b->W2,a->gate,DIM,HIDDEN,S);
    for (int i=0;i<DIM*S;i++) out[i]=a->x2[i]+res_alpha*ff[i];
    free(o); free(ff);
}

// dout -> dx (accumulated +=), weight grads accumulated into gb.
static void mtp_block_bwd(float *dx, const float *dout, const MtpBlock *b, const MtpBlockAct *a,
                          MtpBlock *gb, int S, float res_alpha) {
    size_t sd=DIM*S, sh=HIDDEN*S, sq=Q_DIM*S, sk=KV_DIM*S;
    float *dx2=(float*)malloc(sd*4),*dff=(float*)malloc(sd*4),*dgate=(float*)malloc(sh*4);
    float *dsilu=(float*)malloc(sh*4),*dg1=(float*)malloc(sh*4),*dg3=(float*)malloc(sh*4);
    float *dh2=(float*)malloc(sd*4),*dattn=(float*)malloc(sq*4),*dQ=(float*)malloc(sq*4);
    float *dK=(float*)malloc(sk*4),*dV=(float*)malloc(sk*4),*dh1=(float*)malloc(sd*4);
    float *dxa=(float*)calloc(sd,4),*tmp=(float*)malloc(sd*4),*dxin=(float*)malloc(sd*4),*do_=(float*)malloc(sd*4);
    // out = x2 + ra*ff
    for (size_t i=0;i<sd;i++){ dx2[i]=dout[i]; dff[i]=res_alpha*dout[i]; }
    mtp_dWacc(gb->W2,dff,a->gate,DIM,HIDDEN,S); mtp_mmWT(dgate,b->W2,dff,DIM,HIDDEN,S);
    for (size_t i=0;i<sh;i++){ float s=1.0f/(1.0f+expf(-a->g1[i]));
        dsilu[i]=dgate[i]*a->g3[i]; dg3[i]=dgate[i]*a->silu[i];
        dg1[i]=dsilu[i]*(s*(1.0f+a->g1[i]*(1.0f-s))); }
    mtp_dWacc(gb->W1,dg1,a->h2,HIDDEN,DIM,S); mtp_dWacc(gb->W3,dg3,a->h2,HIDDEN,DIM,S);
    mtp_mmWT(dh2,b->W1,dg1,HIDDEN,DIM,S); mtp_mmWT(tmp,b->W3,dg3,HIDDEN,DIM,S);
    for (size_t i=0;i<sd;i++) dh2[i]+=tmp[i];
    rmsnorm_bwd(dxa,gb->rms_ffn,dh2,a->x2,b->rms_ffn,DIM,S);
    for (size_t i=0;i<sd;i++) dx2[i]+=dxa[i];
    // x2 = xin + ra*o
    for (size_t i=0;i<sd;i++){ dxin[i]=dx2[i]; do_[i]=res_alpha*dx2[i]; }
    mtp_dWacc(gb->Wo,do_,a->attn,DIM,Q_DIM,S); mtp_mmWT(dattn,b->Wo,do_,DIM,Q_DIM,S);
    attn_cpu_backward(dattn,a->Q,a->K,a->V,NULL,NULL,NULL,dQ,dK,dV,NULL,NULL,NULL,S);
    rope_apply(dQ,Q_DIM,S,1); rope_apply(dK,KV_DIM,S,1);
    mtp_dWacc(gb->Wq,dQ,a->h1,Q_DIM,DIM,S); mtp_dWacc(gb->Wk,dK,a->h1,KV_DIM,DIM,S); mtp_dWacc(gb->Wv,dV,a->h1,KV_DIM,DIM,S);
    mtp_mmWT(dh1,b->Wq,dQ,Q_DIM,DIM,S);
    mtp_mmWT(tmp,b->Wk,dK,KV_DIM,DIM,S); for (size_t i=0;i<sd;i++) dh1[i]+=tmp[i];
    mtp_mmWT(tmp,b->Wv,dV,KV_DIM,DIM,S); for (size_t i=0;i<sd;i++) dh1[i]+=tmp[i];
    memset(dxa,0,sd*4); rmsnorm_bwd(dxa,gb->rms_att,dh1,a->xin,b->rms_att,DIM,S);
    for (size_t i=0;i<sd;i++) dx[i]+= dxin[i]+dxa[i];
    free(dx2);free(dff);free(dgate);free(dsilu);free(dg1);free(dg3);free(dh2);
    free(dattn);free(dQ);free(dK);free(dV);free(dh1);free(dxa);free(tmp);free(dxin);free(do_);
}
#endif // MTP_DEPTH > 0

// RoPE backward (in-place): inverse rotation on dQ/dK gradients
// Data layout: [DIM, SEQ] channel-first, DIM = nheads * hd
static void rope_backward_inplace(float *dx, int seq, int dim, int hd) {
    int nheads = dim / hd;
    int norot = HD - ROPE_ROTARY_EFF;   // partial RoPE (#10): leading dims unrotated
    for (int h = 0; h < nheads; h++) {
        for (int i = 0; i < hd/2; i++) {
            if (2*i < norot) continue;  // identity rotation -> identity gradient
            float freq = 1.0f / powf(10000.0f, 2.0f * i / (float)hd);
            for (int p = 0; p < seq; p++) {
                float theta = p * freq;
                float cos_t = cosf(theta), sin_t = sinf(theta);
                int idx0 = (h * hd + 2 * i) * seq + p;
                int idx1 = (h * hd + 2 * i + 1) * seq + p;
                float v0 = dx[idx0], v1 = dx[idx1];
                dx[idx0] = v0 * cos_t + v1 * sin_t;
                dx[idx1] = -v0 * sin_t + v1 * cos_t;
            }
        }
    }
}
