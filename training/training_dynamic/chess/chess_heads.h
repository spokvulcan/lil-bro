// chess_heads.h — the NEW chess policy/value heads + 2D posenc, pure C / fp32.
//
// Build-step 2 of the chess-RL-on-ANE build order (ADR 0005, issue #16). This is
// the SINGLE source of truth for the head math: the finite-difference gate
// (chess/test_heads_cpu.c) and the ANE trainer (train_chess.m) both call these,
// so the backward proven analytically by FD is the exact backward the gate runs.
//
// Everything here is the CPU "head" half of the CPU/ANE split (ADR 0004 /
// [[ane-resident-training-cpu-floor]]): the trunk matmuls run on the ANE in fp16;
// projection + softmax + cross-entropy + their backward run on the CPU in fp32,
// mirroring the LM classifier head (train.m cross_entropy_loss). The WDL value
// head is 3-way, so it lives on the CPU floor anyway (3 is not a mult-of-32 ANE
// packed dim — [[ane-dispatch-bound-fusion-lever]]).
//
// Layout convention (matches cpu_ops.h / attn_cpu): activations are CHANNEL-MAJOR
// [dim, seq] => x[d*seq + p]. Weights are [IN, OUT] row-major => W[i*OUT + o], so
// the ANE forward matmul gen_dyn_matmul_mil(IN,OUT,seq) consumes them transpose-
// free. Dimensions are PARAMETERS (not macros) so the FD test can exercise the
// identical code at tiny sizes.
//
// Pure C, zero deps (math.h/string.h/stdint.h only) — the project law. No ANE,
// no Accelerate: callable from the dependency-free FD test.
#ifndef LILBRO_CHESS_HEADS_H
#define LILBRO_CHESS_HEADS_H

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// =============================================================================
// 2D positional encoding (ADR 0005 decision 10): learned rank + file embeddings
// summed at the input, RoPE OFF. Board geometry is LERF (chess.h): for sequence
// position p in [0, nboard) the position IS the square, so rank = p>>3, file = p&7
// (8-wide board). The remaining positions p in [nboard, seq) are the state tokens
// (stm / castling / ep / clocks) + padding; they have no board rank/file, so they
// get their own learned per-offset embedding (misc table, seq-nboard rows).
//
// rank_emb : [8,   dim]   file_emb : [8,   dim]   misc_emb : [seq-nboard, dim]
// All summed (+=) into x[dim, seq]; call AFTER the token embedding lookup.
// =============================================================================
static inline void chess_posenc_forward(float *x, const float *rank_emb,
                                        const float *file_emb, const float *misc_emb,
                                        int dim, int seq, int nboard) {
    for (int p = 0; p < seq; p++) {
        if (p < nboard) {
            int rank = p >> 3, file = p & 7;
            for (int d = 0; d < dim; d++)
                x[d*seq + p] += rank_emb[rank*dim + d] + file_emb[file*dim + d];
        } else {
            int m = p - nboard;
            for (int d = 0; d < dim; d++)
                x[d*seq + p] += misc_emb[m*dim + d];
        }
    }
}

// Backward: scatter dx[dim,seq] into the three tables (accumulated +=). dy/dx for a
// sum is identity, so each table row collects the columns that read it.
static inline void chess_posenc_backward(const float *dx, float *d_rank,
                                         float *d_file, float *d_misc,
                                         int dim, int seq, int nboard) {
    for (int p = 0; p < seq; p++) {
        if (p < nboard) {
            int rank = p >> 3, file = p & 7;
            for (int d = 0; d < dim; d++) {
                d_rank[rank*dim + d] += dx[d*seq + p];
                d_file[file*dim + d] += dx[d*seq + p];
            }
        } else {
            int m = p - nboard;
            for (int d = 0; d < dim; d++)
                d_misc[m*dim + d] += dx[d*seq + p];
        }
    }
}

// =============================================================================
// Policy head (ADR 0005 decision 5): per-square 8x8x73 logits, legal-masked
// BEFORE softmax via an additive -inf bias on illegal moves (the same additive-
// bias trick as the causal mask). Reads the board-square columns 0..nboard-1 of
// the final hidden x[dim,seq]; W_pol is [dim, planes] ([IN,OUT]). The flat policy
// index is sq*planes + plane (chess.h move<->index map), N = nboard*planes.
//
// Masked softmax CE to a target distribution `target` (sums to 1 over legal):
//   logit_i = sum_d W_pol[d,pl] * x[d,sq]          (i = sq*planes+pl)
//   p_i = softmax over LEGAL i ; illegal p_i := 0  (illegal logit = -inf)
//   loss = -sum_i target_i * log p_i
// Gradient on the legal set: dlogit_i = p_i - target_i ; illegal dlogit_i = 0
// (the -inf bias is a constant, no gradient). Returns the scalar loss; if dx and
// dW_pol are non-NULL they are ACCUMULATED (+=) so the value head can add into the
// same dx. Pass dx=dW_pol=NULL for a forward-only loss (the FD numeric probe).
static inline float chess_policy_loss(const float *x, const float *W_pol,
                                      int dim, int seq, int nboard, int planes,
                                      const uint8_t *legal_mask, const float *target,
                                      float *dx, float *dW_pol) {
    int N = nboard * planes;
    // logits + masked-max in one pass over legal entries.
    float *logit = (float*)malloc((size_t)N * sizeof(float));
    float m = -INFINITY;
    for (int sq = 0; sq < nboard; sq++) {
        for (int pl = 0; pl < planes; pl++) {
            int i = sq*planes + pl;
            if (!legal_mask[i]) { logit[i] = -INFINITY; continue; }
            float acc = 0.0f;
            for (int d = 0; d < dim; d++) acc += W_pol[d*planes + pl] * x[d*seq + sq];
            logit[i] = acc;
            if (acc > m) m = acc;
        }
    }
    // softmax over legal + cross-entropy.
    float Z = 0.0f;
    float *p = (float*)malloc((size_t)N * sizeof(float));
    for (int i = 0; i < N; i++) {
        if (!legal_mask[i]) { p[i] = 0.0f; continue; }
        p[i] = expf(logit[i] - m);
        Z += p[i];
    }
    // Terminal position (no legal moves): nothing to predict — zero loss, zero grad.
    // Guards the masked softmax against Z==0/-inf before RL self-play (G2) feeds
    // checkmate/stalemate leaves; the G0 startpos always has legal moves.
    if (Z == 0.0f) { free(logit); free(p); return 0.0f; }
    float invZ = 1.0f / Z;
    float loss = 0.0f;
    for (int i = 0; i < N; i++) {
        if (!legal_mask[i]) continue;
        p[i] *= invZ;
        if (target[i] > 0.0f) loss -= target[i] * logf(p[i] + 1e-30f);
    }
    // gradient: dlogit_i = p_i - target_i (legal only).
    if (dx || dW_pol) {
        for (int sq = 0; sq < nboard; sq++) {
            for (int pl = 0; pl < planes; pl++) {
                int i = sq*planes + pl;
                if (!legal_mask[i]) continue;
                float g = p[i] - target[i];
                if (g == 0.0f) continue;
                for (int d = 0; d < dim; d++) {
                    if (dx)     dx[d*seq + sq]   += W_pol[d*planes + pl] * g;
                    if (dW_pol) dW_pol[d*planes + pl] += x[d*seq + sq]   * g;
                }
            }
        }
    }
    free(logit); free(p);
    return loss;
}

// =============================================================================
// Value head (ADR 0005 decision 5): a 3-way Win/Draw/Loss softmax from a pooled
// representation. Pool = mean over the nreal real tokens (padding excluded) of the
// final hidden; W_val is [dim, nwdl] ([IN,OUT]). WDL cross-entropy to the outcome
// target z (one-hot {W,D,L} for the gate; a soft distribution in general).
//   pooled_d = (1/nreal) sum_{p<nreal} x[d,p]
//   vlogit_k = sum_d W_val[d,k] * pooled_d ;  q = softmax(vlogit)
//   loss = -sum_k target_k log q_k
// dvlogit_k = q_k - target_k. dx[d,p<nreal] += (sum_k W_val[d,k] dvlogit_k)/nreal.
// Returns the scalar loss; dx / dW_val ACCUMULATED (+=) when non-NULL.
static inline float chess_value_loss(const float *x, const float *W_val,
                                     int dim, int seq, int nreal, int nwdl,
                                     const float *target, float *dx, float *dW_val) {
    float *pooled = (float*)malloc((size_t)dim * sizeof(float));
    float invn = 1.0f / (float)nreal;
    for (int d = 0; d < dim; d++) {
        float acc = 0.0f;
        for (int p = 0; p < nreal; p++) acc += x[d*seq + p];
        pooled[d] = acc * invn;
    }
    float *vlogit = (float*)malloc((size_t)nwdl * sizeof(float));
    float m = -INFINITY;
    for (int k = 0; k < nwdl; k++) {
        float acc = 0.0f;
        for (int d = 0; d < dim; d++) acc += W_val[d*nwdl + k] * pooled[d];
        vlogit[k] = acc;
        if (acc > m) m = acc;
    }
    float Z = 0.0f, *q = (float*)malloc((size_t)nwdl * sizeof(float));
    for (int k = 0; k < nwdl; k++) { q[k] = expf(vlogit[k] - m); Z += q[k]; }
    float invZ = 1.0f / Z, loss = 0.0f;
    for (int k = 0; k < nwdl; k++) {
        q[k] *= invZ;
        if (target[k] > 0.0f) loss -= target[k] * logf(q[k] + 1e-30f);
    }
    if (dx || dW_val) {
        // dpooled then broadcast (1/nreal) back to the real columns.
        for (int d = 0; d < dim; d++) {
            float dp = 0.0f;
            for (int k = 0; k < nwdl; k++) {
                float g = q[k] - target[k];
                if (dW_val) dW_val[d*nwdl + k] += pooled[d] * g;
                dp += W_val[d*nwdl + k] * g;
            }
            if (dx) { float v = dp * invn; for (int p = 0; p < nreal; p++) dx[d*seq + p] += v; }
        }
    }
    free(pooled); free(vlogit); free(q);
    return loss;
}

// =============================================================================
// L2 weight penalty (ADR 0005 decision 13: "small L2"). loss += l2 * sum(w^2);
// grad += 2*l2*w (accumulated). Returns the penalty so the trainer can report it
// separately from the two cross-entropies (which are what G0 drives to ~0).
static inline float chess_l2_penalty(const float *w, int n, float l2, float *grad) {
    if (l2 == 0.0f) return 0.0f;
    float s = 0.0f, two_l2 = 2.0f * l2;
    for (int i = 0; i < n; i++) { s += w[i]*w[i]; if (grad) grad[i] += two_l2 * w[i]; }
    return l2 * s;
}

#endif  // LILBRO_CHESS_HEADS_H
