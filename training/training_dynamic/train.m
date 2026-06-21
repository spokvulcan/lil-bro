// train.m — Dynamic weight ANE training (model-agnostic GQA support)
// Model selected at compile time via: make MODEL=qwen3_06b (or stories110m)
// Compile kernels ONCE at startup, update weights via IOSurface every step.
#include "mil_dynamic.h"
#include "cpu_ops.h"
#include "mhc.h"     // manifold-constrained hyper-connections (issue #11); inert at N_HC==1

// Dynamic kernel set per layer
typedef struct {
    Kern *sdpaFwd;     // QKV matmul + RoPE + GQA tile + SDPA (no Wo)
    Kern *woFwd;       // attn_out @ Wo^T → o_out (Q_DIM → DIM)
    Kern *ffnFused;    // W1,W3 + SiLU + W2 + residual (fused)
    Kern *ffnBwdW2t;   // dffn @ W2^T → dsilu_raw (DIM → HIDDEN)
    Kern *ffnBwdW13t;  // dh1@W1^T + dh3@W3^T → dx_ffn (HIDDEN → DIM)
    Kern *wotBwd;      // dx2 @ Wo → da (DIM → Q_DIM)
    Kern *sdpaBwd1;    // Q,K,V,da → dV_full,probs,dp (weight-free, has mask)
    Kern *sdpaBwd2;    // probs,dp,Q,K → dQ,dK_full (weight-free)
    Kern *qBwd;        // dq @ Wq → dx_q (Q_DIM → DIM)
    Kern *kvBwd;       // dk@Wk + dv@Wv → dx_kv (KV_DIM → DIM)
#if FUSE_QKVBWD
    Kern *qkvBwd;      // fused: dq@Wq + dk@Wk + dv@Wv → dx_attn (one eval, MHA-only)
#endif
} DynLayerKernels;

// Transpose W[rows,cols] → W^T[cols,rows] stored as [cols channels, rows spatial]
static void transpose_weight(float *dst, const float *src, int rows, int cols) {
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            dst[c * rows + r] = src[r * cols + c];
}

// ===== Compile all dynamic kernels (ONCE) =====
static bool compile_dynamic_kernels(DynLayerKernels *dk) {
    NSDictionary *mask_w = @{@"@model_path/weights/mask.bin": @{@"offset":@0, @"data":get_mask_blob()}};
    NSDictionary *sdpa_fwd_w = @{
        @"@model_path/weights/mask.bin": @{@"offset":@0, @"data":get_mask_blob()},
        @"@model_path/weights/rope_cos.bin": @{@"offset":@0, @"data":get_rope_cos_blob()},
        @"@model_path/weights/rope_sin.bin": @{@"offset":@0, @"data":get_rope_sin_blob()}
    };

    int sdpa_out_ch = Q_DIM + Q_DIM + KV_DIM + KV_DIM + DIM;

    // SDPA forward (no Wo): [1, DIM, 1, SDPA_FWD_SP] → [1, sdpa_out_ch, 1, SEQ]
    printf("  Compiling sdpaFwd (GQA)...\n");
    dk->sdpaFwd = compile_kern_mil_w(gen_sdpa_fwd_dynamic(), sdpa_fwd_w,
        DIM*SDPA_FWD_SP*2, sdpa_out_ch*SEQ*2);
    if (!dk->sdpaFwd) return false;

    // Wo forward: [1, Q_DIM, 1, SEQ+DIM] → [1, DIM, 1, SEQ]
    printf("  Compiling woFwd...\n");
#if WO_FUNCPARAM
    dk->woFwd = compile_kern_mil_2in(gen_wo_fwd_2in(),
        Q_DIM*SEQ*2, Q_DIM*DIM*2, DIM*SEQ*2);
#else
    dk->woFwd = compile_kern_mil_w(gen_wo_fwd_dynamic(), @{},
        Q_DIM*WO_FWD_SP*2, DIM*SEQ*2);
#endif
    if (!dk->woFwd) return false;

    // Fused FFN: [1, DIM, 1, FFN_FUSED_SP] → [1, DIM+3*HIDDEN, 1, SEQ]
    printf("  Compiling ffnFused...\n");
    int ffn_fused_och = DIM + 3*HIDDEN;
    dk->ffnFused = compile_kern_mil_w(gen_ffn_fused_dynamic(), @{},
        DIM*FFN_FUSED_SP*2, ffn_fused_och*SEQ*2);
    if (!dk->ffnFused) return false;

    // FFN backward W2^T: [1, DIM, 1, SEQ+HIDDEN] → [1, HIDDEN, 1, SEQ]
    printf("  Compiling ffnBwdW2t...\n");
#if W2T_FUNCPARAM
#if CONV_PROBE
    dk->ffnBwdW2t = compile_kern_mil_2in(gen_conv_2in(DIM, HIDDEN, SEQ),
        DIM*SEQ*2, DIM*HIDDEN*2, HIDDEN*SEQ*2);
#else
    dk->ffnBwdW2t = compile_kern_mil_2in(gen_matmul_2in(DIM, HIDDEN, SEQ),
        DIM*SEQ*2, DIM*HIDDEN*2, HIDDEN*SEQ*2);
#endif
#else
    dk->ffnBwdW2t = compile_kern_mil_w(gen_ffn_bwd_w2t_dynamic(), @{},
        DIM*FFN_BWD_W2T_SP*2, HIDDEN*SEQ*2);
#endif
    if (!dk->ffnBwdW2t) return false;

    // FFN backward W1^T+W3^T: [1, HIDDEN, 1, 2*SEQ+2*DIM] → [1, DIM, 1, SEQ]
    printf("  Compiling ffnBwdW13t...\n");
    dk->ffnBwdW13t = compile_kern_mil_w(gen_ffn_bwd_w13t_dynamic(), @{},
        HIDDEN*FFN_BWD_W13T_SP*2, DIM*SEQ*2);
    if (!dk->ffnBwdW13t) return false;

    // Wo^T backward: [1, DIM, 1, SEQ+Q_DIM] → [1, Q_DIM, 1, SEQ]
    printf("  Compiling wotBwd...\n");
#if CONV_DATAPATH
    dk->wotBwd = compile_kern_mil_2in(gen_conv_2in(DIM, Q_DIM, SEQ),
        DIM*SEQ*2, DIM*Q_DIM*2, Q_DIM*SEQ*2);
#elif CONV1IN == 2
    // PRD #26 Step B: conv with weight pre-transposed in staging -> zero in-MIL
    // transposes. SAME packed input surface (only the weight region's layout
    // differs, handled by stage_wot_bwd_weights_convB).
    dk->wotBwd = compile_kern_mil_w(gen_conv_1in_mil_B(DIM, Q_DIM, SEQ), @{},
        DIM*WOT_BWD_SP*2, Q_DIM*SEQ*2);
#elif CONV1IN
    // PRD #26 Step A single-input conv: SAME packed input surface + staging as
    // the matmul path (DIM*WOT_BWD_SP*2 in, Q_DIM*SEQ*2 out) — only the MIL swaps.
    dk->wotBwd = compile_kern_mil_w(gen_conv_1in_mil(DIM, Q_DIM, SEQ), @{},
        DIM*WOT_BWD_SP*2, Q_DIM*SEQ*2);
#else
    dk->wotBwd = compile_kern_mil_w(gen_wot_dynamic(), @{},
        DIM*WOT_BWD_SP*2, Q_DIM*SEQ*2);
#endif
    if (!dk->wotBwd) return false;

    // SDPA bwd1 (weight-free, has mask): [1, 4*Q_DIM, 1, SEQ] → [1, Q_DIM+2*SCORE_CH, 1, SEQ]
    printf("  Compiling sdpaBwd1 (GQA)...\n");
    dk->sdpaBwd1 = compile_kern_mil_w(gen_sdpa_bwd1_noweight(), mask_w,
        4*Q_DIM*SEQ*2, (Q_DIM+2*SCORE_CH)*SEQ*2);
    if (!dk->sdpaBwd1) return false;

    // SDPA bwd2 (weight-free): [1, 2*SCORE_CH+2*Q_DIM, 1, SEQ] → [1, 2*Q_DIM, 1, SEQ]
    printf("  Compiling sdpaBwd2 (GQA)...\n");
    dk->sdpaBwd2 = compile_kern_mil_w(gen_sdpa_bwd2(), @{},
        (2*SCORE_CH+2*Q_DIM)*SEQ*2, 2*Q_DIM*SEQ*2);
    if (!dk->sdpaBwd2) return false;

#if FUSE_QKVBWD
    // Fused QKV backward (MHA-only): one eval replaces qBwd + kvBwd.
    // [1, DIM, 1, 3*SEQ+3*DIM] → [1, DIM, 1, SEQ]
    printf("  Compiling qkvBwd (fused)...\n");
    dk->qkvBwd = compile_kern_mil_w(gen_qkv_bwd_fused_mil(), @{},
        DIM*QKV_BWD_SP*2, DIM*SEQ*2);
    if (!dk->qkvBwd) return false;
#else
    // Q backward: [1, Q_DIM, 1, SEQ+DIM] → [1, DIM, 1, SEQ]
    printf("  Compiling qBwd...\n");
#if CONV_DATAPATH
    dk->qBwd = compile_kern_mil_2in(gen_conv_2in(Q_DIM, DIM, SEQ),
        Q_DIM*SEQ*2, Q_DIM*DIM*2, DIM*SEQ*2);
#else
    dk->qBwd = compile_kern_mil_w(gen_q_bwd_dynamic(), @{},
        Q_DIM*Q_BWD_SP*2, DIM*SEQ*2);
#endif
    if (!dk->qBwd) return false;

    // KV backward: [1, KV_DIM, 1, 2*SEQ+2*DIM] → [1, DIM, 1, SEQ]
    printf("  Compiling kvBwd...\n");
    dk->kvBwd = compile_kern_mil_w(gen_kv_bwd_dynamic(), @{},
        KV_DIM*KV_BWD_SP*2, DIM*SEQ*2);
    if (!dk->kvBwd) return false;
#endif

    return true;
}

// ===== Checkpoint =====
#if N_HC > 1
// mHC param block (v5): the per-(layer,sub-layer) maps + AdamW state, appended
// after embed/sink/qnorm. Defined down in the mHC section; forward-declared here so
// save/load can call it while the file handle is open. See ADR 0002.
static void mhc_ckpt_io(FILE *f, int writing);
#endif
static void save_checkpoint(const char *path, int step, int total_steps, float lr, float loss,
                            double ct, double cw, int cs, int adam_t,
                            LayerWeights *lw, LayerAdam *la, float *rms_final, AdamState *arms_final,
                            float *embed, AdamState *aembed
#if ATTN_SINK
                            , float *attn_sink, AdamState *asink
#endif
#if QK_NORM
                            , float *qnorm_w, AdamState *aqnorm, float *knorm_w, AdamState *aknorm
#endif
                            ) {
    FILE *f = fopen(path, "wb");
    CkptHdr h = {0};
    // v5 carries the appended mHC block (#if N_HC>1); a vanilla build still stamps v4
    // so existing v4 checkpoints remain loadable. See ADR 0002 (build-tied format).
    h.magic = 0x424C5A54;
#if N_HC > 1
    h.version = 5;
#else
    h.version = 4;
#endif
    h.step = step; h.total_steps = total_steps;
    h.n_layers = NLAYERS; h.vocab_size = VOCAB; h.dim = DIM;
    h.hidden_dim = HIDDEN; h.n_heads = HEADS; h.seq_len = SEQ;
    h.lr = lr; h.loss = loss;
    h.cum_train = ct; h.cum_wall = cw; h.cum_steps = cs; h.adam_t = adam_t;
    h.kv_heads = KV_HEADS; h.head_dim = HD; h.q_dim = Q_DIM;
    fwrite(&h, sizeof(h), 1, f);
    for (int L = 0; L < NLAYERS; L++) {
        fwrite(lw[L].Wq,4,WQ_SZ,f); fwrite(lw[L].Wk,4,WK_SZ,f);
        fwrite(lw[L].Wv,4,WV_SZ,f); fwrite(lw[L].Wo,4,WO_SZ,f);
        fwrite(lw[L].W1,4,W1_SZ,f); fwrite(lw[L].W2,4,W2_SZ,f); fwrite(lw[L].W3,4,W3_SZ,f);
        fwrite(lw[L].rms_att,4,DIM,f); fwrite(lw[L].rms_ffn,4,DIM,f);
        fwrite(la[L].Wq.m,4,WQ_SZ,f); fwrite(la[L].Wq.v,4,WQ_SZ,f);
        fwrite(la[L].Wk.m,4,WK_SZ,f); fwrite(la[L].Wk.v,4,WK_SZ,f);
        fwrite(la[L].Wv.m,4,WV_SZ,f); fwrite(la[L].Wv.v,4,WV_SZ,f);
        fwrite(la[L].Wo.m,4,WO_SZ,f); fwrite(la[L].Wo.v,4,WO_SZ,f);
        fwrite(la[L].W1.m,4,W1_SZ,f); fwrite(la[L].W1.v,4,W1_SZ,f);
        fwrite(la[L].W2.m,4,W2_SZ,f); fwrite(la[L].W2.v,4,W2_SZ,f);
        fwrite(la[L].W3.m,4,W3_SZ,f); fwrite(la[L].W3.v,4,W3_SZ,f);
        fwrite(la[L].rms_att.m,4,DIM,f); fwrite(la[L].rms_att.v,4,DIM,f);
        fwrite(la[L].rms_ffn.m,4,DIM,f); fwrite(la[L].rms_ffn.v,4,DIM,f);
    }
    fwrite(rms_final,4,DIM,f);
    fwrite(arms_final->m,4,DIM,f); fwrite(arms_final->v,4,DIM,f);
    fwrite(embed,4,VOCAB*DIM,f);
    fwrite(aembed->m,4,VOCAB*DIM,f); fwrite(aembed->v,4,VOCAB*DIM,f);
#if ATTN_SINK
    // Per-head sink logits + Adam state, appended (issue #8). Only present in
    // ATTN_SINK builds; a sink build and a non-sink build never share a run.
    fwrite(attn_sink,4,(size_t)NLAYERS*HEADS,f);
    fwrite(asink->m,4,(size_t)NLAYERS*HEADS,f); fwrite(asink->v,4,(size_t)NLAYERS*HEADS,f);
#endif
#if QK_NORM
    // Per-layer Q/K RMSNorm gains + Adam state, appended (issue #7).
    fwrite(qnorm_w,4,(size_t)NLAYERS*HD,f);
    fwrite(aqnorm->m,4,(size_t)NLAYERS*HD,f); fwrite(aqnorm->v,4,(size_t)NLAYERS*HD,f);
    fwrite(knorm_w,4,(size_t)NLAYERS*HD,f);
    fwrite(aknorm->m,4,(size_t)NLAYERS*HD,f); fwrite(aknorm->v,4,(size_t)NLAYERS*HD,f);
#endif
#if N_HC > 1
    // mHC maps + AdamW state (issue #11). Last block, so any reader that stops at
    // embed (e.g. the dashboard's vanilla loader) stays aligned. See ADR 0002.
    mhc_ckpt_io(f, 1);
#endif
    fclose(f);
}

static bool load_checkpoint(const char *path, int *step, int *total_steps, float *lr, float *loss,
                             double *ct, double *cw, int *cs, int *adam_t,
                             LayerWeights *lw, LayerAdam *la, float *rms_final, AdamState *arms_final,
                             float *embed, AdamState *aembed
#if ATTN_SINK
                             , float *attn_sink, AdamState *asink
#endif
#if QK_NORM
                             , float *qnorm_w, AdamState *aqnorm, float *knorm_w, AdamState *aknorm
#endif
                             ) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    CkptHdr h;
    fread(&h, sizeof(h), 1, f);
    // A flagship (N_HC>1) build requires v5 (the mHC block must be present, else the
    // reads below run off the end); a vanilla build accepts v4 or v5. See ADR 0002.
#if N_HC > 1
    if (h.magic != 0x424C5A54 || h.version != 5) { fclose(f); return false; }
#else
    if (h.magic != 0x424C5A54 || (h.version != 4 && h.version != 5)) { fclose(f); return false; }
#endif
    *step = h.step; *total_steps = h.total_steps; *lr = h.lr; *loss = h.loss;
    *ct = h.cum_train; *cw = h.cum_wall; *cs = h.cum_steps; *adam_t = h.adam_t;
    for (int L = 0; L < NLAYERS; L++) {
        fread(lw[L].Wq,4,WQ_SZ,f); fread(lw[L].Wk,4,WK_SZ,f);
        fread(lw[L].Wv,4,WV_SZ,f); fread(lw[L].Wo,4,WO_SZ,f);
        fread(lw[L].W1,4,W1_SZ,f); fread(lw[L].W2,4,W2_SZ,f); fread(lw[L].W3,4,W3_SZ,f);
        fread(lw[L].rms_att,4,DIM,f); fread(lw[L].rms_ffn,4,DIM,f);
        fread(la[L].Wq.m,4,WQ_SZ,f); fread(la[L].Wq.v,4,WQ_SZ,f);
        fread(la[L].Wk.m,4,WK_SZ,f); fread(la[L].Wk.v,4,WK_SZ,f);
        fread(la[L].Wv.m,4,WV_SZ,f); fread(la[L].Wv.v,4,WV_SZ,f);
        fread(la[L].Wo.m,4,WO_SZ,f); fread(la[L].Wo.v,4,WO_SZ,f);
        fread(la[L].W1.m,4,W1_SZ,f); fread(la[L].W1.v,4,W1_SZ,f);
        fread(la[L].W2.m,4,W2_SZ,f); fread(la[L].W2.v,4,W2_SZ,f);
        fread(la[L].W3.m,4,W3_SZ,f); fread(la[L].W3.v,4,W3_SZ,f);
        fread(la[L].rms_att.m,4,DIM,f); fread(la[L].rms_att.v,4,DIM,f);
        fread(la[L].rms_ffn.m,4,DIM,f); fread(la[L].rms_ffn.v,4,DIM,f);
    }
    fread(rms_final,4,DIM,f);
    fread(arms_final->m,4,DIM,f); fread(arms_final->v,4,DIM,f);
    fread(embed,4,VOCAB*DIM,f);
    fread(aembed->m,4,VOCAB*DIM,f); fread(aembed->v,4,VOCAB*DIM,f);
#if ATTN_SINK
    fread(attn_sink,4,(size_t)NLAYERS*HEADS,f);
    fread(asink->m,4,(size_t)NLAYERS*HEADS,f); fread(asink->v,4,(size_t)NLAYERS*HEADS,f);
#endif
#if QK_NORM
    fread(qnorm_w,4,(size_t)NLAYERS*HD,f);
    fread(aqnorm->m,4,(size_t)NLAYERS*HD,f); fread(aqnorm->v,4,(size_t)NLAYERS*HD,f);
    fread(knorm_w,4,(size_t)NLAYERS*HD,f);
    fread(aknorm->m,4,(size_t)NLAYERS*HD,f); fread(aknorm->v,4,(size_t)NLAYERS*HD,f);
#endif
#if N_HC > 1
    // Read the mHC block into the already-allocated globals (mhc_global_init runs
    // before load_checkpoint, so this overwrites the fresh random init). See ADR 0002.
    mhc_ckpt_io(f, 0);
#endif
    fclose(f);
    return true;
}

// ===== R1 bridge: shared init in, raw gradients out =====
// Flat float32 layout shared with lilbro/ane_bridge (param_spec dense order):
//   embed[VOCAB*DIM], then per layer {Wq,Wk,Wv,Wo,W1,W2,W3,rms_att,rms_ffn},
//   then rms_final[DIM]. No MTP (the ANE trainer has no MTP path).
static bool load_init_weights(const char *path, LayerWeights *lw,
                              float *rms_final, float *embed) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    size_t want = (size_t)VOCAB*DIM
                + (size_t)NLAYERS*(WQ_SZ+WK_SZ+WV_SZ+WO_SZ+W1_SZ+W2_SZ+W3_SZ+2*DIM)
                + DIM;
    fseek(f, 0, SEEK_END); long have = ftell(f); fseek(f, 0, SEEK_SET);
    if ((size_t)have != want*4) {
        printf("  init size mismatch: %ld bytes, expected %zu\n", have, want*4);
        fclose(f); return false;
    }
    size_t n = 0;
    n += fread(embed, 4, (size_t)VOCAB*DIM, f);
    for (int L=0; L<NLAYERS; L++) {
        n += fread(lw[L].Wq,4,WQ_SZ,f); n += fread(lw[L].Wk,4,WK_SZ,f);
        n += fread(lw[L].Wv,4,WV_SZ,f); n += fread(lw[L].Wo,4,WO_SZ,f);
        n += fread(lw[L].W1,4,W1_SZ,f); n += fread(lw[L].W2,4,W2_SZ,f);
        n += fread(lw[L].W3,4,W3_SZ,f);
        n += fread(lw[L].rms_att,4,DIM,f); n += fread(lw[L].rms_ffn,4,DIM,f);
    }
    n += fread(rms_final,4,DIM,f);
    fclose(f);
    return n == want;
}

// Dump raw (unscaled, unclipped) gradients in the same flat layout as the init.
static void dump_grads(const char *path, LayerGrads *grads,
                       float *grms_final, float *gembed) {
    FILE *f = fopen(path, "wb");
    if (!f) { printf("  cannot open %s for grad dump\n", path); return; }
    fwrite(gembed,4,(size_t)VOCAB*DIM,f);
    for (int L=0; L<NLAYERS; L++) {
        LayerGrads *g = &grads[L];
        fwrite(g->Wq,4,WQ_SZ,f); fwrite(g->Wk,4,WK_SZ,f);
        fwrite(g->Wv,4,WV_SZ,f); fwrite(g->Wo,4,WO_SZ,f);
        fwrite(g->W1,4,W1_SZ,f); fwrite(g->W2,4,W2_SZ,f); fwrite(g->W3,4,W3_SZ,f);
        fwrite(g->rms_att,4,DIM,f); fwrite(g->rms_ffn,4,DIM,f);
    }
    fwrite(grms_final,4,DIM,f);
    fclose(f);
}

// Dump current weights in the same flat layout as the init/grad files. Used by
// the optimizer step-diff harness: run one update, dump the post-step weights,
// and compare against the numpy twin's optimizer applied to the same init+grads.
static void dump_flat_weights(const char *path, LayerWeights *lw,
                              float *rms_final, float *embed) {
    FILE *f = fopen(path, "wb");
    if (!f) { printf("  cannot open %s for weight dump\n", path); return; }
    fwrite(embed,4,(size_t)VOCAB*DIM,f);
    for (int L=0; L<NLAYERS; L++) {
        fwrite(lw[L].Wq,4,WQ_SZ,f); fwrite(lw[L].Wk,4,WK_SZ,f);
        fwrite(lw[L].Wv,4,WV_SZ,f); fwrite(lw[L].Wo,4,WO_SZ,f);
        fwrite(lw[L].W1,4,W1_SZ,f); fwrite(lw[L].W2,4,W2_SZ,f); fwrite(lw[L].W3,4,W3_SZ,f);
        fwrite(lw[L].rms_att,4,DIM,f); fwrite(lw[L].rms_ffn,4,DIM,f);
    }
    fwrite(rms_final,4,DIM,f);
    fclose(f);
}

// ===== Shared forward pass (training + validation) =====
// Runs all layers + final RMSNorm, leaving the normed final hidden in x_final
// and per-layer activations in acts[] (training needs them for backward; val
// ignores them). This is the model definition: validation loss is measured by
// the SAME code path that trains — so R0/R1, which gate the gradients of this
// exact forward, also vouch for the forward val uses. dw_grp is drained per
// layer exactly as the training loop did (empty during val -> no-op). Pass
// tm=NULL to skip timing accumulation.
typedef struct { double rms, ane_fwd, io_fwd, cblas_wait; } FwdTiming;

#if N_HC > 1
// ===== mHC state (issue #11) — file-scope so forward_hidden and the optimizer reach it =====
// Per (layer, sub-layer) map params + grads + AdamW state; heap tapes/wide buffers.
// All guarded #if N_HC>1, so N_HC==1 (default) compiles this away byte-for-byte.
typedef struct { AdamState Wpre,Wres,Wpost,Spre,Sres,Spost,a_pre,a_res,a_post; } MhcAdam;
typedef struct {
    MhcMap  *mapA, *mapF;       // [NLAYERS] attention- and FFN-sublayer maps (→ globals)
    MhcTape *tpA,  *tpF;        // tapes: [NLAYERS] when save, else [1] scratch (validation)
    float  **XwA, **XwF;        // [NLAYERS] saved input wide streams (NULL when !save)
    float  **FoutA,**FoutF;     // [NLAYERS] saved sub-layer outputs (NULL when !save)
    float   *Xw, *Xw2, *Ftmp;   // running wide stream (double-buffered) + Fout scratch
    int      save;              // 1 training (index by L, persist), 0 validation (scratch)
} MhcFwd;

static MhcMap  g_mapA[NLAYERS], g_mapF[NLAYERS];
static MhcGrad g_gmapA[NLAYERS], g_gmapF[NLAYERS];
static MhcAdam g_amapA[NLAYERS], g_amapF[NLAYERS];
static MhcFwd  g_mhc_train, g_mhc_val;
static float  *g_zero_ds;       // [DIM*SEQ] zeros — FFN residual base (mHC adds its own)
// backward scratch (allocated in mhc_global_init)
static float  *g_dXw, *g_dXw2, *g_dfout, *g_dB_a, *g_dC_a, *g_dB_f, *g_dC_f;

static MhcAdam mhc_adam_alloc(void){ MhcAdam a;
    a.Wpre=adam_alloc(MHC_M*N_HC); a.Wres=adam_alloc(MHC_M*N_HC*N_HC); a.Wpost=adam_alloc(MHC_M*N_HC);
    a.Spre=adam_alloc(N_HC); a.Sres=adam_alloc(N_HC*N_HC); a.Spost=adam_alloc(N_HC);
    a.a_pre=adam_alloc(1); a.a_res=adam_alloc(1); a.a_post=adam_alloc(1); return a; }

// v5 checkpoint (de)serialization of one (map, adam) pair. MHC_F is float in the
// trainer, and adam m/v are float, so every field is 4-byte; MHC_IO picks fwrite or
// fread on `writing`. Weights and AdamW moments interleaved so a partial file is
// obvious. Field order/sizes mirror mhc_adam_alloc exactly. See ADR 0002.
static void mhc_one_ckpt_io(MhcMap *m, MhcAdam *a, FILE *f, int writing){
    #define MHC_IO(p,n) do{ if(writing) fwrite((p),4,(size_t)(n),f); else fread((p),4,(size_t)(n),f); }while(0)
    MHC_IO(m->Wpre, MHC_M*N_HC);      MHC_IO(a->Wpre.m, MHC_M*N_HC);      MHC_IO(a->Wpre.v, MHC_M*N_HC);
    MHC_IO(m->Wres, MHC_M*N_HC*N_HC); MHC_IO(a->Wres.m, MHC_M*N_HC*N_HC); MHC_IO(a->Wres.v, MHC_M*N_HC*N_HC);
    MHC_IO(m->Wpost,MHC_M*N_HC);      MHC_IO(a->Wpost.m,MHC_M*N_HC);      MHC_IO(a->Wpost.v,MHC_M*N_HC);
    MHC_IO(m->Spre, N_HC);            MHC_IO(a->Spre.m, N_HC);            MHC_IO(a->Spre.v, N_HC);
    MHC_IO(m->Sres, N_HC*N_HC);       MHC_IO(a->Sres.m, N_HC*N_HC);       MHC_IO(a->Sres.v, N_HC*N_HC);
    MHC_IO(m->Spost,N_HC);            MHC_IO(a->Spost.m,N_HC);            MHC_IO(a->Spost.v,N_HC);
    MHC_IO(&m->a_pre, 1);  MHC_IO(a->a_pre.m, 1);  MHC_IO(a->a_pre.v, 1);
    MHC_IO(&m->a_res, 1);  MHC_IO(a->a_res.m, 1);  MHC_IO(a->a_res.v, 1);
    MHC_IO(&m->a_post,1);  MHC_IO(a->a_post.m,1);  MHC_IO(a->a_post.v,1);
    #undef MHC_IO
}
static void mhc_ckpt_io(FILE *f, int writing){
    for(int L=0;L<NLAYERS;L++){
        mhc_one_ckpt_io(&g_mapA[L], &g_amapA[L], f, writing);
        mhc_one_ckpt_io(&g_mapF[L], &g_amapF[L], f, writing);
    }
}

static float mhc_one_gnsq(const MhcGrad *g){ float s,acc=0;
    vDSP_dotpr(g->Wpre,1,g->Wpre,1,&s,(vDSP_Length)(MHC_M*N_HC)); acc+=s;
    vDSP_dotpr(g->Wres,1,g->Wres,1,&s,(vDSP_Length)(MHC_M*N_HC*N_HC)); acc+=s;
    vDSP_dotpr(g->Wpost,1,g->Wpost,1,&s,(vDSP_Length)(MHC_M*N_HC)); acc+=s;
    vDSP_dotpr(g->Spre,1,g->Spre,1,&s,(vDSP_Length)N_HC); acc+=s;
    vDSP_dotpr(g->Sres,1,g->Sres,1,&s,(vDSP_Length)(N_HC*N_HC)); acc+=s;
    vDSP_dotpr(g->Spost,1,g->Spost,1,&s,(vDSP_Length)N_HC); acc+=s;
    acc += g->a_pre*g->a_pre + g->a_res*g->a_res + g->a_post*g->a_post; return acc; }
static float mhc_gradnorm_sq(void){ float a=0;
    for(int L=0;L<NLAYERS;L++){ a+=mhc_one_gnsq(&g_gmapA[L]); a+=mhc_one_gnsq(&g_gmapF[L]); } return a; }

static void mhc_one_scale(MhcGrad *g, float s){
    vDSP_vsmul(g->Wpre,1,&s,g->Wpre,1,(vDSP_Length)(MHC_M*N_HC));
    vDSP_vsmul(g->Wres,1,&s,g->Wres,1,(vDSP_Length)(MHC_M*N_HC*N_HC));
    vDSP_vsmul(g->Wpost,1,&s,g->Wpost,1,(vDSP_Length)(MHC_M*N_HC));
    vDSP_vsmul(g->Spre,1,&s,g->Spre,1,(vDSP_Length)N_HC);
    vDSP_vsmul(g->Sres,1,&s,g->Sres,1,(vDSP_Length)(N_HC*N_HC));
    vDSP_vsmul(g->Spost,1,&s,g->Spost,1,(vDSP_Length)N_HC);
    g->a_pre*=s; g->a_res*=s; g->a_post*=s; }
static void mhc_scale_grads(float s){
    for(int L=0;L<NLAYERS;L++){ mhc_one_scale(&g_gmapA[L],s); mhc_one_scale(&g_gmapF[L],s); } }

static void mhc_zero_grads(void){
    for(int L=0;L<NLAYERS;L++){ mhc_grad_zero(&g_gmapA[L]); mhc_grad_zero(&g_gmapF[L]); } }

// AdamW: W* with weight decay; S*/alpha bias-like (no wd). The auxiliary map
// projections use AdamW on both --opt paths (the main 2D-Muon rule doesn't extend
// to these tiny dynamic generators; AdamW is lower-risk for the R0 gate).
static void mhc_one_opt(MhcMap *w, MhcGrad *g, MhcAdam *a, int t,
                        float lr,float b1,float b2,float eps,float wd){
    adam_update(w->Wpre, g->Wpre, &a->Wpre, t,lr,b1,b2,eps,wd);
    adam_update(w->Wres, g->Wres, &a->Wres, t,lr,b1,b2,eps,wd);
    adam_update(w->Wpost,g->Wpost,&a->Wpost,t,lr,b1,b2,eps,wd);
    adam_update(w->Spre, g->Spre, &a->Spre, t,lr,b1,b2,eps,0.0f);
    adam_update(w->Sres, g->Sres, &a->Sres, t,lr,b1,b2,eps,0.0f);
    adam_update(w->Spost,g->Spost,&a->Spost,t,lr,b1,b2,eps,0.0f);
    adam_update(&w->a_pre, &g->a_pre, &a->a_pre, t,lr,b1,b2,eps,0.0f);
    adam_update(&w->a_res, &g->a_res, &a->a_res, t,lr,b1,b2,eps,0.0f);
    adam_update(&w->a_post,&g->a_post,&a->a_post,t,lr,b1,b2,eps,0.0f); }
static void mhc_optimize(int t, float lr,float b1,float b2,float eps,float wd){
    for(int L=0;L<NLAYERS;L++){
        mhc_one_opt(&g_mapA[L],&g_gmapA[L],&g_amapA[L],t,lr,b1,b2,eps,wd);
        mhc_one_opt(&g_mapF[L],&g_gmapF[L],&g_amapF[L],t,lr,b1,b2,eps,wd); } }

// Worst |rowsum-1|,|colsum-1| of B over every saved training tape (acceptance:
// the Sinkhorn residual map must stay doubly-stochastic — #5 proved τ≥0.5, t_max=20
// holds even in fp16; here the CPU maps are fp32, a tighter bound).
static float mhc_ds_max_train(void){
    float w=0; for(int L=0;L<NLAYERS;L++){
        float a=mhc_ds_residual(&g_mhc_train.tpA[L]); if(a>w)w=a;
        float f=mhc_ds_residual(&g_mhc_train.tpF[L]); if(f>w)w=f; } return w; }

static float *mhc_xw_alloc(void){ return (float*)malloc((size_t)N_HC*DIM*SEQ*4); }
// Allocate maps/grads/adam + the two forward bundles (training saves per-layer,
// validation reuses a single scratch tape) + backward scratch.
static void mhc_global_init(void){
    unsigned seed=0x1234567u;
    for(int L=0;L<NLAYERS;L++){
        g_mapA[L]=mhc_map_alloc(); mhc_map_init(&g_mapA[L],&seed);
        g_mapF[L]=mhc_map_alloc(); mhc_map_init(&g_mapF[L],&seed);
        g_gmapA[L]=mhc_grad_alloc(); g_gmapF[L]=mhc_grad_alloc();
        g_amapA[L]=mhc_adam_alloc(); g_amapF[L]=mhc_adam_alloc();
    }
    g_zero_ds=(float*)calloc((size_t)DIM*SEQ,4);
    g_mhc_train.mapA=g_mapA; g_mhc_train.mapF=g_mapF; g_mhc_train.save=1;
    g_mhc_train.tpA=(MhcTape*)malloc((size_t)NLAYERS*sizeof(MhcTape));
    g_mhc_train.tpF=(MhcTape*)malloc((size_t)NLAYERS*sizeof(MhcTape));
    g_mhc_train.XwA=(float**)malloc(NLAYERS*sizeof(float*));
    g_mhc_train.XwF=(float**)malloc(NLAYERS*sizeof(float*));
    g_mhc_train.FoutA=(float**)malloc(NLAYERS*sizeof(float*));
    g_mhc_train.FoutF=(float**)malloc(NLAYERS*sizeof(float*));
    for(int L=0;L<NLAYERS;L++){
        g_mhc_train.XwA[L]=mhc_xw_alloc(); g_mhc_train.XwF[L]=mhc_xw_alloc();
        g_mhc_train.FoutA[L]=(float*)malloc((size_t)DIM*SEQ*4);
        g_mhc_train.FoutF[L]=(float*)malloc((size_t)DIM*SEQ*4);
    }
    g_mhc_train.Xw=mhc_xw_alloc(); g_mhc_train.Xw2=mhc_xw_alloc();
    g_mhc_train.Ftmp=(float*)malloc((size_t)DIM*SEQ*4);
    g_mhc_val.mapA=g_mapA; g_mhc_val.mapF=g_mapF; g_mhc_val.save=0;
    g_mhc_val.tpA=(MhcTape*)malloc(sizeof(MhcTape));
    g_mhc_val.tpF=(MhcTape*)malloc(sizeof(MhcTape));
    g_mhc_val.XwA=g_mhc_val.XwF=g_mhc_val.FoutA=g_mhc_val.FoutF=NULL;
    g_mhc_val.Xw=mhc_xw_alloc(); g_mhc_val.Xw2=mhc_xw_alloc();
    g_mhc_val.Ftmp=(float*)malloc((size_t)DIM*SEQ*4);
    g_dXw=mhc_xw_alloc(); g_dXw2=mhc_xw_alloc(); g_dfout=(float*)malloc((size_t)DIM*SEQ*4);
    g_dB_a=(float*)malloc((size_t)SEQ*N_HC*N_HC*4); g_dC_a=(float*)malloc((size_t)SEQ*N_HC*4);
    g_dB_f=(float*)malloc((size_t)SEQ*N_HC*N_HC*4); g_dC_f=(float*)malloc((size_t)SEQ*N_HC*4);
}
#endif // N_HC > 1

static void forward_hidden(
    DynLayerKernels *dk, PerLayerSurfaces *pls, PerLayerRequests *plr,
    LayerWeights *lw, const float *rms_final, LayerActs *acts,
    float *x_cur, float *xnorm_buf, float *x_final, float res_alpha,
    const float *attn_sink, const float *qnorm_w, const float *knorm_w,
    void *mhc, dispatch_group_t dw_grp, FwdTiming *tm)
{
    (void)attn_sink; (void)qnorm_w; (void)knorm_w; (void)mhc;
#if N_HC > 1
    MhcFwd *mf = (MhcFwd*)mhc;
    // Entry: broadcast the embedded hidden into all N_HC streams (X_l ∈ ℝ^{N_HC×d}).
    for (int i=0;i<N_HC;i++) memcpy(mf->Xw + (size_t)i*DIM*SEQ, x_cur, (size_t)DIM*SEQ*4);
#endif
    uint64_t t0;
    double r_rms = 0, r_ane = 0, r_io = 0, r_wait = 0;
    for (int L = 0; L < NLAYERS; L++) {
        LayerActs *ac = &acts[L];
#if N_HC > 1
        // mHC attention sub-layer: collapse the N_HC streams to the attention input
        // u = A·X (ac->layer_in), generating + saving the per-position A/B/C maps.
        MhcTape *tpa = mf->save ? &mf->tpA[L] : &mf->tpA[0];
        if (mf->save) memcpy(mf->XwA[L], mf->Xw, (size_t)N_HC*DIM*SEQ*4);
        t0 = mach_absolute_time();
        mhc_premap(mf->Xw, &mf->mapA[L], tpa, ac->layer_in);
        rmsnorm(xnorm_buf, ac->layer_in, lw[L].rms_att, DIM, SEQ);
        r_rms += tb_ms(mach_absolute_time() - t0);
#else
        memcpy(ac->layer_in, x_cur, SEQ*DIM*4);

        // RMSNorm1 (CPU)
        t0 = mach_absolute_time();
        rmsnorm(xnorm_buf, x_cur, lw[L].rms_att, DIM, SEQ);
        r_rms += tb_ms(mach_absolute_time() - t0);
#endif
        memcpy(ac->xnorm, xnorm_buf, SEQ*DIM*4);

        // Wait for any pending dW cblas (empty during validation)
        t0 = mach_absolute_time();
        dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);
        r_wait += tb_ms(mach_absolute_time() - t0);

        // SDPA forward (ANE)
        t0 = mach_absolute_time();
        write_sdpa_fwd_acts(pls[L].sdpaFwd_in, xnorm_buf);
        r_io += tb_ms(mach_absolute_time() - t0);
        t0 = mach_absolute_time();
        ane_eval_req(dk->sdpaFwd, plr[L].sdpaFwd);
        r_ane += tb_ms(mach_absolute_time() - t0);

        t0 = mach_absolute_time();
        IOSurfaceLock(dk->sdpaFwd->ioOut, kIOSurfaceLockReadOnly, NULL);
        _Float16 *fwd_out = (_Float16*)IOSurfaceGetBaseAddress(dk->sdpaFwd->ioOut);
        int off = 0;
        cvt_f16_f32(ac->attn_out, fwd_out + off, Q_DIM*SEQ); off += Q_DIM*SEQ;
        cvt_f16_f32(ac->Q,        fwd_out + off, Q_DIM*SEQ); off += Q_DIM*SEQ;
        cvt_f16_f32(ac->K,        fwd_out + off, KV_DIM*SEQ); off += KV_DIM*SEQ;
        cvt_f16_f32(ac->V,        fwd_out + off, KV_DIM*SEQ); off += KV_DIM*SEQ;
        IOSurfaceUnlock(dk->sdpaFwd->ioOut, kIOSurfaceLockReadOnly, NULL);
        r_io += tb_ms(mach_absolute_time() - t0);

#if ATTN_CPU
        // Re-run the attention core on CPU for the knobs the ANE softmax kernel
        // can't express: the per-head sink logit (#8) and/or QK-norm before the
        // scores (#7). Q/K/V from the kernel are kept; attn_out is overwritten.
        // Backward bypasses sdpaBwd1/2 (see attn_cpu_backward). Each knob is
        // NULL when off, so this stays exact baseline attention unless one is on.
        t0 = mach_absolute_time();
        attn_cpu_forward(ac->attn_out, ac->Q, ac->K, ac->V,
                         ATTN_SINK ? attn_sink + L*HEADS : NULL,
                         QK_NORM ? qnorm_w + L*HD : NULL,
                         QK_NORM ? knorm_w + L*HD : NULL, SEQ);
        r_rms += tb_ms(mach_absolute_time() - t0);
#endif

        // Wo forward (ANE)
        t0 = mach_absolute_time();
#if WO_FUNCPARAM
        write_wo_fwd_acts_fp(pls[L].woFwd_in, ac->attn_out);
#else
        write_wo_fwd_acts(pls[L].woFwd_in, ac->attn_out);
#endif
        r_io += tb_ms(mach_absolute_time() - t0);
        t0 = mach_absolute_time();
        ane_eval_req(dk->woFwd, plr[L].woFwd);
        r_ane += tb_ms(mach_absolute_time() - t0);
        t0 = mach_absolute_time();
        io_read_dyn(dk->woFwd->ioOut, ac->o_out, DIM, SEQ);
        r_io += tb_ms(mach_absolute_time() - t0);

#if N_HC > 1
        // mHC attention recombine: X ← B·X + C⊗(res_alpha·o_out). Then collapse the
        // updated streams to the FFN input u = A·X (ac->x2), and run the FFN with a
        // ZERO residual base so the kernel returns the bare res_alpha·ffn (mHC adds
        // its own residual via the next recombine).
        t0 = mach_absolute_time();
        for (int i=0;i<DIM*SEQ;i++) mf->Ftmp[i] = res_alpha * ac->o_out[i];
        if (mf->save) memcpy(mf->FoutA[L], mf->Ftmp, (size_t)DIM*SEQ*4);
        mhc_recombine(mf->Xw, mf->Ftmp, tpa, mf->Xw2);
        { float *sw=mf->Xw; mf->Xw=mf->Xw2; mf->Xw2=sw; }
        MhcTape *tpf = mf->save ? &mf->tpF[L] : &mf->tpF[0];
        if (mf->save) memcpy(mf->XwF[L], mf->Xw, (size_t)N_HC*DIM*SEQ*4);
        mhc_premap(mf->Xw, &mf->mapF[L], tpf, ac->x2);
        rmsnorm(ac->x2norm, ac->x2, lw[L].rms_ffn, DIM, SEQ);
        r_rms += tb_ms(mach_absolute_time() - t0);

        // Fused FFN (ANE) — zero residual base
        t0 = mach_absolute_time();
        write_ffn_fused_acts(pls[L].ffnFused_in, ac->x2norm, g_zero_ds);
        r_io += tb_ms(mach_absolute_time() - t0);
#else
        // CPU: scaled residual + RMSNorm2
        t0 = mach_absolute_time();
        vDSP_vsma(ac->o_out, 1, &res_alpha, x_cur, 1, ac->x2, 1, (vDSP_Length)(SEQ*DIM));
        rmsnorm(ac->x2norm, ac->x2, lw[L].rms_ffn, DIM, SEQ);
        r_rms += tb_ms(mach_absolute_time() - t0);

        // Fused FFN (ANE)
        t0 = mach_absolute_time();
        write_ffn_fused_acts(pls[L].ffnFused_in, ac->x2norm, ac->x2);
        r_io += tb_ms(mach_absolute_time() - t0);
#endif
        t0 = mach_absolute_time();
        ane_eval_req(dk->ffnFused, plr[L].ffnFused);
        r_ane += tb_ms(mach_absolute_time() - t0);

        t0 = mach_absolute_time();
        IOSurfaceLock(dk->ffnFused->ioOut, kIOSurfaceLockReadOnly, NULL);
        _Float16 *ffn_out = (_Float16*)IOSurfaceGetBaseAddress(dk->ffnFused->ioOut);
        off = 0;
#if N_HC > 1
        cvt_f16_f32(mf->Ftmp,    ffn_out + off, DIM*SEQ);     off += DIM*SEQ;  // bare res_alpha·ffn
#else
        cvt_f16_f32(x_cur,       ffn_out + off, DIM*SEQ);     off += DIM*SEQ;
#endif
        cvt_f16_f32(ac->h1,      ffn_out + off, HIDDEN*SEQ);  off += HIDDEN*SEQ;
        cvt_f16_f32(ac->h3,      ffn_out + off, HIDDEN*SEQ);  off += HIDDEN*SEQ;
        cvt_f16_f32(ac->silu_out,ffn_out + off, HIDDEN*SEQ);
        IOSurfaceUnlock(dk->ffnFused->ioOut, kIOSurfaceLockReadOnly, NULL);
        r_io += tb_ms(mach_absolute_time() - t0);
#if N_HC > 1
        // mHC FFN recombine: X ← B·X + C⊗(res_alpha·ffn).
        t0 = mach_absolute_time();
        if (mf->save) memcpy(mf->FoutF[L], mf->Ftmp, (size_t)DIM*SEQ*4);
        mhc_recombine(mf->Xw, mf->Ftmp, tpf, mf->Xw2);
        { float *sw=mf->Xw; mf->Xw=mf->Xw2; mf->Xw2=sw; }
        r_rms += tb_ms(mach_absolute_time() - t0);
#endif
    }

#if N_HC > 1
    // Exit: collapse the N_HC streams back to a single hidden (sum) before the head.
    memset(x_cur, 0, (size_t)DIM*SEQ*4);
    for (int i=0;i<N_HC;i++)
        vDSP_vadd(x_cur, 1, mf->Xw + (size_t)i*DIM*SEQ, 1, x_cur, 1, (vDSP_Length)(DIM*SEQ));
#endif

    // Final RMSNorm (CPU)
    t0 = mach_absolute_time();
    rmsnorm(x_final, x_cur, rms_final, DIM, SEQ);
    r_rms += tb_ms(mach_absolute_time() - t0);

    if (tm) { tm->rms += r_rms; tm->ane_fwd += r_ane; tm->io_fwd += r_io; tm->cblas_wait += r_wait; }
}

// ===== Validation: mean CE over a fixed deterministic set of val batches =====
// Evenly-spaced start positions over the held-out shard (no RNG) so the number
// is comparable step-to-step and run-to-run. b=1 per micro-batch (as in
// training). Uses the SAME compact LM head as training, so val loss is on the
// same scale as the printed train loss. Targets absent from the compact vocab
// are skipped (see cross_entropy_loss_only).
static float eval_val_loss(
    DynLayerKernels *dk, PerLayerSurfaces *pls, PerLayerRequests *plr,
    LayerWeights *lw, const float *rms_final,
    const float *embed, const float *cembed, int CV, const VocabMap *vm,
    const uint16_t *vdata, size_t vntok, int nbatches,
    LayerActs *acts, float *x_cur, float *xnorm_buf, float *x_final,
    float *logits, float res_alpha, const float *attn_sink,
    const float *qnorm_w, const float *knorm_w, dispatch_group_t dw_grp)
{
    if (!vdata || vntok < (size_t)(SEQ + 1) || nbatches < 1) return 0.0f;
    size_t maxpos = vntok - SEQ - 1;
    double total = 0; long total_valid = 0;
    int targets[SEQ];
    for (int b = 0; b < nbatches; b++) {
        size_t pos = (size_t)((double)b * (double)maxpos / (double)nbatches);
        const uint16_t *in = vdata + pos;
        const uint16_t *tg = vdata + pos + 1;
        for (int t = 0; t < SEQ; t++) targets[t] = vm->full_to_compact[tg[t]];
        embed_lookup(x_cur, embed, in, DIM, SEQ);
        forward_hidden(dk, pls, plr, lw, rms_final, acts,
                       x_cur, xnorm_buf, x_final, res_alpha,
                       attn_sink, qnorm_w, knorm_w,
#if N_HC > 1
                       &g_mhc_val,
#else
                       NULL,
#endif
                       dw_grp, NULL);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    CV, SEQ, DIM, 1.0f, cembed, DIM, x_final, SEQ, 0.0f, logits, SEQ);
        int nv = 0;
        float l = cross_entropy_loss_only(logits, targets, CV, SEQ, &nv);
        total += (double)l * nv; total_valid += nv;
    }
    return total_valid > 0 ? (float)(total / total_valid) : 0.0f;
}

// ===== In-trainer faithful sampler (ADR 0002) =====
// Sibling of eval_val_loss: same faithful forward_hidden (mHC included #if N_HC>1),
// but autoregressive over a prompt instead of CE over held-out data. Full-sequence
// recompute per token — attention is causal and mHC maps are per-position, so the
// padding tail (positions >= n) cannot leak into the last real position's logits.
// Emits "[gen step=N] <full-token-ids>"; the dashboard decodes with its tokenizer.
//
// Pick the next compact class from the logits column at `pos`: argmax for temp<=0,
// else temperature softmax over the top_k logits.
static int mhc_sample_col(const float *logits, int CV, int seq, int pos,
                          float temperature, int top_k){
    if (temperature < 1e-4f){
        int best=0; float bv=logits[(size_t)0*seq+pos];
        for(int c=1;c<CV;c++){ float v=logits[(size_t)c*seq+pos]; if(v>bv){bv=v;best=c;} }
        return best;
    }
    if (top_k<=0 || top_k>CV) top_k=CV;
    int   *idx=(int*)malloc((size_t)top_k*sizeof(int));
    float *val=(float*)malloc((size_t)top_k*sizeof(float));
    int filled=0, worst_i=0; float worst=0;
    for(int c=0;c<CV;c++){
        float v=logits[(size_t)c*seq+pos];
        if(filled<top_k){
            idx[filled]=c; val[filled]=v; filled++;
            if(filled==top_k){ worst=val[0]; worst_i=0;
                for(int j=1;j<top_k;j++) if(val[j]<worst){worst=val[j];worst_i=j;} }
        } else if(v>worst){
            idx[worst_i]=c; val[worst_i]=v;
            worst=val[0]; worst_i=0;
            for(int j=1;j<top_k;j++) if(val[j]<worst){worst=val[j];worst_i=j;}
        }
    }
    float mx=-1e30f; for(int j=0;j<top_k;j++){ val[j]/=temperature; if(val[j]>mx)mx=val[j]; }
    float sum=0; for(int j=0;j<top_k;j++){ val[j]=expf(val[j]-mx); sum+=val[j]; }
    float r=(float)drand48()*sum, acc=0; int pick=idx[top_k-1];
    for(int j=0;j<top_k;j++){ acc+=val[j]; if(acc>=r){ pick=idx[j]; break; } }
    free(idx); free(val);
    return pick;
}

static void sample_and_emit(
    DynLayerKernels *dk, PerLayerSurfaces *pls, PerLayerRequests *plr,
    LayerWeights *lw, const float *rms_final,
    const float *embed, const float *cembed, int CV, const VocabMap *vm,
    LayerActs *acts, float *x_cur, float *xnorm_buf, float *x_final,
    float *logits, float res_alpha, const float *attn_sink,
    const float *qnorm_w, const float *knorm_w, dispatch_group_t dw_grp,
    int step, const int *prompt_ids, int prompt_len, int n_new,
    float temperature, int top_k)
{
    uint16_t in[SEQ];
    int toks[SEQ];
    int n = prompt_len > 0 ? prompt_len : 1;
    if (n > SEQ-1) n = SEQ-1;
    for (int i=0;i<n;i++) toks[i] = prompt_ids ? prompt_ids[i] : 1;  // default BOS=1
    for (int g=0; g<n_new && n<SEQ; g++){
        for (int t=0;t<SEQ;t++) in[t] = (uint16_t)(t<n ? toks[t] : 0);
        embed_lookup(x_cur, embed, in, DIM, SEQ);
        forward_hidden(dk, pls, plr, lw, rms_final, acts,
                       x_cur, xnorm_buf, x_final, res_alpha,
                       attn_sink, qnorm_w, knorm_w,
#if N_HC > 1
                       &g_mhc_val,
#else
                       NULL,
#endif
                       dw_grp, NULL);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    CV, SEQ, DIM, 1.0f, cembed, DIM, x_final, SEQ, 0.0f, logits, SEQ);
        int c = mhc_sample_col(logits, CV, SEQ, n-1, temperature, top_k);
        int full = (c>=0 && c<CV) ? vm->compact_to_full[c] : 0;
        toks[n++] = full;
        if (full == 2) break;  // EOS
    }
    printf("[gen step=%d]", step);
    for (int i=0;i<n;i++) printf(" %d", toks[i]);
    printf("\n");
    fflush(stdout);
}

#if MTP_DEPTH > 0
// ===== Multi-Token Prediction orchestration (issue #6, CPU-first) =====
// Per depth kk (1..MTP_DEPTH): hp = previous hidden truncated to Sk=SEQ-kk; e =
// embedding of the kk-shifted target; hk = block(proj·concat(rms_h(hp),rms_e(e)));
// loss_kk = CE(compact-head(hk), targets[kk:]). Combined term = lambda·mean. The
// trunk and embed gradients flow back into the shared tables. See cpu_ops.h
// mtp_block_* and test_mtp.c (FD-verified end-to-end).
typedef struct { float rms_h[DIM], rms_e[DIM], proj[2*DIM*DIM]; MtpBlock blk; } MtpParams;
typedef struct { AdamState rms_h, rms_e, proj, Wq, Wk, Wv, Wo, W1, W2, W3, rms_att, rms_ffn; } MtpAdam;
typedef struct { MtpBlockAct act; float hk[DIM*SEQ], hp[DIM*SEQ], ev[DIM*SEQ], normed[2*DIM*SEQ]; int Sk; } MtpSaved;

static int mtp_ndep(void) { int n=0; for(int kk=1;kk<=MTP_DEPTH;kk++){ if(SEQ-kk>0) n++; } return n; }

// Forward: returns the MTP loss term (lambda*mean of per-depth CE), fills saved[].
static float mtp_forward(const float *trunk, const uint16_t *tgt_raw, const uint16_t *ctgt,
                         const float *embed, const float *cembed, int CV, const float *rms_final,
                         const MtpParams *mtp, float res_alpha, MtpSaved *saved) {
    int prevS=SEQ, ndep=0; const float *h_prev=trunk; double sum=0;
    float *nh=(float*)malloc(DIM*SEQ*4),*ne=(float*)malloc(DIM*SEQ*4),*hkin=(float*)malloc(DIM*SEQ*4),*hn=(float*)malloc(DIM*SEQ*4);
    for (int kk=1; kk<=MTP_DEPTH; kk++) { int Sk=SEQ-kk; if(Sk<=0) break; int d=kk-1; ndep++;
        const MtpParams *M=&mtp[d]; MtpSaved *sv=&saved[d]; sv->Sk=Sk;
        for(int i=0;i<DIM;i++) for(int t=0;t<Sk;t++) sv->hp[i*Sk+t]=h_prev[i*prevS+t];
        for(int i=0;i<DIM;i++) for(int t=0;t<Sk;t++) sv->ev[i*Sk+t]=embed[(size_t)tgt_raw[kk-1+t]*DIM+i];
        rmsnorm(nh, sv->hp, M->rms_h, DIM, Sk); rmsnorm(ne, sv->ev, M->rms_e, DIM, Sk);
        for(int i=0;i<DIM;i++) for(int t=0;t<Sk;t++) sv->normed[i*Sk+t]=nh[i*Sk+t];
        for(int i=0;i<DIM;i++) for(int t=0;t<Sk;t++) sv->normed[(DIM+i)*Sk+t]=ne[i*Sk+t];
        mtp_mm(hkin, M->proj, sv->normed, DIM, 2*DIM, Sk);
        mtp_block_fwd(sv->hk, hkin, &M->blk, &sv->act, Sk, res_alpha);
        rmsnorm(hn, sv->hk, rms_final, DIM, Sk);
        float *lg=(float*)malloc((size_t)CV*Sk*4); mtp_mm(lg, cembed, hn, CV, DIM, Sk);
        double L=0;
        for(int t=0;t<Sk;t++){ int ct=ctgt[kk+t]; if(ct<0||ct>=CV) continue;
            float m=-1e30f; for(int v=0;v<CV;v++){float l=lg[v*Sk+t]; if(l>m)m=l;}
            float Z=0; for(int v=0;v<CV;v++) Z+=expf(lg[v*Sk+t]-m); L += (m+logf(Z)) - lg[ct*Sk+t]; }
        sum += L/Sk; free(lg);
        h_prev=sv->hk; prevS=Sk;
    }
    free(nh);free(ne);free(hkin);free(hn);
    return ndep ? (float)(MTP_LAMBDA*sum/ndep) : 0.0f;
}

// Backward: accumulates d_trunk, gembed (input), gcembed/grms_final (head), and
// the per-depth MTP grads. All scaled by loss_scale to match the main loss.
static void mtp_backward(const MtpSaved *saved, int ndep, const uint16_t *tgt_raw, const uint16_t *ctgt,
                         const float *cembed, int CV, const float *rms_final, const MtpParams *mtp,
                         float res_alpha, float loss_scale,
                         float *d_trunk, float *gembed, float *gcembed, float *grms_final, MtpParams *gmtp) {
    float *dh_next=(float*)malloc(DIM*SEQ*4); int dh_next_S=0;
    for (int d=ndep-1; d>=0; d--) { int kk=d+1; int Sk=saved[d].Sk; const MtpParams *M=&mtp[d]; MtpParams *G=&gmtp[d];
        const MtpSaved *sv=&saved[d];
        float *hn=(float*)malloc(DIM*Sk*4); rmsnorm(hn, sv->hk, rms_final, DIM, Sk);
        float *lg=(float*)malloc((size_t)CV*Sk*4); mtp_mm(lg, cembed, hn, CV, DIM, Sk);
        float *dlg=(float*)calloc((size_t)CV*Sk,4);
        float sc = loss_scale * MTP_LAMBDA / (float)(ndep * Sk);
        for(int t=0;t<Sk;t++){ int ct=ctgt[kk+t]; if(ct<0||ct>=CV) continue;
            float m=-1e30f; for(int v=0;v<CV;v++){float l=lg[v*Sk+t]; if(l>m)m=l;}
            float Z=0; for(int v=0;v<CV;v++) Z+=expf(lg[v*Sk+t]-m);
            for(int v=0;v<CV;v++){ float p=expf(lg[v*Sk+t]-m)/Z; dlg[v*Sk+t]=(p-(v==ct?1.0f:0.0f))*sc; } }
        mtp_dWacc(gcembed, dlg, hn, CV, DIM, Sk);
        float *dhn=(float*)malloc(DIM*Sk*4); mtp_mmWT(dhn, cembed, dlg, CV, DIM, Sk);
        float *dhk=(float*)calloc(DIM*Sk,4); rmsnorm_bwd(dhk, grms_final, dhn, sv->hk, rms_final, DIM, Sk);
        if (d<ndep-1) for(int i=0;i<DIM;i++) for(int t=0;t<dh_next_S;t++) dhk[i*Sk+t]+=dh_next[i*dh_next_S+t];
        float *dhkin=(float*)calloc(DIM*Sk,4); mtp_block_bwd(dhkin, dhk, &M->blk, &sv->act, &G->blk, Sk, res_alpha);
        mtp_dWacc(G->proj, dhkin, sv->normed, DIM, 2*DIM, Sk);
        float *dnormed=(float*)malloc((size_t)2*DIM*Sk*4); mtp_mmWT(dnormed, M->proj, dhkin, DIM, 2*DIM, Sk);
        float *dnh=(float*)malloc(DIM*Sk*4),*dne=(float*)malloc(DIM*Sk*4);
        for(int i=0;i<DIM;i++) for(int t=0;t<Sk;t++){ dnh[i*Sk+t]=dnormed[i*Sk+t]; dne[i*Sk+t]=dnormed[(DIM+i)*Sk+t]; }
        float *dhp=(float*)calloc(DIM*Sk,4); rmsnorm_bwd(dhp, G->rms_h, dnh, sv->hp, M->rms_h, DIM, Sk);
        float *de=(float*)calloc(DIM*Sk,4); rmsnorm_bwd(de, G->rms_e, dne, sv->ev, M->rms_e, DIM, Sk);
        for(int i=0;i<DIM;i++) for(int t=0;t<Sk;t++) gembed[(size_t)tgt_raw[kk-1+t]*DIM+i]+=de[i*Sk+t];
        if (d==0) for(int i=0;i<DIM;i++) for(int t=0;t<Sk;t++) d_trunk[i*SEQ+t]+=dhp[i*Sk+t];
        else { memcpy(dh_next, dhp, DIM*Sk*4); dh_next_S=Sk; }
        free(hn);free(lg);free(dlg);free(dhn);free(dhk);free(dhkin);free(dnormed);free(dnh);free(dne);free(dhp);free(de);
    }
    free(dh_next);
}

static void mtp_alloc_init(MtpParams *w, MtpParams *g, MtpAdam *a, uint64_t *rng) {
    for (int d=0; d<MTP_DEPTH; d++) {
        for (int i=0;i<DIM;i++){ w[d].rms_h[i]=1.0f; w[d].rms_e[i]=1.0f; w[d].blk.rms_att[i]=1.0f; w[d].blk.rms_ffn[i]=1.0f; }
        // Small random init for the 2D matrices (proj + block), like the main weights.
        float *mats[]={w[d].proj,w[d].blk.Wq,w[d].blk.Wk,w[d].blk.Wv,w[d].blk.Wo,w[d].blk.W1,w[d].blk.W2,w[d].blk.W3};
        size_t ns[]={2*DIM*DIM,Q_DIM*DIM,KV_DIM*DIM,KV_DIM*DIM,DIM*Q_DIM,HIDDEN*DIM,DIM*HIDDEN,HIDDEN*DIM};
        for (int k=0;k<8;k++) for (size_t i=0;i<ns[k];i++){
            *rng = *rng*6364136223846793005ULL+1442695040888963407ULL;
            float u = ((float)((*rng>>40)&0xFFFFFF)/16777216.0f)*2.0f-1.0f;
            mats[k][i] = u*0.02f;
        }
        memset(&g[d], 0, sizeof(MtpParams));
        a[d].rms_h=adam_alloc(DIM); a[d].rms_e=adam_alloc(DIM);
        a[d].proj=adam_alloc(2*DIM*DIM);
        a[d].Wq=adam_alloc(Q_DIM*DIM); a[d].Wk=adam_alloc(KV_DIM*DIM); a[d].Wv=adam_alloc(KV_DIM*DIM);
        a[d].Wo=adam_alloc(DIM*Q_DIM); a[d].W1=adam_alloc(HIDDEN*DIM); a[d].W2=adam_alloc(DIM*HIDDEN);
        a[d].W3=adam_alloc(HIDDEN*DIM); a[d].rms_att=adam_alloc(DIM); a[d].rms_ffn=adam_alloc(DIM);
    }
}

// Enumerate one depth's 12 tensors in a fixed order (matches mtp_adam_list and the
// AdamState fields). is2d marks the Muon-eligible 2D matrices (proj + block W*).
static int mtp_tensors(MtpParams *p, float **ptr, size_t *sz, int *is2d, int *rows, int *cols) {
    int n=0;
    #define MT(P,S,D,R,C) do{ ptr[n]=(P); sz[n]=(S); is2d[n]=(D); rows[n]=(R); cols[n]=(C); n++; }while(0)
    MT(p->rms_h,DIM,0,0,0); MT(p->rms_e,DIM,0,0,0); MT(p->proj,2*DIM*DIM,1,DIM,2*DIM);
    MT(p->blk.Wq,Q_DIM*DIM,1,Q_DIM,DIM); MT(p->blk.Wk,KV_DIM*DIM,1,KV_DIM,DIM); MT(p->blk.Wv,KV_DIM*DIM,1,KV_DIM,DIM);
    MT(p->blk.Wo,DIM*Q_DIM,1,DIM,Q_DIM); MT(p->blk.W1,HIDDEN*DIM,1,HIDDEN,DIM); MT(p->blk.W2,DIM*HIDDEN,1,DIM,HIDDEN);
    MT(p->blk.W3,HIDDEN*DIM,1,HIDDEN,DIM); MT(p->blk.rms_att,DIM,0,0,0); MT(p->blk.rms_ffn,DIM,0,0,0);
    #undef MT
    return n;
}
static int mtp_adam_list(MtpAdam *a, AdamState **as) {
    int n=0; as[n++]=&a->rms_h; as[n++]=&a->rms_e; as[n++]=&a->proj;
    as[n++]=&a->Wq; as[n++]=&a->Wk; as[n++]=&a->Wv; as[n++]=&a->Wo;
    as[n++]=&a->W1; as[n++]=&a->W2; as[n++]=&a->W3; as[n++]=&a->rms_att; as[n++]=&a->rms_ffn;
    return n;
}
static void mtp_scale_grads(MtpParams *g, float s) {
    for (int d=0; d<MTP_DEPTH; d++){ float *gp[12]; size_t sz[12]; int a[12],b[12],c[12];
        int n=mtp_tensors(&g[d],gp,sz,a,b,c); for(int k=0;k<n;k++) for(size_t i=0;i<sz[k];i++) gp[k][i]*=s; }
}
static float mtp_gradnorm_sq(MtpParams *g) {
    double s=0; for (int d=0; d<MTP_DEPTH; d++){ float *gp[12]; size_t sz[12]; int a[12],b[12],c[12];
        int n=mtp_tensors(&g[d],gp,sz,a,b,c); for(int k=0;k<n;k++) for(size_t i=0;i<sz[k];i++) s+=(double)gp[k][i]*gp[k][i]; }
    return (float)s;
}
static void mtp_zero_grads(MtpParams *g) { for (int d=0; d<MTP_DEPTH; d++) memset(&g[d],0,sizeof(MtpParams)); }

// Optimizer step over all MTP params: Muon for the 2D matrices (when opt_is_muon),
// AdamW for the norm vectors (always, no weight decay) — mirrors the main loop.
static void mtp_optimize(MtpParams *w, MtpParams *g, MtpAdam *a, int opt_is_muon, int muon_is_v4,
                         int adam_t, float lr, float b1, float b2, float eps, float wd) {
    for (int d=0; d<MTP_DEPTH; d++) {
        float *wp[12],*gp[12]; size_t sz[12],szg[12]; int is2d[12],rows[12],cols[12],ig[12],rg[12],cg[12];
        AdamState *as[12];
        mtp_tensors(&w[d], wp, sz, is2d, rows, cols);
        mtp_tensors(&g[d], gp, szg, ig, rg, cg);
        mtp_adam_list(&a[d], as);
        for (int k=0; k<12; k++) {
            if (is2d[k] && opt_is_muon)
                muon_update(wp[k], gp[k], as[k]->m, rows[k], cols[k], lr, 0.95f, 1, muon_is_v4, wd);
            else
                adam_update(wp[k], gp[k], as[k], adam_t, lr, b1, b2, eps, is2d[k]?wd:0.0f);
        }
    }
}
#endif // MTP_DEPTH > 0

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        ane_init();
        mach_timebase_info(&g_tb);

        int total_steps = 10000;
        float max_lr = 3e-4f;
        float adam_b1=0.9f, adam_b2=0.95f, adam_eps=1e-8f, wd=0.1f;
        int adam_t = 0, start_step = 0;
        int accum_steps = 10;
        int warmup_steps = 100;
        float grad_clip = 1.0f;
        float loss_scale = 256.0f;
        float res_alpha = 1.0f / sqrtf(2.0f * NLAYERS);
        float min_lr_frac = 0.1f;

        bool do_resume = false, from_scratch = false, overfit = false;
        const char *data_path = DEFAULT_DATA_PATH;
        const char *init_path = NULL;        // R1: load shared init from flat binary
        const char *dump_grads_path = NULL;  // R1: dump raw grads after one batch, exit
        const char *ckpt_path = CKPT_PATH;   // checkpoint output (runtime override)
        const char *val_data_path = NULL;    // held-out shard for periodic val loss
        int val_every = 0;                   // 0 = no validation
        int val_batches = 20;                // fixed val batch count
        int sample_every = 0;                // ADR 0002: emit faithful samples every K steps (0=off)
        int sample_tokens = 64;              // tokens to generate per sample
        int sample_prompt[SEQ]; int sample_prompt_len = 0;  // default (len 0) = BOS only
        const char *dump_weights_path = NULL;// step-diff: dump post-update weights, exit
        int opt_is_muon = OPTIMIZER_IS_MUON; // optimizer (runtime --opt overrides the header)
        int muon_is_v4 = 1;                  // Muon variant: 1 = V4 hybrid NS + 0.18-RMS rescale (#4),
                                             // 0 = prior Keller-Jordan (--muon-variant prior)
        for (int i=1; i<argc; i++) {
            if (strcmp(argv[i], "--resume") == 0) do_resume = true;
            else if (strcmp(argv[i], "--scratch") == 0) from_scratch = true;
            else if (strcmp(argv[i], "--overfit") == 0) overfit = true;  // R0 gate: pin one batch
            else if (strcmp(argv[i], "--steps") == 0 && i+1<argc) total_steps = atoi(argv[++i]);
            else if (strcmp(argv[i], "--lr") == 0 && i+1<argc) max_lr = atof(argv[++i]);
            else if (strcmp(argv[i], "--wd") == 0 && i+1<argc) wd = atof(argv[++i]);
            else if (strcmp(argv[i], "--accum") == 0 && i+1<argc) accum_steps = atoi(argv[++i]);
            else if (strcmp(argv[i], "--warmup") == 0 && i+1<argc) warmup_steps = atoi(argv[++i]);
            else if (strcmp(argv[i], "--clip") == 0 && i+1<argc) grad_clip = atof(argv[++i]);
            else if (strcmp(argv[i], "--data") == 0 && i+1<argc) data_path = argv[++i];
            else if (strcmp(argv[i], "--init") == 0 && i+1<argc) init_path = argv[++i];
            else if (strcmp(argv[i], "--dump-grads") == 0 && i+1<argc) dump_grads_path = argv[++i];
            else if (strcmp(argv[i], "--ckpt") == 0 && i+1<argc) ckpt_path = argv[++i];
            else if (strcmp(argv[i], "--val-data") == 0 && i+1<argc) val_data_path = argv[++i];
            else if (strcmp(argv[i], "--val-every") == 0 && i+1<argc) val_every = atoi(argv[++i]);
            else if (strcmp(argv[i], "--val-batches") == 0 && i+1<argc) val_batches = atoi(argv[++i]);
            else if (strcmp(argv[i], "--sample-every") == 0 && i+1<argc) sample_every = atoi(argv[++i]);
            else if (strcmp(argv[i], "--sample-tokens") == 0 && i+1<argc) sample_tokens = atoi(argv[++i]);
            else if (strcmp(argv[i], "--sample-prompt-ids") == 0 && i+1<argc) {
                // space-separated full-vocab token ids (the dashboard tokenizes text → ids)
                char *s = argv[++i]; sample_prompt_len = 0;
                for (char *p = strtok(s, " ,"); p && sample_prompt_len < SEQ-1; p = strtok(NULL, " ,"))
                    sample_prompt[sample_prompt_len++] = atoi(p);
            }
            else if (strcmp(argv[i], "--dump-weights") == 0 && i+1<argc) dump_weights_path = argv[++i];
            else if (strcmp(argv[i], "--opt") == 0 && i+1<argc) {
                const char *o = argv[++i];
                if (strcmp(o, "muon") == 0) opt_is_muon = 1;
                else if (strcmp(o, "adamw") == 0) opt_is_muon = 0;
                else { printf("unknown --opt %s (use adamw|muon)\n", o); return 1; }
            }
            else if (strcmp(argv[i], "--muon-variant") == 0 && i+1<argc) {
                const char *v = argv[++i];
                if (strcmp(v, "v4") == 0) muon_is_v4 = 1;
                else if (strcmp(v, "prior") == 0) muon_is_v4 = 0;
                else { printf("unknown --muon-variant %s (use v4|prior)\n", v); return 1; }
            }
        }
        float lr = max_lr;

        // Allocate per-layer state
        LayerWeights lw[NLAYERS]; LayerAdam la[NLAYERS];
        LayerActs acts[NLAYERS]; LayerGrads grads[NLAYERS];
        for (int L=0; L<NLAYERS; L++) {
            lw[L] = layer_weights_alloc(); la[L] = layer_adam_alloc();
            acts[L] = layer_acts_alloc(); grads[L] = layer_grads_alloc();
        }
        float *rms_final = (float*)malloc(DIM*4);
        float *embed = (float*)malloc(VOCAB*DIM*4);
        float *grms_final = (float*)calloc(DIM, 4);
        float *gembed = (float*)calloc(VOCAB*DIM, 4);
        AdamState arms_final = adam_alloc(DIM);
        AdamState aembed = adam_alloc((size_t)VOCAB*DIM);

        // Attention sink (issue #8): one learnable logit per (layer, head). Init 0
        // so exp(sink)=1 from the start — the sink behaves like one extra all-zero
        // key, active from step 0 (no warm-up needed for the R0 gate). AdamW like
        // the other per-element bias/norm params. attn_sink stays allocated even
        // when the knob is off (passed to forward_hidden, used only #if ATTN_SINK).
        // The grad arrays (gsink/gqnorm/gknorm) are always declared — they appear in
        // the backward call's compile-time ternaries (`KNOB ? g... : NULL`) for
        // every build — but the optimizer/checkpoint only touch them #if KNOB, and
        // attn_cpu_backward only writes them when its matching pointer is non-NULL.
        float *attn_sink = (float*)calloc((size_t)NLAYERS*HEADS, 4);
        float *gsink = (float*)calloc((size_t)NLAYERS*HEADS, 4);
#if ATTN_SINK
        AdamState asink = adam_alloc((size_t)NLAYERS*HEADS);
#endif
        // Q/KV RMSNorm (issue #7): per-layer, per-head_dim gain on Q and K, applied
        // just before the scores (post-RoPE). Init 1.0 (active from step 0).
        float *qnorm_w = (float*)malloc((size_t)NLAYERS*HD*4);
        float *knorm_w = (float*)malloc((size_t)NLAYERS*HD*4);
        for (size_t i=0;i<(size_t)NLAYERS*HD;i++){ qnorm_w[i]=1.0f; knorm_w[i]=1.0f; }
        float *gqnorm = (float*)calloc((size_t)NLAYERS*HD, 4);
        float *gknorm = (float*)calloc((size_t)NLAYERS*HD, 4);
#if QK_NORM
        AdamState aqnorm = adam_alloc((size_t)NLAYERS*HD);
        AdamState aknorm = adam_alloc((size_t)NLAYERS*HD);
#endif
#if MTP_DEPTH > 0
        // MTP blocks (issue #6): one extra transformer block + glue per depth.
        // Heap-allocated (each MtpParams is a full block; large at big rungs).
        // Not yet threaded into the checkpoint — resume reinitialises them (the R0
        // gate runs in one process; ANE-by-profile + persistence are follow-ups).
        MtpParams *mtpw = (MtpParams*)malloc(MTP_DEPTH*sizeof(MtpParams));
        MtpParams *mtpg = (MtpParams*)malloc(MTP_DEPTH*sizeof(MtpParams));
        MtpAdam   *mtpa = (MtpAdam*)malloc(MTP_DEPTH*sizeof(MtpAdam));
        { uint64_t mtp_rng = 0x9E3779B97F4A7C15ULL; mtp_alloc_init(mtpw, mtpg, mtpa, &mtp_rng); }
        MtpSaved *mtp_saved = (MtpSaved*)malloc(MTP_DEPTH*sizeof(MtpSaved));
        int mtp_nd = mtp_ndep();
#endif
#if N_HC > 1
        // mHC (issue #11): expand the residual stream to N_HC parallel streams and
        // wrap every sub-layer in dynamic input/residual/output maps (A/B/C). Maps,
        // grads, AdamW state, and forward/backward scratch all live in file-scope
        // globals (see mhc_global_init). Persisted in the v5 checkpoint (ADR 0002):
        // init here first, then load_checkpoint below overwrites on --resume. Inert at
        // N_HC==1 (this whole block compiles out).
        mhc_global_init();
        printf("mHC enabled: N_HC=%d streams, %d sub-layer maps (A=sigmoid, B=Sinkhorn, C=2*sigmoid)\n",
               N_HC, 2*NLAYERS);
#endif

        double cum_train=0, cum_wall=0; int cum_steps=0;
        float resume_loss = 0;
        bool resuming = false;
        if (do_resume) {
            resuming = load_checkpoint(ckpt_path, &start_step, &total_steps, &lr, &resume_loss,
                &cum_train, &cum_wall, &cum_steps, &adam_t,
                lw, la, rms_final, &arms_final, embed, &aembed
#if ATTN_SINK
                , attn_sink, &asink
#endif
#if QK_NORM
                , qnorm_w, &aqnorm, knorm_w, &aknorm
#endif
                );
            if (resuming) printf("[RESUMED step %d, loss=%.4f]\n", start_step, resume_loss);
        }
        if (!resuming) {
            printf("=== ANE Dynamic Training: %s (%d layers, GQA %d/%d heads) ===\n",
                   MODEL_NAME, NLAYERS, HEADS, KV_HEADS);
            printf("dim=%d q_dim=%d kv_dim=%d hd=%d hidden=%d seq=%d vocab=%d\n",
                   DIM, Q_DIM, KV_DIM, HD, HIDDEN, SEQ, VOCAB);
            double xformer_m = (double)NLAYERS*(WQ_SZ + WK_SZ + WV_SZ + (double)WO_SZ + W1_SZ + W2_SZ + W3_SZ + 2.0*DIM) / 1e6;
            double embed_m = (double)VOCAB*DIM / 1e6;
            printf("Params: %.1fM (transformer %.1fM + embed %.1fM)\n", xformer_m+embed_m, xformer_m, embed_m);
            printf("Kernels: 10 compiled (sdpaFwd+woFwd, ffnFused, ffnBwdW2t+W13t, wotBwd, sdpaBwd1+2, qBwd+kvBwd)\n");
            printf("Accum %d steps, LR=%g, optimizer=%s%s\n", accum_steps, max_lr,
                   opt_is_muon ? "muon" : "adamw",
                   opt_is_muon ? (muon_is_v4 ? " (v4 hybrid NS)" : " (prior NS)") : "");
            // V4 ablation knobs (PRD #2). Printed so a run's config is on the
            // record; every knob is off/identity by default → plain transformer.
            printf("V4 knobs: qk_norm=%d attn_sink=%d swiglu_clamp=%d "
                   "rope_rotary_dims=%d (eff %d of HD=%d) n_hc=%d mtp_depth=%d\n",
                   QK_NORM, ATTN_SINK, SWIGLU_CLAMP,
                   ROPE_ROTARY_DIMS, ROPE_ROTARY_EFF, HD, N_HC, MTP_DEPTH);
            double fwd_flops = 2.0*NLAYERS*((double)WQ_SZ + WK_SZ + WV_SZ + WO_SZ + W1_SZ + W2_SZ + W3_SZ) * SEQ;
            double total_flops = 3.0 * fwd_flops;
            printf("FLOPs/step: fwd=%.1fM total=%.1fM\n", fwd_flops/1e6, total_flops/1e6);
            if (from_scratch) {
                printf("  Training from scratch (random init)\n");
                srand48(42);
                float scale_d=1.0f/sqrtf(DIM), scale_qd=1.0f/sqrtf(Q_DIM), scale_h=1.0f/sqrtf(HIDDEN);
                float res_scale = 1.0f/sqrtf(2.0f*NLAYERS);
                for (int L=0; L<NLAYERS; L++) {
                    for(size_t i=0;i<WQ_SZ;i++) lw[L].Wq[i]=scale_d*(2*drand48()-1);
                    for(size_t i=0;i<WK_SZ;i++) lw[L].Wk[i]=scale_d*(2*drand48()-1);
                    for(size_t i=0;i<WV_SZ;i++) lw[L].Wv[i]=scale_d*(2*drand48()-1);
                    for(size_t i=0;i<WO_SZ;i++) lw[L].Wo[i]=scale_qd*res_scale*(2*drand48()-1);
                    for(size_t i=0;i<W1_SZ;i++) lw[L].W1[i]=scale_h*(2*drand48()-1);
                    for(size_t i=0;i<W2_SZ;i++) lw[L].W2[i]=scale_d*res_scale*(2*drand48()-1);
                    for(size_t i=0;i<W3_SZ;i++) lw[L].W3[i]=scale_h*(2*drand48()-1);
                    for(int i=0;i<DIM;i++){lw[L].rms_att[i]=1.0f; lw[L].rms_ffn[i]=1.0f;}
                }
                for(int i=0;i<DIM;i++) rms_final[i]=1.0f;
                float escale = 0.02f;
                for(size_t i=0;i<(size_t)VOCAB*DIM;i++) embed[i]=escale*(2*drand48()-1);
            } else {
                printf("  ERROR: Pretrained weight loading not implemented for Qwen3. Use --scratch.\n");
                return 1;
            }
        }

        // R1: overwrite init with the shared numpy weights (same the twins load),
        // so the gradient diff compares identical models from identical weights.
        if (init_path) {
            if (!load_init_weights(init_path, lw, rms_final, embed)) {
                printf("Cannot load init weights from %s\n", init_path); return 1;
            }
            printf("  [R1: shared init loaded from %s]\n", init_path);
        }

        // Precompute transposed weights for forward/backward kernels
        // Forward: sdpaFwd needs Wq^T[Q_DIM,DIM], Wk^T[KV_DIM,DIM], Wv^T[KV_DIM,DIM]
        //          woFwd needs Wo^T[DIM,Q_DIM]
        // Backward uses original (non-transposed) weights
        float *Wqt_buf[NLAYERS], *Wkt_buf[NLAYERS], *Wvt_buf[NLAYERS], *Wot_buf[NLAYERS];
        float *W1t_buf[NLAYERS], *W2t_buf[NLAYERS], *W3t_buf[NLAYERS];
        for (int L=0; L<NLAYERS; L++) {
            Wqt_buf[L]=(float*)malloc(WQ_SZ*4); Wkt_buf[L]=(float*)malloc(WK_SZ*4);
            Wvt_buf[L]=(float*)malloc(WV_SZ*4); Wot_buf[L]=(float*)malloc(WO_SZ*4);
            W1t_buf[L]=(float*)malloc(W1_SZ*4); W2t_buf[L]=(float*)malloc(W2_SZ*4);
            W3t_buf[L]=(float*)malloc(W3_SZ*4);
            // Wq is [Q_DIM, DIM] → Wq^T is [DIM, Q_DIM] (staged as [DIM channels, Q_DIM spatial])
            transpose_weight(Wqt_buf[L], lw[L].Wq, Q_DIM, DIM);
            // Wk is [KV_DIM, DIM] → Wk^T is [DIM, KV_DIM]
            transpose_weight(Wkt_buf[L], lw[L].Wk, KV_DIM, DIM);
            // Wv is [KV_DIM, DIM] → Wv^T is [DIM, KV_DIM]
            transpose_weight(Wvt_buf[L], lw[L].Wv, KV_DIM, DIM);
            // Wo is [DIM, Q_DIM] → Wo^T is [Q_DIM, DIM]
            transpose_weight(Wot_buf[L], lw[L].Wo, DIM, Q_DIM);
            transpose_weight(W1t_buf[L], lw[L].W1, HIDDEN, DIM);
            transpose_weight(W2t_buf[L], lw[L].W2, DIM, HIDDEN);
            transpose_weight(W3t_buf[L], lw[L].W3, HIDDEN, DIM);
        }

        // mmap token data
        int data_fd = open(data_path, O_RDONLY);
        if (data_fd < 0) { printf("Cannot open %s\n", data_path); return 1; }
        struct stat st; fstat(data_fd, &st);
        size_t data_len = st.st_size;
        uint16_t *token_data = (uint16_t*)mmap(NULL, data_len, PROT_READ, MAP_PRIVATE, data_fd, 0);
        if (token_data == MAP_FAILED) { printf("mmap failed\n"); return 1; }
        size_t n_tokens = data_len / 2;
        printf("Token data: %zu tokens (%.1f MB)\n", n_tokens, data_len/1e6);

        // Vocab compaction
        VocabMap vm = vocab_map_build(token_data, n_tokens, VOCAB);
        int CV = vm.compact_vocab;
        printf("Vocab compaction: %d → %d active tokens (%.1fx reduction)\n", VOCAB, CV, (float)VOCAB/CV);

        float *cembed = vocab_compact_embed(embed, &vm, DIM);
        float *gcembed = (float*)calloc((size_t)CV*DIM, 4);
        AdamState acembed = adam_alloc((size_t)CV*DIM);

        // Held-out validation shard (data01): periodic forward-only val loss.
        uint16_t *val_data = NULL; size_t val_ntokens = 0, val_len = 0; int val_fd = -1;
        if (val_data_path && val_every > 0) {
            val_fd = open(val_data_path, O_RDONLY);
            if (val_fd < 0) { printf("Cannot open val data %s (val disabled)\n", val_data_path); }
            else {
                struct stat vst; fstat(val_fd, &vst); val_len = vst.st_size;
                val_data = (uint16_t*)mmap(NULL, val_len, PROT_READ, MAP_PRIVATE, val_fd, 0);
                if (val_data == MAP_FAILED) { val_data = NULL; printf("val mmap failed (val disabled)\n"); }
                else {
                    val_ntokens = val_len / 2;
                    printf("Val data: %zu tokens from %s (every %d steps, %d batches)\n",
                           val_ntokens, val_data_path, val_every, val_batches);
                }
            }
        }

        // ===== Compile all kernels ONCE =====
        printf("Compiling 10 dynamic kernels (one-time)...\n");
        uint64_t tc = mach_absolute_time();
        DynLayerKernels dk;
        if (!compile_dynamic_kernels(&dk)) {
            printf("Compilation failed!\n"); return 1;
        }
        double compile_ms = tb_ms(mach_absolute_time() - tc);
        printf("Compiled 10 kernels in %.0fms (shared across all %d layers)\n", compile_ms, NLAYERS);

        // Allocate per-layer IOSurfaces + requests
        printf("Allocating per-layer IOSurfaces...\n");
        PerLayerSurfaces pls[NLAYERS];
        PerLayerRequests plr[NLAYERS];
        for (int L = 0; L < NLAYERS; L++) {
            pls[L].sdpaFwd_in    = make_surface(DIM*SDPA_FWD_SP*2);
#if WO_FUNCPARAM
            pls[L].woFwd_in      = make_surface(Q_DIM*SEQ*2);
            pls[L].woFwd_w       = make_surface(Q_DIM*DIM*2);
#else
            pls[L].woFwd_in      = make_surface(Q_DIM*WO_FWD_SP*2);
#endif
            pls[L].ffnFused_in   = make_surface(DIM*FFN_FUSED_SP*2);
#if W2T_FUNCPARAM
            pls[L].ffnBwdW2t_in  = make_surface(DIM*SEQ*2);
            pls[L].ffnBwdW2t_w   = make_surface(DIM*HIDDEN*2);
#else
            pls[L].ffnBwdW2t_in  = make_surface(DIM*FFN_BWD_W2T_SP*2);
#endif
            pls[L].ffnBwdW13t_in = make_surface(HIDDEN*FFN_BWD_W13T_SP*2);
#if CONV_DATAPATH
            pls[L].wotBwd_in     = make_surface(DIM*SEQ*2);
            pls[L].wotBwd_w      = make_surface(DIM*Q_DIM*2);
            pls[L].qBwd_in       = make_surface(Q_DIM*SEQ*2);
            pls[L].qBwd_w        = make_surface(Q_DIM*DIM*2);
#else
            pls[L].wotBwd_in     = make_surface(DIM*WOT_BWD_SP*2);
#if !FUSE_QKVBWD
            pls[L].qBwd_in       = make_surface(Q_DIM*Q_BWD_SP*2);
#endif
#endif
#if FUSE_QKVBWD
            pls[L].qkvBwd_in     = make_surface(DIM*QKV_BWD_SP*2);
#else
            pls[L].kvBwd_in      = make_surface(KV_DIM*KV_BWD_SP*2);
#endif

            plr[L].sdpaFwd   = make_request(dk.sdpaFwd,   pls[L].sdpaFwd_in);
#if WO_FUNCPARAM
            plr[L].woFwd     = make_request_2in(dk.woFwd, pls[L].woFwd_in, pls[L].woFwd_w);
#else
            plr[L].woFwd     = make_request(dk.woFwd,     pls[L].woFwd_in);
#endif
            plr[L].ffnFused  = make_request(dk.ffnFused,  pls[L].ffnFused_in);
#if W2T_FUNCPARAM
            plr[L].ffnBwdW2t = make_request_2in(dk.ffnBwdW2t, pls[L].ffnBwdW2t_in, pls[L].ffnBwdW2t_w);
#else
            plr[L].ffnBwdW2t = make_request(dk.ffnBwdW2t, pls[L].ffnBwdW2t_in);
#endif
            plr[L].ffnBwdW13t= make_request(dk.ffnBwdW13t,pls[L].ffnBwdW13t_in);
#if CONV_DATAPATH
            plr[L].wotBwd    = make_request_2in(dk.wotBwd, pls[L].wotBwd_in, pls[L].wotBwd_w);
            plr[L].qBwd      = make_request_2in(dk.qBwd,   pls[L].qBwd_in,   pls[L].qBwd_w);
#else
            plr[L].wotBwd    = make_request(dk.wotBwd,    pls[L].wotBwd_in);
#if !FUSE_QKVBWD
            plr[L].qBwd      = make_request(dk.qBwd,      pls[L].qBwd_in);
#endif
#endif
#if FUSE_QKVBWD
            plr[L].qkvBwd    = make_request(dk.qkvBwd,    pls[L].qkvBwd_in);
#else
            plr[L].kvBwd     = make_request(dk.kvBwd,     pls[L].kvBwd_in);
#endif
        }

        // Stage weights into per-layer surfaces
        for (int L = 0; L < NLAYERS; L++) {
            stage_sdpa_fwd_weights(pls[L].sdpaFwd_in, Wqt_buf[L], Wkt_buf[L], Wvt_buf[L]);
#if WO_FUNCPARAM
            stage_wo_fwd_w_fp(pls[L].woFwd_w, Wot_buf[L]);
#else
            stage_wo_fwd_weights(pls[L].woFwd_in, Wot_buf[L]);
#endif
            stage_ffn_fused_weights(pls[L].ffnFused_in, W1t_buf[L], W3t_buf[L], lw[L].W2);
#if W2T_FUNCPARAM
#if CONV_PROBE
            { static float w2t_tmp[HIDDEN*DIM]; transpose_weight(w2t_tmp, lw[L].W2, DIM, HIDDEN);
              io_write_fp16_at(pls[L].ffnBwdW2t_w, 0, w2t_tmp, HIDDEN, DIM); }  // conv weight = W2^T [OC,IC,1,1]
#else
            io_write_fp16_at(pls[L].ffnBwdW2t_w, 0, lw[L].W2, DIM, HIDDEN);
#endif
#else
            stage_ffn_bwd_w2t_weights(pls[L].ffnBwdW2t_in, lw[L].W2);
#endif
            stage_ffn_bwd_w13t_weights(pls[L].ffnBwdW13t_in, lw[L].W1, lw[L].W3);
#if CONV_DATAPATH
            { static float t_wot_i[Q_DIM*DIM]; transpose_weight(t_wot_i, lw[L].Wo, DIM, Q_DIM);
              io_write_fp16_at(pls[L].wotBwd_w, 0, t_wot_i, Q_DIM, DIM); }  // conv weight = Wo^T [OC=Q_DIM,IC=DIM,1,1]
            { static float t_q_i[DIM*Q_DIM]; transpose_weight(t_q_i, lw[L].Wq, Q_DIM, DIM);
              io_write_fp16_at(pls[L].qBwd_w, 0, t_q_i, DIM, Q_DIM); }      // conv weight = Wq^T [OC=DIM,IC=Q_DIM,1,1]
#else
#if CONV1IN == 2
            stage_wot_bwd_weights_convB(pls[L].wotBwd_in, lw[L].Wo);
#else
            stage_wot_bwd_weights(pls[L].wotBwd_in, lw[L].Wo);
#endif
#if !FUSE_QKVBWD
            stage_q_bwd_weights(pls[L].qBwd_in, lw[L].Wq);
#endif
#endif
#if FUSE_QKVBWD
            stage_qkv_bwd_weights(pls[L].qkvBwd_in, lw[L].Wq, lw[L].Wk, lw[L].Wv);
#else
            stage_kv_bwd_weights(pls[L].kvBwd_in, lw[L].Wk, lw[L].Wv);
#endif
        }
        printf("Per-layer weight staging complete\n\n");

        // Gradient + work buffers (GQA: Q has Q_DIM, K/V have KV_DIM)
        float *dy = (float*)malloc(SEQ*DIM*4);
        float *dffn = (float*)malloc(SEQ*DIM*4);
        float *dx_ffn = (float*)malloc(SEQ*DIM*4);
        float *dx2 = (float*)malloc(SEQ*DIM*4);
        float *dx_attn = (float*)malloc(SEQ*DIM*4);
        float *dq = (float*)malloc(SEQ*Q_DIM*4);     // Q_DIM for Q grads
        float *dk_buf = (float*)malloc(SEQ*KV_DIM*4); // KV_DIM for K grads
        float *dv = (float*)malloc(SEQ*KV_DIM*4);     // KV_DIM for V grads
        float *da_buf = (float*)malloc(SEQ*Q_DIM*4);  // Q_DIM for attn grads
        float *x_cur = (float*)malloc(SEQ*DIM*4);
        float *x_final = (float*)malloc(SEQ*DIM*4);
        float *xnorm_buf = (float*)malloc(SEQ*DIM*4);
        float *logits = (float*)malloc(SEQ*CV*4);
        float *dlogits = (float*)malloc(SEQ*CV*4);
        float *gate_buf = (float*)malloc(SEQ*HIDDEN*4);
        float *dh1 = (float*)malloc(SEQ*HIDDEN*4);
        float *dh3 = (float*)malloc(SEQ*HIDDEN*4);
        float *dsilu = (float*)malloc(SEQ*HIDDEN*4);
        float *silu_tmp = (float*)malloc(SEQ*HIDDEN*4);
        float *silu_tmp2 = (float*)malloc(SEQ*HIDDEN*4);
        // GQA tile/reduce buffers
        float *k_tiled = (float*)malloc(SEQ*Q_DIM*4);  // KV_DIM → Q_DIM
        float *v_tiled = (float*)malloc(SEQ*Q_DIM*4);
        float *dq_full = (float*)malloc(SEQ*Q_DIM*4);  // from sdpaBwd2
        float *dk_full = (float*)malloc(SEQ*Q_DIM*4);  // from sdpaBwd2 (needs reduce)
        float *dv_full = (float*)malloc(SEQ*Q_DIM*4);  // from sdpaBwd1 (needs reduce)

        dispatch_queue_t dw_q = dispatch_queue_create("dw_cblas", DISPATCH_QUEUE_SERIAL);
        dispatch_group_t dw_grp = dispatch_group_create();

        float last_loss = 999.0f;
        float best_loss = resume_loss > 0 ? resume_loss : 999.0f;
        double total_train_ms = 0;
        int total_steps_done = 0;
        uint64_t t_wall_start = mach_absolute_time();
        srand48(42 + start_step);

        for (int step = start_step; step < total_steps; step++) {
            uint64_t t0, t1, t_step = mach_absolute_time();

            // Periodic validation on the held-out shard. Current weights are
            // staged in the per-layer surfaces; this clobbers x_cur/acts/logits,
            // which the training forward below recomputes from scratch.
            if (val_data && (step % val_every == 0)) {
                float vl = eval_val_loss(&dk, pls, plr, lw, rms_final, embed, cembed,
                    CV, &vm, val_data, val_ntokens, val_batches,
                    acts, x_cur, xnorm_buf, x_final, logits, res_alpha,
                    attn_sink, qnorm_w, knorm_w, dw_grp);
                printf("  [val] step=%d val_loss=%.4f\n", step, vl);
            }

            // Faithful in-trainer sampling (ADR 0002): same forward the model trains
            // with, autoregressive over a prompt. Clobbers x_cur/acts/logits like the
            // val pass; the training forward below recomputes from scratch.
            if (sample_every > 0 && step % sample_every == 0) {
                sample_and_emit(&dk, pls, plr, lw, rms_final, embed, cembed, CV, &vm,
                    acts, x_cur, xnorm_buf, x_final, logits, res_alpha,
                    attn_sink, qnorm_w, knorm_w, dw_grp, step,
                    sample_prompt_len > 0 ? sample_prompt : NULL, sample_prompt_len,
                    sample_tokens, 0.8f, 40);
            }

            // Sample data (overfit: pin one fixed batch so loss must collapse to ~0)
            size_t max_pos = n_tokens - SEQ - 1;
            size_t pos = overfit ? 0 : (size_t)(drand48() * max_pos);
            uint16_t *input_tokens = token_data + pos;
            uint16_t *target_tokens_raw = token_data + pos + 1;

            uint16_t ctargets[SEQ];
            for (int t = 0; t < SEQ; t++) ctargets[t] = (uint16_t)vm.full_to_compact[target_tokens_raw[t]];

            embed_lookup(x_cur, embed, input_tokens, DIM, SEQ);

            FwdTiming ft = {0};
            double t_ane_bwd=0, t_io_bwd=0, t_silu=0, t_rms_bwd=0, t_cls=0, t_dw_copy=0;

            // ===== FORWARD (shared with validation; see forward_hidden) =====
            forward_hidden(&dk, pls, plr, lw, rms_final, acts,
                           x_cur, xnorm_buf, x_final, res_alpha,
                           attn_sink, qnorm_w, knorm_w,
#if N_HC > 1
                           &g_mhc_train,
#else
                           NULL,
#endif
                           dw_grp, &ft);

#if N_HC > 1
            // Acceptance probe: the Sinkhorn residual maps B must be doubly-stochastic.
            if (step == start_step)
                printf("  mHC doubly-stochasticity: max|rowsum-1|,|colsum-1| over all B = %.2e\n",
                       mhc_ds_max_train());
#endif

            // Classifier + loss (CPU)
            t0 = mach_absolute_time();
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        CV, SEQ, DIM, 1.0f, cembed, DIM, x_final, SEQ, 0.0f, logits, SEQ);
            float loss = cross_entropy_loss(dlogits, logits, ctargets, CV, SEQ);
            t_cls += tb_ms(mach_absolute_time() - t0);

#if MTP_DEPTH > 0
            // MTP forward (uses the trunk x_cur) + its backward, run as a
            // self-contained unit before the main backward's async dW so the
            // shared-grad writes (gcembed/gembed/grms_final) don't race. Produces
            // mtp_dtrunk, injected into dy after the main final-RMSNorm backward.
            t0 = mach_absolute_time();
            float mtp_term = mtp_forward(x_cur, target_tokens_raw, ctargets, embed, cembed, CV,
                                         rms_final, mtpw, res_alpha, mtp_saved);
            loss += mtp_term;
            float *mtp_dtrunk = (float*)calloc(SEQ*DIM, 4);
            mtp_backward(mtp_saved, mtp_nd, target_tokens_raw, ctargets, cembed, CV, rms_final,
                         mtpw, res_alpha, loss_scale, mtp_dtrunk, gembed, gcembed, grms_final, mtpg);
            t_cls += tb_ms(mach_absolute_time() - t0);
#endif
            last_loss = loss;

            // ===== BACKWARD =====
            vDSP_vsmul(dlogits, 1, &loss_scale, dlogits, 1, (vDSP_Length)(SEQ*CV));

            // Classifier backward
            t0 = mach_absolute_time();
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        DIM, SEQ, CV, 1.0f, cembed, DIM, dlogits, SEQ, 0.0f, dy, SEQ);
            t_cls += tb_ms(mach_absolute_time() - t0);

            // dEmbed async
            dispatch_group_async(dw_grp, dw_q, ^{
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            CV, DIM, SEQ, 1.0f, dlogits, SEQ, x_final, SEQ, 1.0f, gcembed, DIM);
            });

            // Final RMSNorm backward
            float *dx_rms_final = (float*)calloc(SEQ*DIM, 4);
            rmsnorm_bwd(dx_rms_final, grms_final, dy, x_cur, rms_final, DIM, SEQ);
            memcpy(dy, dx_rms_final, SEQ*DIM*4);
            free(dx_rms_final);

#if MTP_DEPTH > 0
            // Trunk gradient from the MTP heads (depth-1 hp = trunk[:S1]).
            for (int i=0;i<SEQ*DIM;i++) dy[i] += mtp_dtrunk[i];
            free(mtp_dtrunk);
#endif

#if N_HC > 1
            // mHC exit-collapse backward: collapse was x = Σ_i X[i], so each stream's
            // gradient is dy (broadcast). dXw carries the wide gradient through the loop.
            for (int i=0;i<N_HC;i++) memcpy(g_dXw + (size_t)i*DIM*SEQ, dy, (size_t)DIM*SEQ*4);
#endif

            // ===== BACKWARD (28 layers, reverse) =====
            for (int L=NLAYERS-1; L>=0; L--) {
                LayerActs *ac = &acts[L];
                LayerGrads *gr = &grads[L];

#if N_HC > 1
                // mHC FFN recombine backward: dXw (grad wrt layer output) → dXw2 (grad
                // wrt the pre-FFN wide stream) + dfout + dB_f/dC_f. The FFN sub-layer
                // backward then runs on dfout as its upstream (dFout_ffn).
                mhc_recombine_bwd(g_mhc_train.XwF[L], g_mhc_train.FoutF[L], &g_mhc_train.tpF[L],
                                  g_dXw, g_dXw2, g_dfout, g_dB_f, g_dC_f);
                memcpy(dy, g_dfout, (size_t)SEQ*DIM*4);
#endif
                // dffn = alpha * dy
                vDSP_vsmul(dy, 1, &res_alpha, dffn, 1, (vDSP_Length)(SEQ*DIM));

                // FFN backward: dffn @ W2^T → dsilu_raw
                t0 = mach_absolute_time();
#if W2T_FUNCPARAM
                io_write_fp16_at(pls[L].ffnBwdW2t_in, 0, dffn, DIM, SEQ);
#else
                write_ffn_bwd_w2t_acts(pls[L].ffnBwdW2t_in, dffn);
#endif
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval_req(dk.ffnBwdW2t, plr[L].ffnBwdW2t);
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                io_read_dyn(dk.ffnBwdW2t->ioOut, dsilu, HIDDEN, SEQ);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);

                // SiLU derivative (vectorized). gate = silu(h1) * h3, so:
                //   dh3 = dsilu * silu ;  dh1 = dsilu * h3 * silu'(h1)
                t0 = mach_absolute_time();
                {
                    int n = HIDDEN*SEQ;
                    float minus1 = -1.0f, one = 1.0f;
                    vDSP_vsmul(ac->h1, 1, &minus1, silu_tmp, 1, (vDSP_Length)n);
                    vvexpf(silu_tmp, silu_tmp, &n);
                    vDSP_vsadd(silu_tmp, 1, &one, silu_tmp, 1, (vDSP_Length)n);
                    vvrecf(silu_tmp, silu_tmp, &n);  // sig
#if SWIGLU_CLAMP
                    // SwiGLU clamp backward (V4 §4.2.3): gate = min(silu,10) *
                    // clamp(h3,-10,10). Clamped regions pass zero gradient, so
                    //   dh3 = dsilu * min(silu,10)      * [|h3| < 10]
                    //   dh1 = dsilu * clamp(h3,-10,10)  * [silu < 10] * silu'(h1)
                    vDSP_vmul(ac->h1, 1, silu_tmp, 1, gate_buf, 1, (vDSP_Length)n);  // gate_buf = silu
                    // silu'(h1) = sig*(1 + h1*(1-sig)) -> silu_tmp2
                    vDSP_vsadd(silu_tmp, 1, &minus1, silu_tmp2, 1, (vDSP_Length)n);
                    vDSP_vneg(silu_tmp2, 1, silu_tmp2, 1, (vDSP_Length)n);
                    vDSP_vmul(ac->h1, 1, silu_tmp2, 1, silu_tmp2, 1, (vDSP_Length)n);
                    vDSP_vsadd(silu_tmp2, 1, &one, silu_tmp2, 1, (vDSP_Length)n);
                    vDSP_vmul(silu_tmp, 1, silu_tmp2, 1, silu_tmp2, 1, (vDSP_Length)n); // silu'(h1)
                    for (int j = 0; j < n; j++) {
                        float s = gate_buf[j];                              // silu
                        float siluc = s < 10.0f ? s : 10.0f;               // gate cap
                        float gatemask = s < 10.0f ? 1.0f : 0.0f;
                        float h3v = ac->h3[j];
                        float h3c = h3v > 10.0f ? 10.0f : (h3v < -10.0f ? -10.0f : h3v);
                        float linmask = (h3v < 10.0f && h3v > -10.0f) ? 1.0f : 0.0f;
                        float d = dsilu[j];
                        dh3[j] = d * siluc * linmask;
                        dh1[j] = d * h3c * gatemask * silu_tmp2[j];
                    }
#else
                    // Fuse the 9 elementwise vDSP passes into ONE sweep. silu_tmp
                    // holds sig from the setup above. This bucket was memory-
                    // bandwidth-bound on the separate passes (~9 full sweeps of
                    // HIDDEN*SEQ), not compute-bound; collapsing them to a single
                    // read/write pass cuts the traffic. Identical math:
                    //   dh3 = dsilu * silu ;  dh1 = dsilu * h3 * silu'(h1)
                    // with silu = h1*sig and silu'(h1) = sig*(1 + h1*(1-sig)).
                    // Straight-line FMA, no branches/calls -> clang auto-vectorizes.
                    // Op order matches the vDSP passes exactly (1-sig as -(sig-1),
                    // dh1 as (dsilu*h3)*silu'), so the result is bit-identical.
                    (void)minus1; (void)one;
                    for (int j = 0; j < n; j++) {
                        float sig = silu_tmp[j];
                        float h1v = ac->h1[j];
                        float d   = dsilu[j];
                        float siluprime = sig * (h1v * (-(sig - 1.0f)) + 1.0f);
                        dh3[j] = d * (h1v * sig);
                        dh1[j] = (d * ac->h3[j]) * siluprime;
                    }
#endif
                }
                t_silu += tb_ms(mach_absolute_time() - t0);

                // dh1@W1^T + dh3@W3^T → dx_ffn (ANE)
                t0 = mach_absolute_time();
                write_ffn_bwd_w13t_acts(pls[L].ffnBwdW13t_in, dh1, dh3);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval_req(dk.ffnBwdW13t, plr[L].ffnBwdW13t);
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                io_read_dyn(dk.ffnBwdW13t->ioOut, dx_ffn, DIM, SEQ);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);

                // dW FFN async
                t0 = mach_absolute_time();
                float *capt_dffn = (float*)malloc(SEQ*DIM*4); memcpy(capt_dffn, dffn, SEQ*DIM*4);
                float *capt_silu = (float*)malloc(SEQ*HIDDEN*4); memcpy(capt_silu, ac->silu_out, SEQ*HIDDEN*4);
                float *capt_dh1 = (float*)malloc(SEQ*HIDDEN*4); memcpy(capt_dh1, dh1, SEQ*HIDDEN*4);
                float *capt_dh3 = (float*)malloc(SEQ*HIDDEN*4); memcpy(capt_dh3, dh3, SEQ*HIDDEN*4);
                float *capt_x2n = (float*)malloc(SEQ*DIM*4); memcpy(capt_x2n, ac->x2norm, SEQ*DIM*4);
                t_dw_copy += tb_ms(mach_absolute_time() - t0);
                dispatch_group_async(dw_grp, dw_q, ^{
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, HIDDEN, SEQ,
                                1.0f, capt_dffn, SEQ, capt_silu, SEQ, 1.0f, gr->W2, HIDDEN);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, HIDDEN, DIM, SEQ,
                                1.0f, capt_dh1, SEQ, capt_x2n, SEQ, 1.0f, gr->W1, DIM);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, HIDDEN, DIM, SEQ,
                                1.0f, capt_dh3, SEQ, capt_x2n, SEQ, 1.0f, gr->W3, DIM);
                    free(capt_dffn); free(capt_silu); free(capt_dh1); free(capt_dh3); free(capt_x2n);
                });

                // RMSNorm2 backward.  In mHC mode ac->x2 is the collapsed FFN input
                // u_ffn, so dx2 = du_ffn (no plain residual passthrough — the mHC
                // recombine carries the stream coupling instead).
                t0 = mach_absolute_time();
                memset(dx2, 0, SEQ*DIM*4);
                rmsnorm_bwd(dx2, gr->rms_ffn, dx_ffn, ac->x2, lw[L].rms_ffn, DIM, SEQ);
#if N_HC > 1
                // FFN premap backward: du_ffn (dx2) + dB_f/dC_f → map grads + dXw2.
                // Then attention recombine backward: dXw2 → dXw (grad wrt pre-attn wide
                // stream) + dfout(attn). dx2 becomes dFout_attn for the attention path.
                mhc_premap_bwd(g_mhc_train.XwF[L], &g_mapF[L], &g_mhc_train.tpF[L],
                               dx2, g_dB_f, g_dC_f, g_dXw2, &g_gmapF[L]);
                mhc_recombine_bwd(g_mhc_train.XwA[L], g_mhc_train.FoutA[L], &g_mhc_train.tpA[L],
                                  g_dXw2, g_dXw, g_dfout, g_dB_a, g_dC_a);
                memcpy(dx2, g_dfout, (size_t)SEQ*DIM*4);
#else
                for(int i=0;i<SEQ*DIM;i++) dx2[i] += dy[i];
#endif
                t_rms_bwd += tb_ms(mach_absolute_time() - t0);

                // Wo^T backward (ANE): alpha*dx2 @ Wo → da[Q_DIM]
                float *dx2_scaled = (float*)malloc(SEQ*DIM*4);
                vDSP_vsmul(dx2, 1, &res_alpha, dx2_scaled, 1, (vDSP_Length)(SEQ*DIM));
                t0 = mach_absolute_time();
#if CONV_DATAPATH
                io_write_fp16_at(pls[L].wotBwd_in, 0, dx2_scaled, DIM, SEQ);
#else
                write_wot_bwd_acts(pls[L].wotBwd_in, dx2_scaled);
#endif
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval_req(dk.wotBwd, plr[L].wotBwd);
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                io_read_dyn(dk.wotBwd->ioOut, da_buf, Q_DIM, SEQ);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);

                // dWo async: gr->Wo[DIM,Q_DIM] += dx2_scaled[DIM,SEQ] @ attn_out^T[SEQ,Q_DIM]
                t0 = mach_absolute_time();
                float *capt_do = (float*)malloc(SEQ*DIM*4); memcpy(capt_do, dx2_scaled, SEQ*DIM*4);
                free(dx2_scaled);
                float *capt_attn = (float*)malloc(SEQ*Q_DIM*4); memcpy(capt_attn, ac->attn_out, SEQ*Q_DIM*4);
                t_dw_copy += tb_ms(mach_absolute_time() - t0);
                dispatch_group_async(dw_grp, dw_q, ^{
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, Q_DIM, SEQ,
                                1.0f, capt_do, SEQ, capt_attn, SEQ, 1.0f, gr->Wo, Q_DIM);
                    free(capt_do); free(capt_attn);
                });

#if ATTN_CPU
                // CPU attention backward for the sink (#8) and/or QK-norm (#7).
                // da_buf = d/d(attn_out) from wotBwd above; ac->Q/K/V are the same
                // post-RoPE activations the CPU forward consumed. Produces
                // GQA-reduced dq[Q_DIM], dk_buf[KV_DIM], dv[KV_DIM] directly (dq/dk
                // are w.r.t. post-RoPE Q/K, so rope_backward_inplace below still
                // applies) and accumulates the sink / QK-norm-gain gradients (each
                // zeroed with the other grads per step). Bypasses sdpaBwd1/2.
                t0 = mach_absolute_time();
                attn_cpu_backward(da_buf, ac->Q, ac->K, ac->V,
                                  ATTN_SINK ? attn_sink + L*HEADS : NULL,
                                  QK_NORM ? qnorm_w + L*HD : NULL,
                                  QK_NORM ? knorm_w + L*HD : NULL,
                                  dq, dk_buf, dv,
                                  ATTN_SINK ? gsink + L*HEADS : NULL,
                                  QK_NORM ? gqnorm + L*HD : NULL,
                                  QK_NORM ? gknorm + L*HD : NULL, SEQ);
                t_rms_bwd += tb_ms(mach_absolute_time() - t0);
#else
                // GQA: tile K,V from KV_DIM → Q_DIM for SDPA backward
                t0 = mach_absolute_time();
                gqa_tile_kv(k_tiled, ac->K, SEQ);
                gqa_tile_kv(v_tiled, ac->V, SEQ);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);

                // SDPA backward part 1: Q[Q_DIM],K_tiled[Q_DIM],V_tiled[Q_DIM],da[Q_DIM] → dV_full[Q_DIM],probs,dp
                t0 = mach_absolute_time();
                io_write_fp16_at(dk.sdpaBwd1->ioIn, 0,       ac->Q,    Q_DIM, SEQ);
                io_write_fp16_at(dk.sdpaBwd1->ioIn, Q_DIM,   k_tiled,  Q_DIM, SEQ);
                io_write_fp16_at(dk.sdpaBwd1->ioIn, 2*Q_DIM, v_tiled,  Q_DIM, SEQ);
                io_write_fp16_at(dk.sdpaBwd1->ioIn, 3*Q_DIM, da_buf,   Q_DIM, SEQ);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval(dk.sdpaBwd1);
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);

                // SDPA backward part 2: probs,dp,Q[Q_DIM],K_tiled[Q_DIM] → dQ[Q_DIM],dK_full[Q_DIM]
                t0 = mach_absolute_time();
                io_copy(dk.sdpaBwd2->ioIn, 0, dk.sdpaBwd1->ioOut, Q_DIM, 2*SCORE_CH, SEQ);
                io_write_fp16_at(dk.sdpaBwd2->ioIn, 2*SCORE_CH,       ac->Q,   Q_DIM, SEQ);
                io_write_fp16_at(dk.sdpaBwd2->ioIn, 2*SCORE_CH+Q_DIM, k_tiled, Q_DIM, SEQ);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval(dk.sdpaBwd2);
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);

                // Read SDPA backward outputs
                t0 = mach_absolute_time();
                io_read_fp16(dk.sdpaBwd2->ioOut, dq_full, 0,     Q_DIM, SEQ);  // dQ at full HEADS
                io_read_fp16(dk.sdpaBwd2->ioOut, dk_full, Q_DIM, Q_DIM, SEQ);  // dK at full HEADS
                io_read_fp16(dk.sdpaBwd1->ioOut, dv_full, 0,     Q_DIM, SEQ);  // dV at full HEADS
                t_io_bwd += tb_ms(mach_absolute_time() - t0);

                // GQA: reduce dK, dV from Q_DIM (HEADS) → KV_DIM (KV_HEADS)
                gqa_reduce_kv(dk_buf, dk_full, SEQ);
                gqa_reduce_kv(dv, dv_full, SEQ);
                // dQ stays at Q_DIM — no reduction needed
                memcpy(dq, dq_full, SEQ*Q_DIM*4);
#endif

                // RoPE backward on dQ[Q_DIM] and dK[KV_DIM]
                rope_backward_inplace(dq, SEQ, Q_DIM, HD);
                rope_backward_inplace(dk_buf, SEQ, KV_DIM, HD);

                if (L == 0 && step % 10 == 0) {
                    float dqmx, dkmx, dvmx;
                    vDSP_maxmgv(dq, 1, &dqmx, (vDSP_Length)(SEQ*Q_DIM));
                    vDSP_maxmgv(dk_buf, 1, &dkmx, (vDSP_Length)(SEQ*KV_DIM));
                    vDSP_maxmgv(dv, 1, &dvmx, (vDSP_Length)(SEQ*KV_DIM));
                    printf("    L0 sdpa_bwd: |dq|=%.6f |dk|=%.6f |dv|=%.6f\n", dqmx, dkmx, dvmx);
                }

                // dWq/dWk/dWv async
                // dWq[Q_DIM,DIM] += dq[Q_DIM,SEQ] @ xnorm^T[SEQ,DIM]
                // dWk[KV_DIM,DIM] += dk[KV_DIM,SEQ] @ xnorm^T[SEQ,DIM]
                // dWv[KV_DIM,DIM] += dv[KV_DIM,SEQ] @ xnorm^T[SEQ,DIM]
                t0 = mach_absolute_time();
                float *capt_dq = (float*)malloc(SEQ*Q_DIM*4); memcpy(capt_dq, dq, SEQ*Q_DIM*4);
                float *capt_dk = (float*)malloc(SEQ*KV_DIM*4); memcpy(capt_dk, dk_buf, SEQ*KV_DIM*4);
                float *capt_dv = (float*)malloc(SEQ*KV_DIM*4); memcpy(capt_dv, dv, SEQ*KV_DIM*4);
                float *capt_xn = (float*)malloc(SEQ*DIM*4); memcpy(capt_xn, ac->xnorm, SEQ*DIM*4);
                t_dw_copy += tb_ms(mach_absolute_time() - t0);
                dispatch_group_async(dw_grp, dw_q, ^{
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Q_DIM, DIM, SEQ,
                                1.0f, capt_dq, SEQ, capt_xn, SEQ, 1.0f, gr->Wq, DIM);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, KV_DIM, DIM, SEQ,
                                1.0f, capt_dk, SEQ, capt_xn, SEQ, 1.0f, gr->Wk, DIM);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, KV_DIM, DIM, SEQ,
                                1.0f, capt_dv, SEQ, capt_xn, SEQ, 1.0f, gr->Wv, DIM);
                    free(capt_dq); free(capt_dk); free(capt_dv); free(capt_xn);
                });

#if FUSE_QKVBWD
                // Fused QKV backward (ANE, MHA-only): one eval computes
                // dx_attn = dq@Wq + dk@Wk + dv@Wv (summed in-kernel), replacing
                // the split qBwd + kvBwd evals AND the CPU dx_attn += dx_kv add.
                t0 = mach_absolute_time();
                write_qkv_bwd_acts(pls[L].qkvBwd_in, dq, dk_buf, dv);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval_req(dk.qkvBwd, plr[L].qkvBwd);
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                io_read_dyn(dk.qkvBwd->ioOut, dx_attn, DIM, SEQ);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
#else
                // Q backward (ANE): dq[Q_DIM] @ Wq → dx_q[DIM]
                t0 = mach_absolute_time();
#if CONV_DATAPATH
                io_write_fp16_at(pls[L].qBwd_in, 0, dq, Q_DIM, SEQ);
#else
                write_q_bwd_acts(pls[L].qBwd_in, dq);
#endif
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval_req(dk.qBwd, plr[L].qBwd);
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                io_read_dyn(dk.qBwd->ioOut, dx_attn, DIM, SEQ);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);

                // KV backward (ANE): dk[KV_DIM]@Wk + dv[KV_DIM]@Wv → dx_kv[DIM]
                float *dx_kv = (float*)malloc(SEQ*DIM*4);
                t0 = mach_absolute_time();
                write_kv_bwd_acts(pls[L].kvBwd_in, dk_buf, dv);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval_req(dk.kvBwd, plr[L].kvBwd);
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                io_read_dyn(dk.kvBwd->ioOut, dx_kv, DIM, SEQ);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);

                // dx_attn = dx_q + dx_kv
                for(int i=0; i<SEQ*DIM; i++) dx_attn[i] += dx_kv[i];
                free(dx_kv);
#endif

                // RMSNorm1 backward.  In mHC mode ac->layer_in is the collapsed
                // attention input u_attn, so dx_rms1 = du_attn (no plain passthrough).
                t0 = mach_absolute_time();
                float *dx_rms1 = (float*)calloc(SEQ*DIM, 4);
                rmsnorm_bwd(dx_rms1, gr->rms_att, dx_attn, ac->layer_in, lw[L].rms_att, DIM, SEQ);
#if N_HC > 1
                // Attention premap backward: du_attn (dx_rms1) + dB_a/dC_a → map grads
                // + dXw. dXw now holds the gradient wrt this layer's input wide stream,
                // carrying to the next (earlier) reverse layer.
                mhc_premap_bwd(g_mhc_train.XwA[L], &g_mapA[L], &g_mhc_train.tpA[L],
                               dx_rms1, g_dB_a, g_dC_a, g_dXw, &g_gmapA[L]);
#else
                for(int i=0;i<SEQ*DIM;i++) dy[i] = dx_rms1[i] + dx2[i];
#endif
                free(dx_rms1);
                t_rms_bwd += tb_ms(mach_absolute_time() - t0);
            }

#if N_HC > 1
            // mHC entry-broadcast backward: the embed went into every stream, so the
            // embedding gradient is the sum over streams of the input wide gradient.
            memset(dy, 0, (size_t)DIM*SEQ*4);
            for (int i=0;i<N_HC;i++)
                vDSP_vadd(dy, 1, g_dXw + (size_t)i*DIM*SEQ, 1, dy, 1, (vDSP_Length)(DIM*SEQ));
#endif

            // Embedding backward
            dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);
            embed_backward(gembed, dy, input_tokens, DIM, SEQ);

            double step_ms = tb_ms(mach_absolute_time() - t_step);
            total_train_ms += step_ms;
            total_steps_done++;

            if (step % 10 == 0 || step == start_step) {
                printf("  timing: ane_fwd=%.1f io_fwd=%.1f rms=%.1f ane_bwd=%.1f io_bwd=%.1f silu=%.1f rms_bwd=%.1f cls=%.1f cblas_wait=%.1f dw_copy=%.1f\n",
                       ft.ane_fwd, ft.io_fwd, ft.rms, t_ane_bwd, t_io_bwd, t_silu, t_rms_bwd, t_cls, ft.cblas_wait, t_dw_copy);
                float xmx, xmn;
                vDSP_maxv(x_cur,1,&xmx,(vDSP_Length)(SEQ*DIM));
                vDSP_minv(x_cur,1,&xmn,(vDSP_Length)(SEQ*DIM));
                float dmx, dmn;
                vDSP_maxv(dy,1,&dmx,(vDSP_Length)(SEQ*DIM));
                vDSP_minv(dy,1,&dmn,(vDSP_Length)(SEQ*DIM));
                printf("step %-4d loss=%.4f  lr=%.2e  %.1fms/step  x[%.2f,%.2f] dy[%.3e,%.3e]\n",
                       step, loss, lr, step_ms, xmn, xmx, dmn, dmx);
            }

            // Adam update every accum_steps
            if ((step+1) % accum_steps == 0 || step == total_steps-1) {
                dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);
                float gsc = 1.0f / (accum_steps * loss_scale);
                adam_t++;

                // Scale gradients
                for (int L=0; L<NLAYERS; L++) {
                    LayerGrads *g = &grads[L];
                    for(size_t i=0;i<WQ_SZ;i++) g->Wq[i]*=gsc;
                    for(size_t i=0;i<WK_SZ;i++) g->Wk[i]*=gsc;
                    for(size_t i=0;i<WV_SZ;i++) g->Wv[i]*=gsc;
                    for(size_t i=0;i<WO_SZ;i++) g->Wo[i]*=gsc;
                    for(size_t i=0;i<W1_SZ;i++) g->W1[i]*=gsc;
                    for(size_t i=0;i<W2_SZ;i++) g->W2[i]*=gsc;
                    for(size_t i=0;i<W3_SZ;i++) g->W3[i]*=gsc;
                    for(int i=0;i<DIM;i++){g->rms_att[i]*=gsc; g->rms_ffn[i]*=gsc;}
                }
                for(int i=0;i<DIM;i++) grms_final[i]*=gsc;
#if ATTN_SINK
                for(int i=0;i<NLAYERS*HEADS;i++) gsink[i]*=gsc;
#endif
#if QK_NORM
                for(int i=0;i<NLAYERS*HD;i++){ gqnorm[i]*=gsc; gknorm[i]*=gsc; }
#endif
#if MTP_DEPTH > 0
                mtp_scale_grads(mtpg, gsc);
#endif
#if N_HC > 1
                mhc_scale_grads(gsc);
#endif
                vocab_scatter_grads(gembed, gcembed, &vm, DIM);
                for(size_t i=0;i<(size_t)VOCAB*DIM;i++) gembed[i]*=gsc;

                // R1 gate: grads are now raw (loss_scale cancelled, LM-head folded
                // into gembed) but pre-clip and pre-Adam — the exact ∇ the oracle
                // computes. Dump here. Exit immediately unless --dump-weights also
                // asked for the post-optimizer weights (the step-diff harness wants
                // both: the same grads in, the updated weights out).
                if (dump_grads_path) {
                    dump_grads(dump_grads_path, grads, grms_final, gembed);
                    printf("  [R1: raw grads dumped to %s after one batch]\n", dump_grads_path);
                    if (!dump_weights_path) {
                        munmap(token_data, data_len); close(data_fd);
                        return 0;
                    }
                }

                // Global gradient norm
                float grad_norm_sq = 0;
                for (int L=0; L<NLAYERS; L++) {
                    LayerGrads *g = &grads[L];
                    float s;
                    vDSP_dotpr(g->Wq,1,g->Wq,1,&s,(vDSP_Length)WQ_SZ); grad_norm_sq+=s;
                    vDSP_dotpr(g->Wk,1,g->Wk,1,&s,(vDSP_Length)WK_SZ); grad_norm_sq+=s;
                    vDSP_dotpr(g->Wv,1,g->Wv,1,&s,(vDSP_Length)WV_SZ); grad_norm_sq+=s;
                    vDSP_dotpr(g->Wo,1,g->Wo,1,&s,(vDSP_Length)WO_SZ); grad_norm_sq+=s;
                    vDSP_dotpr(g->W1,1,g->W1,1,&s,(vDSP_Length)W1_SZ); grad_norm_sq+=s;
                    vDSP_dotpr(g->W2,1,g->W2,1,&s,(vDSP_Length)W2_SZ); grad_norm_sq+=s;
                    vDSP_dotpr(g->W3,1,g->W3,1,&s,(vDSP_Length)W3_SZ); grad_norm_sq+=s;
                    vDSP_dotpr(g->rms_att,1,g->rms_att,1,&s,(vDSP_Length)DIM); grad_norm_sq+=s;
                    vDSP_dotpr(g->rms_ffn,1,g->rms_ffn,1,&s,(vDSP_Length)DIM); grad_norm_sq+=s;
                }
                { float s;
                  vDSP_dotpr(grms_final,1,grms_final,1,&s,(vDSP_Length)DIM); grad_norm_sq+=s;
                  vDSP_dotpr(gembed,1,gembed,1,&s,(vDSP_Length)(VOCAB*DIM)); grad_norm_sq+=s;
#if ATTN_SINK
                  vDSP_dotpr(gsink,1,gsink,1,&s,(vDSP_Length)(NLAYERS*HEADS)); grad_norm_sq+=s;
#endif
#if QK_NORM
                  vDSP_dotpr(gqnorm,1,gqnorm,1,&s,(vDSP_Length)(NLAYERS*HD)); grad_norm_sq+=s;
                  vDSP_dotpr(gknorm,1,gknorm,1,&s,(vDSP_Length)(NLAYERS*HD)); grad_norm_sq+=s;
#endif
#if MTP_DEPTH > 0
                  grad_norm_sq += mtp_gradnorm_sq(mtpg);
#endif
#if N_HC > 1
                  grad_norm_sq += mhc_gradnorm_sq();
#endif
                }
                float grad_norm = sqrtf(grad_norm_sq);
                if ((step+1) % 10 == 0) {
                    float attn_sq=0, ffn_sq=0, embed_sq=0;
                    for (int L=0; L<NLAYERS; L++) {
                        LayerGrads *g = &grads[L]; float s;
                        vDSP_dotpr(g->Wq,1,g->Wq,1,&s,(vDSP_Length)WQ_SZ); attn_sq+=s;
                        vDSP_dotpr(g->Wk,1,g->Wk,1,&s,(vDSP_Length)WK_SZ); attn_sq+=s;
                        vDSP_dotpr(g->Wv,1,g->Wv,1,&s,(vDSP_Length)WV_SZ); attn_sq+=s;
                        vDSP_dotpr(g->Wo,1,g->Wo,1,&s,(vDSP_Length)WO_SZ); attn_sq+=s;
                        vDSP_dotpr(g->W1,1,g->W1,1,&s,(vDSP_Length)W1_SZ); ffn_sq+=s;
                        vDSP_dotpr(g->W2,1,g->W2,1,&s,(vDSP_Length)W2_SZ); ffn_sq+=s;
                        vDSP_dotpr(g->W3,1,g->W3,1,&s,(vDSP_Length)W3_SZ); ffn_sq+=s;
                    }
                    { float s;
                      vDSP_dotpr(gembed,1,gembed,1,&s,(vDSP_Length)(VOCAB*DIM)); embed_sq=s;
                    }
                    printf("  grad_norm=%.4f  attn=%.4f ffn=%.4f embed=%.4f\n",
                           grad_norm, sqrtf(attn_sq), sqrtf(ffn_sq), sqrtf(embed_sq));
                }

                // Gradient clipping
                if (grad_clip > 0 && grad_norm > grad_clip) {
                    float clip_scale = grad_clip / grad_norm;
                    for (int L=0; L<NLAYERS; L++) {
                        LayerGrads *g = &grads[L];
                        vDSP_vsmul(g->Wq,1,&clip_scale,g->Wq,1,(vDSP_Length)WQ_SZ);
                        vDSP_vsmul(g->Wk,1,&clip_scale,g->Wk,1,(vDSP_Length)WK_SZ);
                        vDSP_vsmul(g->Wv,1,&clip_scale,g->Wv,1,(vDSP_Length)WV_SZ);
                        vDSP_vsmul(g->Wo,1,&clip_scale,g->Wo,1,(vDSP_Length)WO_SZ);
                        vDSP_vsmul(g->W1,1,&clip_scale,g->W1,1,(vDSP_Length)W1_SZ);
                        vDSP_vsmul(g->W2,1,&clip_scale,g->W2,1,(vDSP_Length)W2_SZ);
                        vDSP_vsmul(g->W3,1,&clip_scale,g->W3,1,(vDSP_Length)W3_SZ);
                        vDSP_vsmul(g->rms_att,1,&clip_scale,g->rms_att,1,(vDSP_Length)DIM);
                        vDSP_vsmul(g->rms_ffn,1,&clip_scale,g->rms_ffn,1,(vDSP_Length)DIM);
                    }
                    vDSP_vsmul(grms_final,1,&clip_scale,grms_final,1,(vDSP_Length)DIM);
                    vDSP_vsmul(gembed,1,&clip_scale,gembed,1,(vDSP_Length)(VOCAB*DIM));
#if ATTN_SINK
                    vDSP_vsmul(gsink,1,&clip_scale,gsink,1,(vDSP_Length)(NLAYERS*HEADS));
#endif
#if QK_NORM
                    vDSP_vsmul(gqnorm,1,&clip_scale,gqnorm,1,(vDSP_Length)(NLAYERS*HD));
                    vDSP_vsmul(gknorm,1,&clip_scale,gknorm,1,(vDSP_Length)(NLAYERS*HD));
#endif
#if MTP_DEPTH > 0
                    mtp_scale_grads(mtpg, clip_scale);
#endif
#if N_HC > 1
                    mhc_scale_grads(clip_scale);
#endif
                }

                // Cosine LR schedule with warmup
                if (step < warmup_steps) {
                    lr = max_lr * ((float)(step + 1)) / warmup_steps;
                } else {
                    float decay_ratio = (float)(step - warmup_steps) / (float)(total_steps - warmup_steps);
                    float min_lr = max_lr * min_lr_frac;
                    lr = min_lr + 0.5f * (1.0f + cosf(M_PI * decay_ratio)) * (max_lr - min_lr);
                }

                // Optimizer update. Muon (Newton-Schulz) for the 2D weight
                // matrices, AdamW for the norm vectors and (below) the embedding —
                // matches is_muon_param in lilbro/mlx_ref/params.py. The Muon
                // momentum buffer reuses the Adam m-slot (unused for these params
                // in Muon mode). AdamW everywhere is the existing control path.
                // Norms are excluded from weight decay (0.0f) on both paths.
                for (int L=0; L<NLAYERS; L++) {
                    LayerGrads *g = &grads[L];
                    if (opt_is_muon) {
                        muon_update(lw[L].Wq, g->Wq, la[L].Wq.m, Q_DIM,  DIM,    lr, 0.95f, 1, muon_is_v4, wd);
                        muon_update(lw[L].Wk, g->Wk, la[L].Wk.m, KV_DIM, DIM,    lr, 0.95f, 1, muon_is_v4, wd);
                        muon_update(lw[L].Wv, g->Wv, la[L].Wv.m, KV_DIM, DIM,    lr, 0.95f, 1, muon_is_v4, wd);
                        muon_update(lw[L].Wo, g->Wo, la[L].Wo.m, DIM,    Q_DIM,  lr, 0.95f, 1, muon_is_v4, wd);
                        muon_update(lw[L].W1, g->W1, la[L].W1.m, HIDDEN, DIM,    lr, 0.95f, 1, muon_is_v4, wd);
                        muon_update(lw[L].W2, g->W2, la[L].W2.m, DIM,    HIDDEN, lr, 0.95f, 1, muon_is_v4, wd);
                        muon_update(lw[L].W3, g->W3, la[L].W3.m, HIDDEN, DIM,    lr, 0.95f, 1, muon_is_v4, wd);
                    } else {
                        adam_update(lw[L].Wq, g->Wq, &la[L].Wq, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                        adam_update(lw[L].Wk, g->Wk, &la[L].Wk, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                        adam_update(lw[L].Wv, g->Wv, &la[L].Wv, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                        adam_update(lw[L].Wo, g->Wo, &la[L].Wo, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                        adam_update(lw[L].W1, g->W1, &la[L].W1, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                        adam_update(lw[L].W2, g->W2, &la[L].W2, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                        adam_update(lw[L].W3, g->W3, &la[L].W3, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                    }
                    adam_update(lw[L].rms_att, g->rms_att, &la[L].rms_att, adam_t, lr, adam_b1, adam_b2, adam_eps, 0.0f);
                    adam_update(lw[L].rms_ffn, g->rms_ffn, &la[L].rms_ffn, adam_t, lr, adam_b1, adam_b2, adam_eps, 0.0f);

                    // Update transposed weight buffers
                    transpose_weight(Wqt_buf[L], lw[L].Wq, Q_DIM, DIM);
                    transpose_weight(Wkt_buf[L], lw[L].Wk, KV_DIM, DIM);
                    transpose_weight(Wvt_buf[L], lw[L].Wv, KV_DIM, DIM);
                    transpose_weight(Wot_buf[L], lw[L].Wo, DIM, Q_DIM);
                    transpose_weight(W1t_buf[L], lw[L].W1, HIDDEN, DIM);
                    transpose_weight(W2t_buf[L], lw[L].W2, DIM, HIDDEN);
                    transpose_weight(W3t_buf[L], lw[L].W3, HIDDEN, DIM);

                    // Re-stage weights
                    stage_sdpa_fwd_weights(pls[L].sdpaFwd_in, Wqt_buf[L], Wkt_buf[L], Wvt_buf[L]);
#if WO_FUNCPARAM
                    stage_wo_fwd_w_fp(pls[L].woFwd_w, Wot_buf[L]);
#else
                    stage_wo_fwd_weights(pls[L].woFwd_in, Wot_buf[L]);
#endif
                    stage_ffn_fused_weights(pls[L].ffnFused_in, W1t_buf[L], W3t_buf[L], lw[L].W2);
#if W2T_FUNCPARAM
#if CONV_PROBE
                    { static float w2t_rs[HIDDEN*DIM]; transpose_weight(w2t_rs, lw[L].W2, DIM, HIDDEN);
                      io_write_fp16_at(pls[L].ffnBwdW2t_w, 0, w2t_rs, HIDDEN, DIM); }
#else
                    io_write_fp16_at(pls[L].ffnBwdW2t_w, 0, lw[L].W2, DIM, HIDDEN);
#endif
#else
                    stage_ffn_bwd_w2t_weights(pls[L].ffnBwdW2t_in, lw[L].W2);
#endif
                    stage_ffn_bwd_w13t_weights(pls[L].ffnBwdW13t_in, lw[L].W1, lw[L].W3);
#if CONV_DATAPATH
                    { static float t_wot_r[Q_DIM*DIM]; transpose_weight(t_wot_r, lw[L].Wo, DIM, Q_DIM);
                      io_write_fp16_at(pls[L].wotBwd_w, 0, t_wot_r, Q_DIM, DIM); }
                    { static float t_q_r[DIM*Q_DIM]; transpose_weight(t_q_r, lw[L].Wq, Q_DIM, DIM);
                      io_write_fp16_at(pls[L].qBwd_w, 0, t_q_r, DIM, Q_DIM); }
#else
#if CONV1IN == 2
                    stage_wot_bwd_weights_convB(pls[L].wotBwd_in, lw[L].Wo);
#else
                    stage_wot_bwd_weights(pls[L].wotBwd_in, lw[L].Wo);
#endif
#if !FUSE_QKVBWD
                    stage_q_bwd_weights(pls[L].qBwd_in, lw[L].Wq);
#endif
#endif
#if FUSE_QKVBWD
                    stage_qkv_bwd_weights(pls[L].qkvBwd_in, lw[L].Wq, lw[L].Wk, lw[L].Wv);
#else
                    stage_kv_bwd_weights(pls[L].kvBwd_in, lw[L].Wk, lw[L].Wv);
#endif
                }
                adam_update(rms_final, grms_final, &arms_final, adam_t, lr, adam_b1, adam_b2, adam_eps, 0.0f);
                adam_update(embed, gembed, &aembed, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
#if ATTN_SINK
                // Sink logits are bias-like scalars → AdamW with no weight decay.
                adam_update(attn_sink, gsink, &asink, adam_t, lr, adam_b1, adam_b2, adam_eps, 0.0f);
#endif
#if QK_NORM
                // RMSNorm gains → AdamW, no weight decay (like the other norm gains).
                adam_update(qnorm_w, gqnorm, &aqnorm, adam_t, lr, adam_b1, adam_b2, adam_eps, 0.0f);
                adam_update(knorm_w, gknorm, &aknorm, adam_t, lr, adam_b1, adam_b2, adam_eps, 0.0f);
#endif
#if MTP_DEPTH > 0
                mtp_optimize(mtpw, mtpg, mtpa, opt_is_muon, muon_is_v4, adam_t,
                             lr, adam_b1, adam_b2, adam_eps, wd);
#endif
#if N_HC > 1
                mhc_optimize(adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
#endif
                free(cembed);
                cembed = vocab_compact_embed(embed, &vm, DIM);

                // Step-diff: weights now hold exactly one optimizer step applied
                // to the (dumped) grads from the shared init. Dump and exit.
                if (dump_weights_path) {
                    dump_flat_weights(dump_weights_path, lw, rms_final, embed);
                    printf("  [step-diff: post-%s weights dumped to %s after one step]\n",
                           opt_is_muon ? "muon" : "adamw", dump_weights_path);
                    munmap(token_data, data_len); close(data_fd);
                    return 0;
                }

                // Zero grads
                for (int L=0; L<NLAYERS; L++) layer_grads_zero(&grads[L]);
                memset(grms_final, 0, DIM*4);
#if ATTN_SINK
                memset(gsink, 0, (size_t)NLAYERS*HEADS*4);
#endif
#if QK_NORM
                memset(gqnorm, 0, (size_t)NLAYERS*HD*4);
                memset(gknorm, 0, (size_t)NLAYERS*HD*4);
#endif
#if MTP_DEPTH > 0
                mtp_zero_grads(mtpg);
#endif
#if N_HC > 1
                mhc_zero_grads();
#endif
                memset(gembed, 0, (size_t)VOCAB*DIM*4);
                memset(gcembed, 0, (size_t)CV*DIM*4);

                // Checkpoint — only save on best loss
                if ((step+1) % 100 == 0 && last_loss < best_loss) {
                    best_loss = last_loss;
                    double wall = tb_ms(mach_absolute_time() - t_wall_start);
                    save_checkpoint(ckpt_path, step+1, total_steps, lr, last_loss,
                        total_train_ms+cum_train, wall+cum_wall, total_steps_done+cum_steps, adam_t,
                        lw, la, rms_final, &arms_final, embed, &aembed
#if ATTN_SINK
                        , attn_sink, &asink
#endif
#if QK_NORM
                        , qnorm_w, &aqnorm, knorm_w, &aknorm
#endif
                        );
                    printf("  [ckpt saved, best_loss=%.4f]\n", best_loss);
                }
            }
        }

        // Report
        double wall = tb_ms(mach_absolute_time() - t_wall_start);
        printf("\n=== Efficiency Report ===\n");
        printf("Total steps:  %d\n", total_steps_done);
        printf("Compile:      %.0fms (one-time, %.1f%%)\n", compile_ms, 100*compile_ms/(wall+cum_wall));
        printf("Train time:   %.0fms (%.1fms/step)\n", total_train_ms, total_train_ms/total_steps_done);
        printf("Wall time:    %.1fs\n", (wall+cum_wall)/1000);

        // Cleanup
        for (int L=0; L<NLAYERS; L++) {
            layer_weights_free(&lw[L]); layer_adam_free(&la[L]);
            layer_acts_free(&acts[L]); layer_grads_free(&grads[L]);
            free(Wqt_buf[L]); free(Wkt_buf[L]); free(Wvt_buf[L]); free(Wot_buf[L]);
            free(W1t_buf[L]); free(W2t_buf[L]); free(W3t_buf[L]);
        }
        free_per_layer(pls, plr);
        free_kern(dk.sdpaFwd); free_kern(dk.woFwd); free_kern(dk.ffnFused);
        free_kern(dk.ffnBwdW2t); free_kern(dk.ffnBwdW13t); free_kern(dk.wotBwd);
        free_kern(dk.sdpaBwd1); free_kern(dk.sdpaBwd2);
#if FUSE_QKVBWD
        free_kern(dk.qkvBwd);
#else
        free_kern(dk.qBwd); free_kern(dk.kvBwd);
#endif
        free(da_buf); free(k_tiled); free(v_tiled);
        free(dq_full); free(dk_full); free(dv_full);
        free(dq); free(dk_buf); free(dv);
        munmap(token_data, data_len); close(data_fd);
        if (val_data) { munmap(val_data, val_len); }
        if (val_fd >= 0) close(val_fd);
    }
    return 0;
}
