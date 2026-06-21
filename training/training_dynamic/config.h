// config.h — Model-agnostic structs, derived sizes, ANE init
// Model-specific dims come from models/*.h, selected via -DMODEL_HEADER
#pragma once
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#import <Accelerate/Accelerate.h>
#include <math.h>
#include <unistd.h>
#include <dispatch/dispatch.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <arm_neon.h>

// Include selected model config
// MODEL_HEADER is set by Makefile via -include models/xxx.h
#ifndef MODEL_NAME
#error "No model selected. Build with: make MODEL=qwen3_06b (or stories110m)"
#endif

// Optimizer default for hand-written model headers that predate the shared
// config (emit_c.py emits this; --opt overrides it at runtime either way).
#ifndef OPTIMIZER_IS_MUON
#define OPTIMIZER_IS_MUON 0
#endif

// DeepSeek-V4 ablation knobs (PRD #2). emit_c.py emits these for generated
// headers; these fallbacks keep the hand-written model headers compiling. Every
// default is off / identity, so the trainer is the plain transformer until a
// knob is turned on. ROPE_ROTARY_DIMS defaults to HD (= full RoPE = identity).
#ifndef QK_NORM
#define QK_NORM 0
#endif
// RMSNorm epsilon, shared by QK-norm and the RMSNorm path. emit_c.py emits this
// for generated headers; fallback keeps hand-written model headers compiling.
#ifndef NORM_EPS
#define NORM_EPS 1e-05f
#endif
#ifndef ATTN_SINK
#define ATTN_SINK 0
#endif
#ifndef SWIGLU_CLAMP
#define SWIGLU_CLAMP 0
#endif
#ifndef ROPE_ROTARY_DIMS
#define ROPE_ROTARY_DIMS HD
#endif
#ifndef N_HC
#define N_HC 1
#endif
// ⚠️ BROKEN ON THIS HARDWARE — the four placement knobs below (WO_FUNCPARAM,
// W2T_FUNCPARAM, CONV_PROBE, CONV_DATAPATH) all route the weight through a
// multi-input ANE request (io.h make_request_2in), which is REJECTED at
// inference on M3 Max / Darwin 25.5 (ANEProgramProcessRequestDirect status=0x1d,
// "Program Inference error"). Enabling any of them now aborts loudly via
// ane_eval_check rather than silently training on zeros (the earlier "cos
// 1.00000" passes were a gate comparing a binary against itself). Kept default-0
// only as a starting point for a future multi-input fix or the single-input conv
// path (gen_conv_1in). Full post-mortem: results/ane_residency.md "CORRECTION".
// V2 placement knob (issue #12): pass woFwd's Wo as a function-parameter
// IOSurface instead of packing it into the activation surface's spatial dim,
// removing the in-kernel weight slice/reshape (upstream PR #22 lever). 0 = the
// original single-input spatial-packed path.
#ifndef WO_FUNCPARAM
#define WO_FUNCPARAM 0
#endif
// Same lever for the FFN-W2 backward matmul (W2 = DIM*HIDDEN, the largest simple
// weight) — the kernel whose unpack should show the clearest signal if any does.
#ifndef W2T_FUNCPARAM
#define W2T_FUNCPARAM 0
#endif
// Experiment: emit the matmul kernels as 1x1 convs (PRD #26, no transposes).
// Probed on ffnBwdW2t under (W2T_FUNCPARAM && CONV_PROBE).
#ifndef CONV_PROBE
#define CONV_PROBE 0
#endif
// (RETRACTED conv "rollout" — see the ⚠️ block above.) Was meant to extend the
// conv datapath (PRD #26) to wotBwd + qBwd, but delivers the weight via the
// func-param multi-input request, so it hits the same status=0x1d wall and emits
// zeros. Left wired (default-0) as scaffolding for the single-input gen_conv_1in
// retry, which is the only path proven to eval on this box.
#ifndef CONV_DATAPATH
#define CONV_DATAPATH 0
#endif
// PRD #26 retry on the SINGLE-input binding (the only conv path that evals on
// this box — CORRECT here, cos 1.00000 / R1 PASS, unlike the 2-in attempts).
// Emits wotBwd's backward matmul as a 1x1 conv that consumes the packed
// activation [1,IC,1,SEQ] directly and outputs [1,OC,1,SEQ] — dropping BOTH
// activation transposes gen_dyn_matmul pays. 0 = matmul; 1 = conv with one
// in-MIL weight transpose; 2 = conv with the weight pre-transposed in CPU
// staging (gen_conv_1in_mil_B, ZERO in-MIL transposes).
// MEASURED RESULT (stories110m, square IC=OC=768): ane_bwd 31.2 / 31.0 / 31.4 ms
// for 0/1/2 — a WASH within run-to-run noise. And =2 (no transposes at all) is
// no faster than matmul, so the transposes were never the bottleneck: ane_bwd is
// ANE matmul compute + fixed per-eval dispatch, not data layout. Conv only wins
// where the activation dominates the weight (tall-skinny: large SEQ or IC>>OC);
// no current kernel is that shape. Kept default-0 as a proven-correct datapath.
#ifndef CONV1IN
#define CONV1IN 0
#endif
// Kernel-fusion lever (results/ane_residency.md "Dispatch-overhead probe"): the
// ANE is DISPATCH-BOUND (~0.12 ms fixed overhead/eval, compute near-free to
// n=1024; fusion PoC saved 38% on 2 matmuls). Re-fuse the qBwd + kvBwd backward
// projections into ONE single-input kernel (qkvBwd) computing
// dx_attn = dq@Wq + dk@Wk + dv@Wv in one eval, removing 1 eval/layer plus the
// CPU dx_attn += dx_kv add. For GQA, the fused surface uses Q_DIM channels and
// the K/V slices address the first KV_DIM channels; only the real K/V rows are
// staged/written. Single-input only (the multi-input binding is dead here — see
// the CORRECTION block). Default on for MHA and GQA.
#ifndef FUSE_QKVBWD
#define FUSE_QKVBWD 1
#endif
// Attention-backward IO lever: bwd1 produces dV/probs/dp and bwd2 consumes
// probs/dp to produce dQ/dK. Keeping that score-sized handoff on IOSurfaces moves
// multiple MB/layer through CPU-visible memory. Mode 1 tries one all-in graph
// (dQ,dK,dV) but currently hits an ANE compiler "cycle path" internal error on
// this box. Mode 2 keeps two evals but recomputes attention in the dQ/dK kernel,
// so probs/dp never leave the ANE; it compiles but measured slower on qwen3_06b
// (345.8ms/step vs 343.0 split after the IO/capture cleanup). 0 = faster current
// split handoff.
#ifndef FUSE_SDPA_BWD
#define FUSE_SDPA_BWD 0
#endif
// VERTICAL-fusion lever (the dispatch-bound corollary): fold the SiLU backward
// — the 6.4 ms/step CPU bucket wedged BETWEEN the two FFN-backward evals
// (ffnBwdW2t -> [silu-bwd] -> ffnBwdW13t) — INTO the ffnBwdW13t kernel. dsilu,
// h1, h3 are all [HIDDEN,SEQ] (channel-HIDDEN), so no mixed-dim packing; the bwd
// ops (sigmoid/mul/add) are exactly the forward SiLU's, proven ANE-expressible.
// The fused kernel takes [dsilu|h1|h3|W1t|W3t] and emits concat(dx_ffn,dh1,dh3)
// so the async dW closure still gets dh1/dh3 (no recompute on the SERIAL dw_q).
// dsilu flows ffnBwdW2t->ffnBwdW13t via a strided ANE->ANE copy (no CPU bounce).
// Moves 6.4 ms of CPU compute onto the dispatch-bound ANE (compute ~free).
// Default-on when SWIGLU_CLAMP is off: measured after the optimizer/update
// cleanup at qwen3_06b 373.9ms/step vs 377.0 and stories110m 106.6 vs 109.9.
// 0 = CPU silu. 1 = kernel outputs concat(dx,dh1,dh3); the async dW reads dh1/dh3
//   back (round-trip tax: +1.8 io + 1.8 ane for the 6x-bigger output). Net −2.0.
// 2 = kernel outputs dx ONLY; the dW closure recomputes dh1/dh3 on the serial
//   dw_q. It still loses after persistent sigmoid scratch: qwen3_06b 422.6ms/step
//   vs 334.1 for mode 1, because the recompute stretches overlap too far.
// Works for MHA and GQA (pure FFN).
#ifndef FUSE_SILU_BWD
#define FUSE_SILU_BWD (!SWIGLU_CLAMP)
#endif
#if FUSE_SILU_BWD && SWIGLU_CLAMP
#error "FUSE_SILU_BWD implements the non-clamp SiLU backward only; build SWIGLU_CLAMP=0 or extend gen_ffn_bwd_w13t_fused_silu_dynamic with the clamp masks."
#endif
// Experimental W13 implementation: keep the same fused-SiLU math/output contract
// but express dh1@W1^T and dh3@W3^T as 1x1 convs. Measured rejected on qwen3_06b:
// 472.0ms/step train, W13 eval 88.9ms vs ~334ms/step and ~33ms W13 for matmul.
// 0 = matmul path.
#ifndef W13_CONV1IN
#define W13_CONV1IN 0
#endif
// Multi-Token Prediction depth (issue #6). 0 = off (plain next-token). emit_c.py
// emits this for generated headers; fallback keeps hand-written headers compiling.
#ifndef MTP_DEPTH
#define MTP_DEPTH 0
#endif
// MTP auxiliary-loss weight (issue #6). Mirrors lilbro/mlx_ref/params.py MTP_LAMBDA.
#ifndef MTP_LAMBDA
#define MTP_LAMBDA 0.3f
#endif
// Dims actually rotated by RoPE: min(HD, ROPE_ROTARY_DIMS). Identity when HD <=
// ROPE_ROTARY_DIMS (every current ladder rung). Mirrors Config.rope_rotary_eff.
#define ROPE_ROTARY_EFF (HD < ROPE_ROTARY_DIMS ? HD : ROPE_ROTARY_DIMS)

// Knobs whose forward softmax the ANE kernel can't express (sink in the
// denominator, QK-norm before the scores) share one CPU attention bypass
// (attn_cpu_forward/backward). Either knob on routes attention through CPU.
#define ATTN_CPU (ATTN_SINK || QK_NORM)

// Derived weight sizes per layer (GQA-aware)
#define WQ_SZ (Q_DIM*DIM)
#define WK_SZ (KV_DIM*DIM)
#define WV_SZ (KV_DIM*DIM)
#define WO_SZ (DIM*Q_DIM)
#define W1_SZ (HIDDEN*DIM)
#define W2_SZ (DIM*HIDDEN)
#define W3_SZ (HIDDEN*DIM)
#define LAYER_PARAMS (WQ_SZ + WK_SZ + WV_SZ + WO_SZ + W1_SZ + W2_SZ + W3_SZ + 2*DIM)

// Attention score channels for SDPA backward
#define SCORE_CH (HEADS*SEQ)

// Per-layer weights
typedef struct {
    float *Wq, *Wk, *Wv, *Wo;
    float *W1, *W2, *W3;
    float *rms_att, *rms_ffn;
} LayerWeights;

// Adam optimizer state
typedef struct { float *m, *v; size_t n; } AdamState;
typedef struct {
    AdamState Wq, Wk, Wv, Wo, W1, W2, W3, rms_att, rms_ffn;
} LayerAdam;

// Per-layer activations (saved for backward)
typedef struct {
    float *layer_in, *xnorm, *Q, *K, *V, *attn_out, *o_out;
    float *x2, *x2norm, *h1, *h3, *silu_out, *ffn_out;
} LayerActs;

// Per-layer gradients
typedef struct {
    float *Wq, *Wk, *Wv, *Wo, *W1, *W2, *W3, *rms_att, *rms_ffn;
} LayerGrads;

// ANE kernel handle. ioIn1 is the optional second input surface for the
// function-parameter path (multi-input MIL func, e.g. woFwd with Wo as a
// separate IOSurface param); NULL for single-input kernels.
typedef struct { void *model; IOSurfaceRef ioIn, ioIn1, ioOut; void *request; void *tmpDir; } Kern;

// Per-layer IOSurfaces for pre-staged weights. woFwd_w is the function-param
// weight surface (Wo as its own IOSurface) used only when WO_FUNCPARAM is set.
typedef struct {
    IOSurfaceRef sdpaFwd_in, woFwd_in, ffnFused_in;
    IOSurfaceRef ffnBwdW2t_in, ffnBwdW13t_in, wotBwd_in, qBwd_in, kvBwd_in;
    IOSurfaceRef woFwd_w, ffnBwdW2t_w, wotBwd_w, qBwd_w;
#if FUSE_QKVBWD
    IOSurfaceRef qkvBwd_in;   // fused [dq|dk|dv|Wq|Wk|Wv] packed surface
#endif
} PerLayerSurfaces;

// Per-layer ANE requests (bound to per-layer IOSurfaces)
typedef struct {
    void *sdpaFwd, *woFwd, *ffnFused;
    void *ffnBwdW2t, *ffnBwdW13t, *wotBwd, *qBwd, *kvBwd;
#if FUSE_QKVBWD
    void *qkvBwd;             // fused qBwd+kvBwd request (replaces both)
#endif
} PerLayerRequests;

// Checkpoint header
typedef struct {
    int magic, version, step, total_steps;
    int n_layers, vocab_size, dim, hidden_dim, n_heads, seq_len;
    float lr, loss;
    double cum_compile, cum_train, cum_wall;
    int cum_steps, cum_batches, adam_t;
    int kv_heads, head_dim, q_dim;  // GQA fields
    // Note: was int pad[3] in v3, now stores GQA info in v4+
} CkptHdr;

// Globals
static Class g_D, g_I, g_AR, g_AIO;
static mach_timebase_info_data_t g_tb;
static int g_compile_count = 0;

static void ane_init(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_I  = NSClassFromString(@"_ANEInMemoryModel");
    g_AR = NSClassFromString(@"_ANERequest");
    g_AIO= NSClassFromString(@"_ANEIOSurfaceObject");
}
static double tb_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

// Alloc helpers
static AdamState adam_alloc(size_t n) { AdamState s; s.m=(float*)calloc(n,4); s.v=(float*)calloc(n,4); s.n=n; return s; }
static void adam_free(AdamState *s) { free(s->m); free(s->v); }

static LayerWeights layer_weights_alloc(void) {
    LayerWeights w;
    w.Wq=(float*)malloc(WQ_SZ*4); w.Wk=(float*)malloc(WK_SZ*4);
    w.Wv=(float*)malloc(WV_SZ*4); w.Wo=(float*)malloc(WO_SZ*4);
    w.W1=(float*)malloc(W1_SZ*4); w.W2=(float*)malloc(W2_SZ*4); w.W3=(float*)malloc(W3_SZ*4);
    w.rms_att=(float*)malloc(DIM*4); w.rms_ffn=(float*)malloc(DIM*4);
    return w;
}
static void layer_weights_free(LayerWeights *w) {
    free(w->Wq);free(w->Wk);free(w->Wv);free(w->Wo);
    free(w->W1);free(w->W2);free(w->W3);free(w->rms_att);free(w->rms_ffn);
}
static LayerAdam layer_adam_alloc(void) {
    LayerAdam a;
    a.Wq=adam_alloc(WQ_SZ); a.Wk=adam_alloc(WK_SZ); a.Wv=adam_alloc(WV_SZ); a.Wo=adam_alloc(WO_SZ);
    a.W1=adam_alloc(W1_SZ); a.W2=adam_alloc(W2_SZ); a.W3=adam_alloc(W3_SZ);
    a.rms_att=adam_alloc(DIM); a.rms_ffn=adam_alloc(DIM);
    return a;
}
static void layer_adam_free(LayerAdam *a) {
    adam_free(&a->Wq);adam_free(&a->Wk);adam_free(&a->Wv);adam_free(&a->Wo);
    adam_free(&a->W1);adam_free(&a->W2);adam_free(&a->W3);
    adam_free(&a->rms_att);adam_free(&a->rms_ffn);
}
static LayerActs layer_acts_alloc(void) {
    LayerActs a;
    a.layer_in=(float*)malloc(SEQ*DIM*4);
    a.xnorm=(float*)malloc(SEQ*DIM*4);
    a.Q=(float*)malloc(SEQ*Q_DIM*4); a.K=(float*)malloc(SEQ*KV_DIM*4); a.V=(float*)malloc(SEQ*KV_DIM*4);
    a.attn_out=(float*)malloc(SEQ*Q_DIM*4); a.o_out=(float*)malloc(SEQ*DIM*4);
    a.x2=(float*)malloc(SEQ*DIM*4); a.x2norm=(float*)malloc(SEQ*DIM*4);
    a.h1=(float*)malloc(SEQ*HIDDEN*4); a.h3=(float*)malloc(SEQ*HIDDEN*4);
    a.silu_out=(float*)malloc(SEQ*HIDDEN*4); a.ffn_out=(float*)malloc(SEQ*DIM*4);
    return a;
}
static void layer_acts_free(LayerActs *a) {
    free(a->layer_in);free(a->xnorm);
    free(a->Q);free(a->K);free(a->V);
    free(a->attn_out);free(a->o_out);free(a->x2);free(a->x2norm);
    free(a->h1);free(a->h3);free(a->silu_out);free(a->ffn_out);
}
static LayerGrads layer_grads_alloc(void) {
    LayerGrads g;
    g.Wq=(float*)calloc(WQ_SZ,4); g.Wk=(float*)calloc(WK_SZ,4);
    g.Wv=(float*)calloc(WV_SZ,4); g.Wo=(float*)calloc(WO_SZ,4);
    g.W1=(float*)calloc(W1_SZ,4); g.W2=(float*)calloc(W2_SZ,4); g.W3=(float*)calloc(W3_SZ,4);
    g.rms_att=(float*)calloc(DIM,4); g.rms_ffn=(float*)calloc(DIM,4);
    return g;
}
static void layer_grads_zero(LayerGrads *g) {
    memset(g->Wq,0,WQ_SZ*4);memset(g->Wk,0,WK_SZ*4);
    memset(g->Wv,0,WV_SZ*4);memset(g->Wo,0,WO_SZ*4);
    memset(g->W1,0,W1_SZ*4);memset(g->W2,0,W2_SZ*4);memset(g->W3,0,W3_SZ*4);
    memset(g->rms_att,0,DIM*4);memset(g->rms_ffn,0,DIM*4);
}
static void layer_grads_free(LayerGrads *g) {
    free(g->Wq);free(g->Wk);free(g->Wv);free(g->Wo);
    free(g->W1);free(g->W2);free(g->W3);free(g->rms_att);free(g->rms_ffn);
}

typedef struct {
    float *dffn, *dh1, *dh3;
#if FUSE_SILU_BWD == 2
    float *dsilu, *sig;
#endif
    float *dwo;
    float *dq, *dk, *dv;
} DwCapture;

static DwCapture dw_capture_alloc(void) {
    DwCapture c;
    c.dffn=(float*)malloc(SEQ*DIM*4);
    c.dh1=(float*)malloc(SEQ*HIDDEN*4);
    c.dh3=(float*)malloc(SEQ*HIDDEN*4);
#if FUSE_SILU_BWD == 2
    c.dsilu=(float*)malloc(SEQ*HIDDEN*4);
    c.sig=(float*)malloc(SEQ*HIDDEN*4);
#endif
    c.dwo=(float*)malloc(SEQ*DIM*4);
    c.dq=(float*)malloc(SEQ*Q_DIM*4);
    c.dk=(float*)malloc(SEQ*KV_DIM*4);
    c.dv=(float*)malloc(SEQ*KV_DIM*4);
    return c;
}
static void dw_capture_free(DwCapture *c) {
    free(c->dffn);free(c->dh1);free(c->dh3);
#if FUSE_SILU_BWD == 2
    free(c->dsilu);free(c->sig);
#endif
    free(c->dwo);
    free(c->dq);free(c->dk);free(c->dv);
}
