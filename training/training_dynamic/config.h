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
} PerLayerSurfaces;

// Per-layer ANE requests (bound to per-layer IOSurfaces)
typedef struct {
    void *sdpaFwd, *woFwd, *ffnFused;
    void *ffnBwdW2t, *ffnBwdW13t, *wotBwd, *qBwd, *kvBwd;
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
