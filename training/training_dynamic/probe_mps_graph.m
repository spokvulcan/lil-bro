// probe_mps_graph.m — Phase 2 tracer bullet: the FULL chess trunk as ONE MPSGraph.
//
// Phase 1 (probe_mps.m) showed the matmul-seam is a dead end: MPS dispatch is
// slower than ANE dispatch (1.16ms vs 0.71ms per matmul), and the io savings
// (fp16 convert + IOSurfaceLock) exactly offset it. Root cause: GPU pipeline
// bubbles — the GPU sits idle between matmuls while the CPU does attention/rms/
// silu, so each matmul pays scheduling latency.
//
// This probe tests the REAL hypothesis: if ALL trunk ops (matmuls + attention +
// rms + silu + residual) run on the GPU as ONE MPSGraph in ONE command buffer,
// there are NO pipeline bubbles — the GPU processes the entire trunk as a
// continuous pipeline. That's the "could approach 100x" hypothesis from the
// speedup doc.
//
// Measures:
//   (1) CORRECTNESS: MPSGraph trunk vs CPU reference (cblas + scalar)
//   (2) THROUGHPUT: ms/forward at B=64/96/160 vs ANE+CPU (16.9ms/fwd at B=64)
//   (3) Whether the GPU stays busy (pipeline efficiency)
//
// Build: see Makefile target probe_mps_graph.
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <Accelerate/Accelerate.h>
#include <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DIM     256
#define HIDDEN  512
#define HEADS    8
#define HD      32
#define Q_DIM  (HEADS*HD)
#define KV_DIM (HEADS*HD)
#define SEQ     96
#define NLAYERS 2

static double tb_ms(uint64_t t) {
    static double s = -1; if (s < 0) { mach_timebase_info_data_t b; mach_timebase_info(&b);
        s = (double)b.numer / b.denom / 1e6; }
    return (double)t * s;
}

static void fill_randn(float *p, size_t n, unsigned *seed) {
    for (size_t i = 0; i < n; i++) {
        float s = 0;
        for (int k = 0; k < 12; k++) { *seed = *seed * 1103515245u + 12345u; s += (float)(*seed >> 8) / 16777216.0f; }
        p[i] = (s - 6.0f) * 0.1f;
    }
}

static float *pa_alloc(size_t n) {
    void *p = NULL; size_t bytes = ((n*4 + 4095) & ~4095);
    if (posix_memalign(&p, 4096, bytes)) { fprintf(stderr, "OOM\n"); abort(); }
    return (float*)p;
}

// ============================================================================
// CPU reference trunk (for correctness verification).
// ============================================================================
static void cpu_rmsnorm(float *out, const float *x, const float *w, int d, int S) {
    for (int s = 0; s < S; s++) {
        float ss = 0;
        for (int i = 0; i < d; i++) ss += x[i*S+s]*x[i*S+s];
        float r = 1.0f/sqrtf(ss/d + 1e-5f);
        for (int i = 0; i < d; i++) out[i*S+s] = x[i*S+s]*r*w[i];
    }
}
static void cpu_matmul(const float *W, const float *x, float *y, int ic, int oc, int seq) {
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, oc, seq, ic, 1.0f, W, oc, x, seq, 0.0f, y, seq);
}
static void cpu_attention(float *attn_out, const float *Q, const float *K, const float *V, int B) {
    float scale = 1.0f/sqrtf((float)HD);
    float *scores = (float*)malloc(SEQ*SEQ*4);
    int S = B*SEQ;
    for (int b = 0; b < B; b++) {
        int off = b*SEQ;
        for (int h = 0; h < HEADS; h++) {
            for (int i = 0; i < SEQ; i++) {
                for (int j = 0; j <= i; j++) {
                    float s = 0;
                    for (int d = 0; d < HD; d++)
                        s += Q[(h*HD+d)*S + off+i] * K[(h*HD+d)*S + off+j];
                    scores[i*SEQ+j] = s * scale;
                }
                for (int j = i+1; j < SEQ; j++) scores[i*SEQ+j] = -1e9f;
            }
            for (int i = 0; i < SEQ; i++) {
                float mx = scores[i*SEQ];
                for (int j = 1; j < SEQ; j++) if (scores[i*SEQ+j]>mx) mx = scores[i*SEQ+j];
                float sm = 0;
                for (int j = 0; j < SEQ; j++) { scores[i*SEQ+j] = expf(scores[i*SEQ+j]-mx); sm += scores[i*SEQ+j]; }
                for (int j = 0; j < SEQ; j++) scores[i*SEQ+j] /= sm;
            }
            for (int d = 0; d < HD; d++)
                for (int i = 0; i < SEQ; i++) {
                    float s = 0;
                    for (int j = 0; j < SEQ; j++) s += scores[i*SEQ+j] * V[(h*HD+d)*S + off+j];
                    attn_out[(h*HD+d)*S + off+i] = s;
                }
        }
    }
    free(scores);
}
static float *cpu_trunk_forward(const float *x_in, int B,
    const float *Wqkv, const float *Wo, const float *W13, const float *W2,
    const float *rms_att, const float *rms_ffn, const float *rms_final,
    float res_alpha, int S) {
    float *x = pa_alloc((size_t)DIM*S); memcpy(x, x_in, (size_t)DIM*S*4);
    float *xnorm = pa_alloc((size_t)DIM*S), *x2 = pa_alloc((size_t)DIM*S), *x2norm = pa_alloc((size_t)DIM*S);
    float *qkv = pa_alloc((size_t)(Q_DIM+2*KV_DIM)*S), *attn = pa_alloc((size_t)Q_DIM*S);
    float *o = pa_alloc((size_t)DIM*S), *h13 = pa_alloc((size_t)2*HIDDEN*S);
    float *gate = pa_alloc((size_t)HIDDEN*S), *ffn = pa_alloc((size_t)DIM*S);
    float *x_final = pa_alloc((size_t)DIM*S);
    const float *Q = qkv, *K = qkv + (size_t)Q_DIM*S, *V = qkv + (size_t)(Q_DIM+KV_DIM)*S;
    const float *h1 = h13, *h3 = h13 + (size_t)HIDDEN*S;
    for (int L = 0; L < NLAYERS; L++) {
        cpu_rmsnorm(xnorm, x, rms_att + (size_t)L*DIM, DIM, S);
        cpu_matmul(Wqkv + (size_t)L*DIM*(Q_DIM+2*KV_DIM), xnorm, qkv, DIM, Q_DIM+2*KV_DIM, S);
        cpu_attention(attn, Q, K, V, B);
        cpu_matmul(Wo + (size_t)L*Q_DIM*DIM, attn, o, Q_DIM, DIM, S);
        for (int i = 0; i < DIM*S; i++) x2[i] = x[i] + res_alpha*o[i];
        cpu_rmsnorm(x2norm, x2, rms_ffn + (size_t)L*DIM, DIM, S);
        cpu_matmul(W13 + (size_t)L*DIM*2*HIDDEN, x2norm, h13, DIM, 2*HIDDEN, S);
        for (int i = 0; i < HIDDEN*S; i++) { float sig = 1.0f/(1.0f+expf(-h1[i])); gate[i] = (h1[i]*sig)*h3[i]; }
        cpu_matmul(W2 + (size_t)L*HIDDEN*DIM, gate, ffn, HIDDEN, DIM, S);
        for (int i = 0; i < DIM*S; i++) x[i] = x2[i] + res_alpha*ffn[i];
    }
    cpu_rmsnorm(x_final, x, rms_final, DIM, S);
    free(x); free(xnorm); free(x2); free(x2norm); free(qkv); free(attn); free(o); free(h13); free(gate); free(ffn);
    return x_final;
}

// ============================================================================
// MPSGraph trunk builder.
// ============================================================================
typedef struct {
    MPSGraph *graph;
    MPSGraphTensor *x_in, *rms_final_ph;
    MPSGraphTensor *Wqkv_ph[NLAYERS], *Wo_ph[NLAYERS], *W13_ph[NLAYERS], *W2_ph[NLAYERS];
    MPSGraphTensor *rms_att_ph[NLAYERS], *rms_ffn_ph[NLAYERS];
    MPSGraphTensor *x_out;
    // Debug: intermediate targets for correctness isolation.
    MPSGraphTensor *dbg_xnorm0, *dbg_qkv0, *dbg_attn0, *dbg_o0, *dbg_x_after_L0, *dbg_x_before_final;
} MpsGraphTrunk;

// matmul: y[oc,seq] = W[ic,oc]^T @ x[ic,seq]. W is [ic,oc], x is [ic,seq].
// MPSGraph: matmul(W^T, x) = matmul(transpose(W,0,1), x) → [oc, seq].
static MPSGraphTensor *g_matmul(MPSGraph *g, MPSGraphTensor *W, MPSGraphTensor *x, NSString *nm) {
    MPSGraphTensor *Wt = [g transposeTensor:W dimension:0 withDimension:1 name:[nm stringByAppendingString:@"_Wt"]];
    return [g matrixMultiplicationWithPrimaryTensor:Wt secondaryTensor:x name:nm];
}

// RMSNorm: out = x * rsqrt(mean(x^2, axis=0) + eps) * weight. x: [DIM, S], w: [DIM].
static MPSGraphTensor *g_rmsnorm(MPSGraph *g, MPSGraphTensor *x, MPSGraphTensor *w, int dim, int S, NSString *nm) {
    MPSGraphTensor *x2 = [g multiplicationWithPrimaryTensor:x secondaryTensor:x name:[nm stringByAppendingString:@"_x2"]];
    MPSGraphTensor *sum = [g reductionSumWithTensor:x2 axis:0 name:[nm stringByAppendingString:@"_sum"]];
    // Reshape [S] → [1, S] so it broadcasts correctly against [DIM, S].
    MPSGraphTensor *sum_2d = [g reshapeTensor:sum withShape:@[@1, @(S)] name:[nm stringByAppendingString:@"_sum2d"]];
    MPSGraphTensor *dim_c = [g constantWithScalar:(double)dim dataType:MPSDataTypeFloat32];
    MPSGraphTensor *mean = [g divisionWithPrimaryTensor:sum_2d secondaryTensor:dim_c name:[nm stringByAppendingString:@"_mean"]];
    MPSGraphTensor *eps_c = [g constantWithScalar:1e-5 dataType:MPSDataTypeFloat32];
    MPSGraphTensor *me = [g additionWithPrimaryTensor:mean secondaryTensor:eps_c name:[nm stringByAppendingString:@"_me"]];
    MPSGraphTensor *sq = [g squareRootWithTensor:me name:[nm stringByAppendingString:@"_sq"]];
    MPSGraphTensor *one = [g constantWithScalar:1.0 dataType:MPSDataTypeFloat32];
    MPSGraphTensor *rrms = [g divisionWithPrimaryTensor:one secondaryTensor:sq name:[nm stringByAppendingString:@"_rrms"]];
    MPSGraphTensor *xn = [g multiplicationWithPrimaryTensor:x secondaryTensor:rrms name:[nm stringByAppendingString:@"_xn"]];
    MPSGraphTensor *wc = [g reshapeTensor:w withShape:@[@(dim), @1] name:[nm stringByAppendingString:@"_wc"]];
    return [g multiplicationWithPrimaryTensor:xn secondaryTensor:wc name:nm];
}

// Attention: Q/K/V [Q_DIM, S], MHA. Returns attn_out [Q_DIM, S].
static MPSGraphTensor *g_attention(MPSGraph *g, MPSGraphTensor *Q, MPSGraphTensor *K,
                                   MPSGraphTensor *V, MPSGraphTensor *mask, int B, int S, NSString *nm) {
    // [Q_DIM, S] → [HEADS, HD, B, SEQ] → [B, HEADS, SEQ, HD]
    MPSGraphTensor *Q4 = [g reshapeTensor:Q withShape:@[@(HEADS), @(HD), @(B), @(SEQ)] name:[nm stringByAppendingString:@"_Q4"]];
    MPSGraphTensor *Qb = [g transposeTensor:Q4 permutation:@[@2, @0, @3, @1] name:[nm stringByAppendingString:@"_Qb"]];
    MPSGraphTensor *K4 = [g reshapeTensor:K withShape:@[@(HEADS), @(HD), @(B), @(SEQ)] name:[nm stringByAppendingString:@"_K4"]];
    MPSGraphTensor *Kb = [g transposeTensor:K4 permutation:@[@2, @0, @3, @1] name:[nm stringByAppendingString:@"_Kb"]];
    MPSGraphTensor *V4 = [g reshapeTensor:V withShape:@[@(HEADS), @(HD), @(B), @(SEQ)] name:[nm stringByAppendingString:@"_V4"]];
    MPSGraphTensor *Vb = [g transposeTensor:V4 permutation:@[@2, @0, @3, @1] name:[nm stringByAppendingString:@"_Vb"]];
    // scores = Q @ K^T * scale → [B, HEADS, SEQ, SEQ]
    MPSGraphTensor *Kt = [g transposeTensor:Kb dimension:2 withDimension:3 name:[nm stringByAppendingString:@"_Kt"]];
    MPSGraphTensor *sc = [g matrixMultiplicationWithPrimaryTensor:Qb secondaryTensor:Kt name:[nm stringByAppendingString:@"_sc"]];
    float scale = 1.0f/sqrtf((float)HD);
    MPSGraphTensor *sc_c = [g constantWithScalar:(double)scale dataType:MPSDataTypeFloat32];
    MPSGraphTensor *scs = [g multiplicationWithPrimaryTensor:sc secondaryTensor:sc_c name:[nm stringByAppendingString:@"_scs"]];
    // mask [SEQ, SEQ] → [1, 1, SEQ, SEQ], broadcast add
    MPSGraphTensor *mb = [g reshapeTensor:mask withShape:@[@1, @1, @(SEQ), @(SEQ)] name:[nm stringByAppendingString:@"_mb"]];
    MPSGraphTensor *msk = [g additionWithPrimaryTensor:scs secondaryTensor:mb name:[nm stringByAppendingString:@"_msk"]];
    // softmax axis -1 (last = 3)
    MPSGraphTensor *pr = [g softMaxWithTensor:msk axis:3 name:[nm stringByAppendingString:@"_sm"]];
    // attn = probs @ V → [B, HEADS, SEQ, HD]
    MPSGraphTensor *at = [g matrixMultiplicationWithPrimaryTensor:pr secondaryTensor:Vb name:[nm stringByAppendingString:@"_at"]];
    // [B, HEADS, SEQ, HD] → [HEADS, HD, B, SEQ] → [Q_DIM, S]
    MPSGraphTensor *at_t = [g transposeTensor:at permutation:@[@1, @3, @0, @2] name:[nm stringByAppendingString:@"_att"]];
    return [g reshapeTensor:at_t withShape:@[@(Q_DIM), @(S)] name:nm];
}

static void build_trunk(MpsGraphTrunk *t, int B, int S) {
    t->graph = [[MPSGraph alloc] init];
    MPSGraph *g = t->graph;
    t->x_in = [g placeholderWithShape:@[@(DIM), @(S)] dataType:MPSDataTypeFloat32 name:@"x_in"];
    t->rms_final_ph = [g placeholderWithShape:@[@(DIM)] dataType:MPSDataTypeFloat32 name:@"rms_final"];
    // Causal mask [SEQ, SEQ]: 0 below diag, -1e9 above.
    float *md = (float*)malloc(SEQ*SEQ*4);
    for (int i = 0; i < SEQ; i++) for (int j = 0; j < SEQ; j++) md[i*SEQ+j] = (j <= i) ? 0.0f : -1e9f;
    NSData *mns = [NSData dataWithBytesNoCopy:md length:SEQ*SEQ*4 freeWhenDone:YES];
    MPSGraphTensor *mask = [g constantWithData:mns shape:@[@(SEQ), @(SEQ)] dataType:MPSDataTypeFloat32];
    float alpha = 1.0f/sqrtf(2.0f*NLAYERS);
    MPSGraphTensor *alpha_c = [g constantWithScalar:(double)alpha dataType:MPSDataTypeFloat32];
    for (int L = 0; L < NLAYERS; L++) {
        NSString *pfx = [NSString stringWithFormat:@"L%d_", L];
        t->Wqkv_ph[L] = [g placeholderWithShape:@[@(DIM), @(Q_DIM+2*KV_DIM)] dataType:MPSDataTypeFloat32 name:[pfx stringByAppendingString:@"Wqkv"]];
        t->Wo_ph[L]   = [g placeholderWithShape:@[@(Q_DIM), @(DIM)] dataType:MPSDataTypeFloat32 name:[pfx stringByAppendingString:@"Wo"]];
        t->W13_ph[L]  = [g placeholderWithShape:@[@(DIM), @(2*HIDDEN)] dataType:MPSDataTypeFloat32 name:[pfx stringByAppendingString:@"W13"]];
        t->W2_ph[L]   = [g placeholderWithShape:@[@(HIDDEN), @(DIM)] dataType:MPSDataTypeFloat32 name:[pfx stringByAppendingString:@"W2"]];
        t->rms_att_ph[L] = [g placeholderWithShape:@[@(DIM)] dataType:MPSDataTypeFloat32 name:[pfx stringByAppendingString:@"rms_att"]];
        t->rms_ffn_ph[L] = [g placeholderWithShape:@[@(DIM)] dataType:MPSDataTypeFloat32 name:[pfx stringByAppendingString:@"rms_ffn"]];
    }
    MPSGraphTensor *x = t->x_in;
    for (int L = 0; L < NLAYERS; L++) {
        NSString *pfx = [NSString stringWithFormat:@"L%d_", L];
        // xnorm = rmsnorm(x, rms_att)
        MPSGraphTensor *xn = g_rmsnorm(g, x, t->rms_att_ph[L], DIM, S, [pfx stringByAppendingString:@"rms_att"]);
        if (L == 0) t->dbg_xnorm0 = xn;
        // QKV = Wqkv^T @ xnorm → [Q_DIM+2*KV_DIM, S], then slice into Q, K, V
        MPSGraphTensor *qkv = g_matmul(g, t->Wqkv_ph[L], xn, [pfx stringByAppendingString:@"qkv"]);
        if (L == 0) t->dbg_qkv0 = qkv;
        MPSGraphTensor *Q = [g sliceTensor:qkv dimension:0 start:0 length:Q_DIM name:[pfx stringByAppendingString:@"Q"]];
        MPSGraphTensor *K = [g sliceTensor:qkv dimension:0 start:Q_DIM length:KV_DIM name:[pfx stringByAppendingString:@"K"]];
        MPSGraphTensor *V = [g sliceTensor:qkv dimension:0 start:Q_DIM+KV_DIM length:KV_DIM name:[pfx stringByAppendingString:@"V"]];
        // attn = attention(Q, K, V) → [Q_DIM, S]
        MPSGraphTensor *at = g_attention(g, Q, K, V, mask, B, S, [pfx stringByAppendingString:@"attn"]);
        if (L == 0) t->dbg_attn0 = at;
        // o = Wo^T @ attn → [DIM, S]
        MPSGraphTensor *o = g_matmul(g, t->Wo_ph[L], at, [pfx stringByAppendingString:@"wo"]);
        if (L == 0) t->dbg_o0 = o;
        // x2 = x + alpha * o
        MPSGraphTensor *ao = [g multiplicationWithPrimaryTensor:o secondaryTensor:alpha_c name:[pfx stringByAppendingString:@"ao"]];
        MPSGraphTensor *x2 = [g additionWithPrimaryTensor:x secondaryTensor:ao name:[pfx stringByAppendingString:@"x2"]];
        // x2norm = rmsnorm(x2, rms_ffn)
        MPSGraphTensor *x2n = g_rmsnorm(g, x2, t->rms_ffn_ph[L], DIM, S, [pfx stringByAppendingString:@"rms_ffn"]);
        // h13 = W13^T @ x2norm → [2*HIDDEN, S], slice into h1, h3
        MPSGraphTensor *h13 = g_matmul(g, t->W13_ph[L], x2n, [pfx stringByAppendingString:@"h13"]);
        MPSGraphTensor *h1 = [g sliceTensor:h13 dimension:0 start:0 length:HIDDEN name:[pfx stringByAppendingString:@"h1"]];
        MPSGraphTensor *h3 = [g sliceTensor:h13 dimension:0 start:HIDDEN length:HIDDEN name:[pfx stringByAppendingString:@"h3"]];
        // gate = silu(h1) * h3 = (h1 * sigmoid(h1)) * h3
        MPSGraphTensor *sig = [g sigmoidWithTensor:h1 name:[pfx stringByAppendingString:@"sig"]];
        MPSGraphTensor *silu = [g multiplicationWithPrimaryTensor:h1 secondaryTensor:sig name:[pfx stringByAppendingString:@"silu"]];
        MPSGraphTensor *gate = [g multiplicationWithPrimaryTensor:silu secondaryTensor:h3 name:[pfx stringByAppendingString:@"gate"]];
        // ffn = W2^T @ gate → [DIM, S]
        MPSGraphTensor *ffn = g_matmul(g, t->W2_ph[L], gate, [pfx stringByAppendingString:@"w2"]);
        // x = x2 + alpha * ffn
        MPSGraphTensor *af = [g multiplicationWithPrimaryTensor:ffn secondaryTensor:alpha_c name:[pfx stringByAppendingString:@"af"]];
        x = [g additionWithPrimaryTensor:x2 secondaryTensor:af name:[pfx stringByAppendingString:@"x"]];
        if (L == 0) t->dbg_x_after_L0 = x;
    }
    // x_final = rmsnorm(x, rms_final)
    t->dbg_x_before_final = x;
    t->x_out = g_rmsnorm(g, x, t->rms_final_ph, DIM, S, @"rms_final");
}

// Execute the graph with zero-copy MTLBuffers.
static float *run_graph(MpsGraphTrunk *t, id<MTLDevice> dev, id<MTLCommandQueue> q,
                        const float *x_in, int B, int S,
                        const float *Wqkv, const float *Wo, const float *W13, const float *W2,
                        const float *rms_att, const float *rms_ffn, const float *rms_final) {
    int S_act = S;
    // Create zero-copy MTLBuffers for all inputs.
    id<MTLBuffer> buf_x = [dev newBufferWithBytesNoCopy:(void*)x_in length:((DIM*S_act*4+4095)&~4095) options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> buf_rf = [dev newBufferWithBytesNoCopy:(void*)rms_final length:((DIM*4+4095)&~4095) options:MTLResourceStorageModeShared deallocator:nil];
    // Build feeds dictionary.
    NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
    feeds[t->x_in] = [[MPSGraphTensorData alloc] initWithMTLBuffer:buf_x shape:@[@(DIM), @(S_act)] dataType:MPSDataTypeFloat32];
    feeds[t->rms_final_ph] = [[MPSGraphTensorData alloc] initWithMTLBuffer:buf_rf shape:@[@(DIM)] dataType:MPSDataTypeFloat32];
    id<MTLBuffer> buf_w[NLAYERS*6]; int bi = 0;
    for (int L = 0; L < NLAYERS; L++) {
        #define FEED(ph, ptr, sz) do { \
            buf_w[bi] = [dev newBufferWithBytesNoCopy:(void*)(ptr) length:(((sz)*4+4095)&~4095) options:MTLResourceStorageModeShared deallocator:nil]; \
            feeds[t->ph[L]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:buf_w[bi++] shape:@[@((size_t)(sz))] dataType:MPSDataTypeFloat32]; \
        } while(0)
        // For 2D tensors, shape is [rows, cols] = [ic, oc] for weights.
        #define FEED2D(ph, ptr, r, c) do { \
            buf_w[bi] = [dev newBufferWithBytesNoCopy:(void*)(ptr) length:(((size_t)(r)*(c)*4+4095)&~4095) options:MTLResourceStorageModeShared deallocator:nil]; \
            feeds[t->ph[L]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:buf_w[bi++] shape:@[@(r), @(c)] dataType:MPSDataTypeFloat32]; \
        } while(0)
        FEED2D(Wqkv_ph, Wqkv + (size_t)L*DIM*(Q_DIM+2*KV_DIM), DIM, Q_DIM+2*KV_DIM);
        FEED2D(Wo_ph,   Wo + (size_t)L*Q_DIM*DIM, Q_DIM, DIM);
        FEED2D(W13_ph,  W13 + (size_t)L*DIM*2*HIDDEN, DIM, 2*HIDDEN);
        FEED2D(W2_ph,   W2 + (size_t)L*HIDDEN*DIM, HIDDEN, DIM);
        FEED(rms_att_ph, rms_att + (size_t)L*DIM, DIM);
        FEED(rms_ffn_ph, rms_ffn + (size_t)L*DIM, DIM);
    }
    // Output buffer (zero-copy into our allocation).
    float *x_out = pa_alloc((size_t)DIM*S_act);
    id<MTLBuffer> buf_out = [dev newBufferWithBytesNoCopy:x_out length:((DIM*S_act*4+4095)&~4095) options:MTLResourceStorageModeShared deallocator:nil];
    MPSGraphTensorData *out_td = [[MPSGraphTensorData alloc] initWithMTLBuffer:buf_out shape:@[@(DIM), @(S_act)] dataType:MPSDataTypeFloat32];
    // Execute using resultsDictionary variant — writes directly into our buffer.
    id<MTLCommandBuffer> cb = [q commandBuffer];
    MPSCommandBuffer *mps_cb = [MPSCommandBuffer commandBufferWithCommandBuffer:cb];
    NSMutableDictionary *results = [NSMutableDictionary dictionary];
    results[t->x_out] = out_td;
    [t->graph encodeToCommandBuffer:mps_cb feeds:feeds targetOperations:nil resultsDictionary:results executionDescriptor:nil];
    [mps_cb commit]; [mps_cb waitUntilCompleted];
    // Debug: print first few values to see if output is zeros or wrong.
    if (getenv("MPS_DEBUG")) {
        fprintf(stderr, "  [debug] gpu[0..4]=%.4f %.4f %.4f %.4f %.4f  (sum=%.2f)\n",
                x_out[0], x_out[1], x_out[2], x_out[3], x_out[4],
                x_out[0]+x_out[1]+x_out[2]+x_out[3]+x_out[4]);
    }
    return x_out;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "no Metal device\n"); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        printf("# probe_mps_graph — full 2-layer trunk as ONE MPSGraph\n");
        printf("# device: %s\n", [[dev name] UTF8String]);
        printf("# dims: DIM=%d HIDDEN=%d HEADS=%d HD=%d SEQ=%d NLAYERS=%d\n\n",
               DIM, HIDDEN, HEADS, HD, SEQ, NLAYERS);

        unsigned seed = 42;
        float alpha = 1.0f/sqrtf(2.0f*NLAYERS);
        // Allocate weights + norms (page-aligned for zero-copy).
        float *Wqkv = pa_alloc((size_t)NLAYERS*DIM*(Q_DIM+2*KV_DIM));
        float *Wo   = pa_alloc((size_t)NLAYERS*Q_DIM*DIM);
        float *W13  = pa_alloc((size_t)NLAYERS*DIM*2*HIDDEN);
        float *W2   = pa_alloc((size_t)NLAYERS*HIDDEN*DIM);
        float *rms_att = pa_alloc((size_t)NLAYERS*DIM);
        float *rms_ffn = pa_alloc((size_t)NLAYERS*DIM);
        float *rms_final = pa_alloc((size_t)DIM);
        fill_randn(Wqkv, NLAYERS*DIM*(Q_DIM+2*KV_DIM), &seed);
        fill_randn(Wo, NLAYERS*Q_DIM*DIM, &seed);
        fill_randn(W13, NLAYERS*DIM*2*HIDDEN, &seed);
        fill_randn(W2, NLAYERS*HIDDEN*DIM, &seed);
        fill_randn(rms_att, NLAYERS*DIM, &seed);
        fill_randn(rms_ffn, NLAYERS*DIM, &seed);
        fill_randn(rms_final, DIM, &seed);

        int Bs[] = {64, 96, 160};
        int nB = (int)(sizeof(Bs)/sizeof(Bs[0]));
        for (int bi = 0; bi < nB; bi++) {
            int B = Bs[bi], S = B*SEQ;
            printf("## B=%d S=%d\n", B, S);
            // Allocate input.
            float *x_in = pa_alloc((size_t)DIM*S);
            fill_randn(x_in, (size_t)DIM*S, &seed);

            // (1) Correctness: CPU reference vs MPSGraph.
            float *ref = cpu_trunk_forward(x_in, B, Wqkv, Wo, W13, W2, rms_att, rms_ffn, rms_final, alpha, S);
            // Build graph for this (B, S) — shapes are baked into the graph.
            MpsGraphTrunk t;
            build_trunk(&t, B, S);
            float *gpu = run_graph(&t, dev, q, x_in, B, S, Wqkv, Wo, W13, W2, rms_att, rms_ffn, rms_final);
            double maxerr = 0, sumerr = 0;
            for (int i = 0; i < DIM*S; i++) {
                double e = fabs((double)gpu[i] - ref[i]);
                if (e > maxerr) maxerr = e;
                sumerr += e;
            }
            double meanerr = sumerr / (DIM*S);
            printf("  correctness: max_err=%.4e  mean_err=%.4e  %s\n", maxerr, meanerr,
                   maxerr < 1e-3 ? "PASS" : "FAIL");
            if (getenv("MPS_DEBUG")) {
                fprintf(stderr, "  [debug] gpu[0..4]=%.4f %.4f %.4f %.4f %.4f\n", gpu[0], gpu[1], gpu[2], gpu[3], gpu[4]);
                fprintf(stderr, "  [debug] ref[0..4]=%.4f %.4f %.4f %.4f %.4f\n", ref[0], ref[1], ref[2], ref[3], ref[4]);
                fprintf(stderr, "  [debug] gpu[S..S+4]=%.4f %.4f %.4f %.4f %.4f\n", gpu[S], gpu[S+1], gpu[S+2], gpu[S+3], gpu[S+4]);
                fprintf(stderr, "  [debug] ref[S..S+4]=%.4f %.4f %.4f %.4f %.4f\n", ref[S], ref[S+1], ref[S+2], ref[S+3], ref[S+4]);
            }
            // Debug: check intermediate outputs (xnorm0, qkv0, attn0) to isolate the bug.
            if (getenv("MPS_DEBUG_INTER")) {
                // Re-run with intermediate targets.
                float *xnorm0 = pa_alloc((size_t)DIM*S), *qkv0 = pa_alloc((size_t)(Q_DIM+2*KV_DIM)*S), *attn0 = pa_alloc((size_t)Q_DIM*S);
                id<MTLBuffer> b_xn = [dev newBufferWithBytesNoCopy:xnorm0 length:((DIM*S*4+4095)&~4095) options:MTLResourceStorageModeShared deallocator:nil];
                id<MTLBuffer> b_qkv = [dev newBufferWithBytesNoCopy:qkv0 length:(((size_t)(Q_DIM+2*KV_DIM)*S*4+4095)&~4095) options:MTLResourceStorageModeShared deallocator:nil];
                id<MTLBuffer> b_at = [dev newBufferWithBytesNoCopy:attn0 length:((Q_DIM*S*4+4095)&~4095) options:MTLResourceStorageModeShared deallocator:nil];
                MPSGraphTensorData *td_xn = [[MPSGraphTensorData alloc] initWithMTLBuffer:b_xn shape:@[@(DIM), @(S)] dataType:MPSDataTypeFloat32];
                MPSGraphTensorData *td_qkv = [[MPSGraphTensorData alloc] initWithMTLBuffer:b_qkv shape:@[@(Q_DIM+2*KV_DIM), @(S)] dataType:MPSDataTypeFloat32];
                MPSGraphTensorData *td_at = [[MPSGraphTensorData alloc] initWithMTLBuffer:b_at shape:@[@(Q_DIM), @(S)] dataType:MPSDataTypeFloat32];
                // Rebuild feeds (the old feeds dict was autoreleased).
                NSMutableDictionary *feeds2 = [NSMutableDictionary dictionary];
                feeds2[t.x_in] = [[MPSGraphTensorData alloc] initWithMTLBuffer:[dev newBufferWithBytesNoCopy:(void*)x_in length:((DIM*S*4+4095)&~4095) options:MTLResourceStorageModeShared deallocator:nil] shape:@[@(DIM), @(S)] dataType:MPSDataTypeFloat32];
                feeds2[t.rms_final_ph] = [[MPSGraphTensorData alloc] initWithMTLBuffer:[dev newBufferWithBytesNoCopy:(void*)rms_final length:((DIM*4+4095)&~4095) options:MTLResourceStorageModeShared deallocator:nil] shape:@[@(DIM)] dataType:MPSDataTypeFloat32];
                for (int L = 0; L < NLAYERS; L++) {
                    feeds2[t.Wqkv_ph[L]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:[dev newBufferWithBytesNoCopy:(void*)(Wqkv+(size_t)L*DIM*(Q_DIM+2*KV_DIM)) length:((DIM*(Q_DIM+2*KV_DIM)*4+4095)&~4095) options:MTLResourceStorageModeShared deallocator:nil] shape:@[@(DIM), @(Q_DIM+2*KV_DIM)] dataType:MPSDataTypeFloat32];
                    feeds2[t.Wo_ph[L]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:[dev newBufferWithBytesNoCopy:(void*)(Wo+(size_t)L*Q_DIM*DIM) length:((Q_DIM*DIM*4+4095)&~4095) options:MTLResourceStorageModeShared deallocator:nil] shape:@[@(Q_DIM), @(DIM)] dataType:MPSDataTypeFloat32];
                    feeds2[t.W13_ph[L]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:[dev newBufferWithBytesNoCopy:(void*)(W13+(size_t)L*DIM*2*HIDDEN) length:((DIM*2*HIDDEN*4+4095)&~4095) options:MTLResourceStorageModeShared deallocator:nil] shape:@[@(DIM), @(2*HIDDEN)] dataType:MPSDataTypeFloat32];
                    feeds2[t.W2_ph[L]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:[dev newBufferWithBytesNoCopy:(void*)(W2+(size_t)L*HIDDEN*DIM) length:((HIDDEN*DIM*4+4095)&~4095) options:MTLResourceStorageModeShared deallocator:nil] shape:@[@(HIDDEN), @(DIM)] dataType:MPSDataTypeFloat32];
                    feeds2[t.rms_att_ph[L]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:[dev newBufferWithBytesNoCopy:(void*)(rms_att+(size_t)L*DIM) length:((DIM*4+4095)&~4095) options:MTLResourceStorageModeShared deallocator:nil] shape:@[@(DIM)] dataType:MPSDataTypeFloat32];
                    feeds2[t.rms_ffn_ph[L]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:[dev newBufferWithBytesNoCopy:(void*)(rms_ffn+(size_t)L*DIM) length:((DIM*4+4095)&~4095) options:MTLResourceStorageModeShared deallocator:nil] shape:@[@(DIM)] dataType:MPSDataTypeFloat32];
                }
                NSMutableDictionary *res2 = [NSMutableDictionary dictionary];
                res2[t.dbg_xnorm0] = td_xn;
                res2[t.dbg_qkv0] = td_qkv;
                res2[t.dbg_attn0] = td_at;
                id<MTLCommandBuffer> cb2 = [q commandBuffer];
                MPSCommandBuffer *mps_cb2 = [MPSCommandBuffer commandBufferWithCommandBuffer:cb2];
                [t.graph encodeToCommandBuffer:mps_cb2 feeds:feeds2 targetOperations:nil resultsDictionary:res2 executionDescriptor:nil];
                [mps_cb2 commit]; [mps_cb2 waitUntilCompleted];
                // CPU reference intermediates.
                float *cpu_xn = pa_alloc((size_t)DIM*S); cpu_rmsnorm(cpu_xn, x_in, rms_att, DIM, S);
                float *cpu_qkv = pa_alloc((size_t)(Q_DIM+2*KV_DIM)*S); cpu_matmul(Wqkv, cpu_xn, cpu_qkv, DIM, Q_DIM+2*KV_DIM, S);
                float *cpu_at = pa_alloc((size_t)Q_DIM*S); cpu_attention(cpu_at, cpu_qkv, cpu_qkv+(size_t)Q_DIM*S, cpu_qkv+(size_t)(Q_DIM+KV_DIM)*S, B);
                double err_xn=0, err_qkv=0, err_at=0;
                for (int i = 0; i < DIM*S; i++) { double e=fabs((double)xnorm0[i]-cpu_xn[i]); if(e>err_xn)err_xn=e; }
                for (int i = 0; i < (Q_DIM+2*KV_DIM)*S; i++) { double e=fabs((double)qkv0[i]-cpu_qkv[i]); if(e>err_qkv)err_qkv=e; }
                for (int i = 0; i < Q_DIM*S; i++) { double e=fabs((double)attn0[i]-cpu_at[i]); if(e>err_at)err_at=e; }
                fprintf(stderr, "  [inter] xnorm0 max_err=%.4e  qkv0 max_err=%.4e  attn0 max_err=%.4e\n", err_xn, err_qkv, err_at);
                // Check o0 and x_after_L0.
                float *o0 = pa_alloc((size_t)DIM*S), *xL0 = pa_alloc((size_t)DIM*S);
                id<MTLBuffer> b_o0 = [dev newBufferWithBytesNoCopy:o0 length:((DIM*S*4+4095)&~4095) options:MTLResourceStorageModeShared deallocator:nil];
                id<MTLBuffer> b_xL0 = [dev newBufferWithBytesNoCopy:xL0 length:((DIM*S*4+4095)&~4095) options:MTLResourceStorageModeShared deallocator:nil];
                MPSGraphTensorData *td_o0 = [[MPSGraphTensorData alloc] initWithMTLBuffer:b_o0 shape:@[@(DIM), @(S)] dataType:MPSDataTypeFloat32];
                MPSGraphTensorData *td_xL0 = [[MPSGraphTensorData alloc] initWithMTLBuffer:b_xL0 shape:@[@(DIM), @(S)] dataType:MPSDataTypeFloat32];
                NSMutableDictionary *res3 = [NSMutableDictionary dictionary];
                res3[t.dbg_o0] = td_o0;
                res3[t.dbg_x_after_L0] = td_xL0;
                id<MTLCommandBuffer> cb3 = [q commandBuffer];
                MPSCommandBuffer *mps_cb3 = [MPSCommandBuffer commandBufferWithCommandBuffer:cb3];
                [t.graph encodeToCommandBuffer:mps_cb3 feeds:feeds2 targetOperations:nil resultsDictionary:res3 executionDescriptor:nil];
                [mps_cb3 commit]; [mps_cb3 waitUntilCompleted];
                // CPU o0: o = Wo^T @ attn
                float *cpu_o0 = pa_alloc((size_t)DIM*S); cpu_matmul(Wo, cpu_at, cpu_o0, Q_DIM, DIM, S);
                // CPU x_after_L0: x2 = x + alpha*o, x2norm, h13, silu* h3, ffn, x = x2 + alpha*ffn
                float *cpu_x2 = pa_alloc((size_t)DIM*S), *cpu_x2n = pa_alloc((size_t)DIM*S);
                float *cpu_h13 = pa_alloc((size_t)2*HIDDEN*S), *cpu_gate = pa_alloc((size_t)HIDDEN*S), *cpu_ffn = pa_alloc((size_t)DIM*S);
                float *cpu_xL0 = pa_alloc((size_t)DIM*S);
                for (int i = 0; i < DIM*S; i++) cpu_x2[i] = x_in[i] + alpha*cpu_o0[i];
                cpu_rmsnorm(cpu_x2n, cpu_x2, rms_ffn, DIM, S);
                cpu_matmul(W13, cpu_x2n, cpu_h13, DIM, 2*HIDDEN, S);
                const float *ch1 = cpu_h13, *ch3 = cpu_h13 + (size_t)HIDDEN*S;
                for (int i = 0; i < HIDDEN*S; i++) { float sig = 1.0f/(1.0f+expf(-ch1[i])); cpu_gate[i] = (ch1[i]*sig)*ch3[i]; }
                cpu_matmul(W2, cpu_gate, cpu_ffn, HIDDEN, DIM, S);
                for (int i = 0; i < DIM*S; i++) cpu_xL0[i] = cpu_x2[i] + alpha*cpu_ffn[i];
                double err_o0=0, err_xL0=0;
                for (int i = 0; i < DIM*S; i++) { double e=fabs((double)o0[i]-cpu_o0[i]); if(e>err_o0)err_o0=e; }
                for (int i = 0; i < DIM*S; i++) { double e=fabs((double)xL0[i]-cpu_xL0[i]); if(e>err_xL0)err_xL0=e; }
                fprintf(stderr, "  [inter] o0 max_err=%.4e  x_after_L0 max_err=%.4e\n", err_o0, err_xL0);
                // Check x_before_final (output of L1) and final rmsnorm.
                float *xbf = pa_alloc((size_t)DIM*S);
                id<MTLBuffer> b_xbf = [dev newBufferWithBytesNoCopy:xbf length:((DIM*S*4+4095)&~4095) options:MTLResourceStorageModeShared deallocator:nil];
                MPSGraphTensorData *td_xbf = [[MPSGraphTensorData alloc] initWithMTLBuffer:b_xbf shape:@[@(DIM), @(S)] dataType:MPSDataTypeFloat32];
                NSMutableDictionary *res4 = [NSMutableDictionary dictionary];
                res4[t.dbg_x_before_final] = td_xbf;
                id<MTLCommandBuffer> cb4 = [q commandBuffer];
                MPSCommandBuffer *mps_cb4 = [MPSCommandBuffer commandBufferWithCommandBuffer:cb4];
                [t.graph encodeToCommandBuffer:mps_cb4 feeds:feeds2 targetOperations:nil resultsDictionary:res4 executionDescriptor:nil];
                [mps_cb4 commit]; [mps_cb4 waitUntilCompleted];
                // CPU x_before_final: run L1 on top of cpu_xL0.
                float *cpu_x1 = pa_alloc((size_t)DIM*S); memcpy(cpu_x1, cpu_xL0, (size_t)DIM*S*4);
                float *cpu_xn1 = pa_alloc((size_t)DIM*S), *cpu_qkv1 = pa_alloc((size_t)(Q_DIM+2*KV_DIM)*S);
                float *cpu_at1 = pa_alloc((size_t)Q_DIM*S), *cpu_o1 = pa_alloc((size_t)DIM*S);
                float *cpu_x2_1 = pa_alloc((size_t)DIM*S), *cpu_x2n1 = pa_alloc((size_t)DIM*S);
                float *cpu_h13_1 = pa_alloc((size_t)2*HIDDEN*S), *cpu_gate1 = pa_alloc((size_t)HIDDEN*S), *cpu_ffn1 = pa_alloc((size_t)DIM*S);
                cpu_rmsnorm(cpu_xn1, cpu_x1, rms_att+DIM, DIM, S);  // L=1 rms_att
                cpu_matmul(Wqkv+DIM*(Q_DIM+2*KV_DIM), cpu_xn1, cpu_qkv1, DIM, Q_DIM+2*KV_DIM, S);  // L=1 Wqkv
                cpu_attention(cpu_at1, cpu_qkv1, cpu_qkv1+(size_t)Q_DIM*S, cpu_qkv1+(size_t)(Q_DIM+KV_DIM)*S, B);
                cpu_matmul(Wo+Q_DIM*DIM, cpu_at1, cpu_o1, Q_DIM, DIM, S);  // L=1 Wo
                for (int i = 0; i < DIM*S; i++) cpu_x2_1[i] = cpu_x1[i] + alpha*cpu_o1[i];
                cpu_rmsnorm(cpu_x2n1, cpu_x2_1, rms_ffn+DIM, DIM, S);  // L=1 rms_ffn
                cpu_matmul(W13+DIM*2*HIDDEN, cpu_x2n1, cpu_h13_1, DIM, 2*HIDDEN, S);  // L=1 W13
                const float *ch1_1 = cpu_h13_1, *ch3_1 = cpu_h13_1 + (size_t)HIDDEN*S;
                for (int i = 0; i < HIDDEN*S; i++) { float sig = 1.0f/(1.0f+expf(-ch1_1[i])); cpu_gate1[i] = (ch1_1[i]*sig)*ch3_1[i]; }
                cpu_matmul(W2+HIDDEN*DIM, cpu_gate1, cpu_ffn1, HIDDEN, DIM, S);  // L=1 W2
                float *cpu_xbf = pa_alloc((size_t)DIM*S);
                for (int i = 0; i < DIM*S; i++) cpu_xbf[i] = cpu_x2_1[i] + alpha*cpu_ffn1[i];
                double err_xbf=0;
                for (int i = 0; i < DIM*S; i++) { double e=fabs((double)xbf[i]-cpu_xbf[i]); if(e>err_xbf)err_xbf=e; }
                fprintf(stderr, "  [inter] x_before_final max_err=%.4e\n", err_xbf);
                // Check: does CPU rmsnorm(xbf, rms_final) match the GPU's x_out?
                float *cpu_final = pa_alloc((size_t)DIM*S);
                cpu_rmsnorm(cpu_final, xbf, rms_final, DIM, S);
                // Re-run the graph to get x_out via feeds2 (same feeds as the debug run).
                float *gpu_final = pa_alloc((size_t)DIM*S);
                id<MTLBuffer> b_gf = [dev newBufferWithBytesNoCopy:gpu_final length:((DIM*S*4+4095)&~4095) options:MTLResourceStorageModeShared deallocator:nil];
                MPSGraphTensorData *td_gf = [[MPSGraphTensorData alloc] initWithMTLBuffer:b_gf shape:@[@(DIM), @(S)] dataType:MPSDataTypeFloat32];
                NSMutableDictionary *res5 = [NSMutableDictionary dictionary];
                res5[t.x_out] = td_gf;
                id<MTLCommandBuffer> cb5 = [q commandBuffer];
                MPSCommandBuffer *mps_cb5 = [MPSCommandBuffer commandBufferWithCommandBuffer:cb5];
                [t.graph encodeToCommandBuffer:mps_cb5 feeds:feeds2 targetOperations:nil resultsDictionary:res5 executionDescriptor:nil];
                [mps_cb5 commit]; [mps_cb5 waitUntilCompleted];
                double err_final=0;
                for (int i = 0; i < DIM*S; i++) { double e=fabs((double)gpu_final[i]-cpu_final[i]); if(e>err_final)err_final=e; }
                fprintf(stderr, "  [inter] final_rmsnorm: gpu_x_out vs cpu_rmsnorm(gpu_xbf) max_err=%.4e\n", err_final);
                fprintf(stderr, "  [inter] gpu_final[0..2]=%.4f %.4f %.4f  cpu_final[0..2]=%.4f %.4f %.4f\n", gpu_final[0],gpu_final[1],gpu_final[2], cpu_final[0],cpu_final[1],cpu_final[2]);
                free(cpu_final); free(gpu_final);
                if (err_xbf > 1e-4) fprintf(stderr, "  [inter] xbf gpu[0..2]=%.4f %.4f %.4f  cpu[0..2]=%.4f %.4f %.4f\n", xbf[0],xbf[1],xbf[2], cpu_xbf[0],cpu_xbf[1],cpu_xbf[2]);
                free(xbf); free(cpu_x1); free(cpu_xn1); free(cpu_qkv1); free(cpu_at1); free(cpu_o1);
                free(cpu_x2_1); free(cpu_x2n1); free(cpu_h13_1); free(cpu_gate1); free(cpu_ffn1); free(cpu_xbf);
                if (err_o0 > 1e-4) fprintf(stderr, "  [inter] o0 gpu[0..2]=%.4f %.4f %.4f  cpu[0..2]=%.4f %.4f %.4f\n", o0[0],o0[1],o0[2], cpu_o0[0],cpu_o0[1],cpu_o0[2]);
                if (err_xL0 > 1e-4) fprintf(stderr, "  [inter] xL0 gpu[0..2]=%.4f %.4f %.4f  cpu[0..2]=%.4f %.4f %.4f\n", xL0[0],xL0[1],xL0[2], cpu_xL0[0],cpu_xL0[1],cpu_xL0[2]);
                free(o0); free(xL0); free(cpu_o0); free(cpu_x2); free(cpu_x2n); free(cpu_h13); free(cpu_gate); free(cpu_ffn); free(cpu_xL0);
                fprintf(stderr, "  [inter] xnorm0 gpu[0..2]=%.4f %.4f %.4f  cpu[0..2]=%.4f %.4f %.4f\n", xnorm0[0],xnorm0[1],xnorm0[2], cpu_xn[0],cpu_xn[1],cpu_xn[2]);
                fprintf(stderr, "  [inter] qkv0   gpu[0..2]=%.4f %.4f %.4f  cpu[0..2]=%.4f %.4f %.4f\n", qkv0[0],qkv0[1],qkv0[2], cpu_qkv[0],cpu_qkv[1],cpu_qkv[2]);
                fprintf(stderr, "  [inter] attn0   gpu[0..2]=%.4f %.4f %.4f  cpu[0..2]=%.4f %.4f %.4f\n", attn0[0],attn0[1],attn0[2], cpu_at[0],cpu_at[1],cpu_at[2]);
                free(xnorm0); free(qkv0); free(attn0); free(cpu_xn); free(cpu_qkv); free(cpu_at);
            }

            // (2) Throughput: warm up 20 iters, then time 200.
            for (int w = 0; w < 20; w++) {
                float *tmp = run_graph(&t, dev, q, x_in, B, S, Wqkv, Wo, W13, W2, rms_att, rms_ffn, rms_final);
                free(tmp);
            }
            uint64_t t0 = mach_absolute_time();
            int iters = 200;
            for (int i = 0; i < iters; i++) {
                float *tmp = run_graph(&t, dev, q, x_in, B, S, Wqkv, Wo, W13, W2, rms_att, rms_ffn, rms_final);
                free(tmp);
            }
            double ms = tb_ms(mach_absolute_time() - t0) / iters;
            double games_h = 3600.0 / (ms * 1e-3) / (SEQ / (double)SEQ);  // games/hour = 3600 / (s/forward * plies/game) — but 1 forward = 1 ply
            // Actually, games/hour = 3600 / (s per forward * plies per game).
            // For max_plies=20: 20 plies/game. But this is just the trunk forward time.
            // Compare to the ANE trunk: 16.9ms/forward at B=64.
            printf("  throughput: %.3f ms/forward  (ANE+CPU ref: ~16.9ms at B=64, ~25.3ms at B=96, ~42.2ms at B=160)\n", ms);
            double ane_ref = (B==64) ? 16.9 : (B==96 ? 25.3 : 42.2);
            printf("  speedup vs ANE+CPU: %.2fx\n", ane_ref / ms);
            free(ref); free(gpu); free(x_in);
            printf("\n");
        }
        printf("## Verdict: if speedup > 1.5x, proceed with MPSGraph integration.\n");
        printf("## If > 3x, the 100x hypothesis is plausible with further optimization.\n");
        return 0;
    }
}
