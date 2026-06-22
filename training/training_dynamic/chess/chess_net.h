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

// MPS/Metal matmul backend (GPU/MPS rewrite, Phase 1 — the ane_matmul seam).
// Guarded by __has_include so non-Metal builds (pure-C tests) stay clean.
#if __has_include(<Metal/Metal.h>)
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#define LILBRO_HAS_MPS 1
#endif

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

#if LILBRO_HAS_MPS
// Forward declarations: the MPS backend is defined after ane_matmul, but ane_matmul
// dispatches to it when g_use_mps=1. Defined in the MPS section below.
static int g_use_mps;
static int g_use_mps_graph;   // Phase 2: MPSGraph whole-forward (set by --mps-graph)
static void mps_matmul(int ic, int oc, int seq, const float *x, const float *W, float *y);
#endif

// Optional trunk-forward sub-profiler (zero-overhead when g_trunk_prof==0). Splits each
// ane_matmul into io (fp32<->fp16 convert + IOSurfaceLock, the CPU copy/dispatch tax) vs
// ane (the blocking evaluateWithQoS dispatch). Set by the bench; read after the run.
// cpu_ops (attn/rms/silu/residual/memcpy) = prof_trunk_s - (g_trunk_io_s + g_trunk_ane_s).
static int    g_trunk_prof = 0;
static double g_trunk_io_s = 0.0, g_trunk_ane_s = 0.0;
static double g_trunk_attn_s = 0.0;   // attn_cpu_forward_batched time (the suspected hot op)
static double g_trunk_rms_s = 0.0, g_trunk_silu_s = 0.0, g_trunk_softmax_s = 0.0;

// Parallel elementwise over N contiguous elements in ~12*8 GCD chunks. ELEMENTWISE BODIES
// ONLY (no cross-element reduction): each element is computed once, in the same FP order as
// the serial loop, just distributed across cores => bit-identical output, bench checksum
// preserved. The seam used to parallelize the trunk's residual/SiLU/embed loops. [iter 3]
static void chess_parallel_for(long N, void (^body)(long lo, long hi)) {
    if (N <= 0) return;
    long nc = N / 2048;                 // ~2048 elems/chunk: coarse enough to amortize dispatch
    if (nc < 1) nc = 1;
    if (nc > 96) nc = 96;               // cap: 12 P-cores * 8 for load balance
    if (nc <= 1) { body(0, N); return; }
    dispatch_queue_t dq = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
    dispatch_apply(nc, dq, ^(size_t ci) {
        long lo = (long)ci * N / nc, hi = (long)(ci + 1) * N / nc;
        body(lo, hi);
    });
}

// NEON expf via 2^x decomposition (x*ln2(e) -> floor + frac -> poly * 2^int). Degree-4 Horner
// poly for exp2 on [0,1): max err ~1e-5 (softmax-grade; the softmax ratio is robust to this).
// ~4x over scalar expf on the attention softmax (the measured 18.5%-of-gen hot spot). [iter 9]
static inline float32x4_t neon_expf(float32x4_t x) {
    x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-88.0f)), vdupq_n_f32(88.0f));
    float32x4_t p = vmulq_n_f32(x, 1.4426950408889634f);    // x * log2(e)
    float32x4_t ip = vrndmq_f32(p);                          // floor(p)
    float32x4_t fp = vsubq_f32(p, ip);                        // frac in [0,1)
    // Horner for exp2(fp) = c0 + c1*f + c2*f^2 + c3*f^3 + c4*f^4; vmlaq_f32(a,b,v)=a+b*v
    float32x4_t a = vdupq_n_f32(0.0096181f);                  // c4
    a = vmlaq_f32(vdupq_n_f32(0.0555041f), a, fp);            // c3 + c4*f
    a = vmlaq_f32(vdupq_n_f32(0.2402265f), a, fp);            // c2 + a*f
    a = vmlaq_f32(vdupq_n_f32(0.6931472f), a, fp);            // c1 + a*f
    a = vmlaq_f32(vdupq_n_f32(1.0f), a, fp);                  // c0 + a*f
    int32x4_t e = vaddq_s32(vcvtq_s32_f32(ip), vdupq_n_s32(127));
    float32x4_t scale = vreinterpretq_f32_s32(vshlq_n_s32(e, 23));
    return vmulq_f32(a, scale);
}

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
#if LILBRO_HAS_MPS
    if (g_use_mps) { mps_matmul(ic, oc, seq, x, W, y); return; }
#endif
    Kern *k = mm_kernel(ic, oc, seq);
    uint64_t _tw = 0, _ta = 0;
    if (g_trunk_prof) _tw = mach_absolute_time();
    io_write_dyn(k->ioIn, x, ic, seq, W, oc);
    if (g_trunk_prof) { g_trunk_io_s += tb_ms(mach_absolute_time() - _tw) * 1e-3; _ta = mach_absolute_time(); }
    ane_eval(k);
    if (g_trunk_prof) { g_trunk_ane_s += tb_ms(mach_absolute_time() - _ta) * 1e-3; _tw = mach_absolute_time(); }
    io_read_dyn(k->ioOut, y, oc, seq);
    if (g_trunk_prof) g_trunk_io_s += tb_ms(mach_absolute_time() - _tw) * 1e-3;
}

// ============================================================================
// MPS matmul backend: y[oc,seq] = W[ic,oc]^T @ x[ic,seq] via Metal/MPS.
// Drop-in replacement for ane_matmul at the same seam. The ANE path pays
// fp32->fp16 + IOSurfaceLock + ~0.2-0.4ms dispatch floor + IOSurfaceLock +
// fp16->fp32 per eval (the ane+io = 40% of gen wall). MPS on Apple Silicon
// unified memory skips ALL of that: zero-copy MTLBuffers (newBufferWithBytesNoCopy)
// wrap existing page-aligned float* with NO copy, MPSDataTypeFloat32 needs NO
// conversion, and MPSMatrixMultiplication encodes the matmul as a Metal compute
// command. probe_mps measured ~2.3x faster per matmul vs ANE at B=64.
//
// Kernel cache: MPSMatrixMultiplication is init'd per (ic,oc,seq) — same key as
// the ANE g_mm cache. Buffer cache: zero-copy MTLBuffers are cached by host ptr
// (the trunk reuses the same CActs/CLayer buffers every forward, so a ptr-keyed
// cache avoids re-creating MTLBuffers). Weights are persistent (optimizer writes
// to the host ptr → automatically visible via shared memory). Activations change
// every forward (CPU writes → GPU reads, GPU writes → CPU reads, same memory).
// Requires page-aligned host memory — fmalloc/fcalloc use posix_memalign(4096).
// ============================================================================
#if LILBRO_HAS_MPS
static id<MTLDevice>        g_mtl_dev   = nil;
static id<MTLCommandQueue>  g_mtl_queue = nil;

typedef struct { int ic, oc, seq; __strong MPSMatrixMultiplication *kern; } MpsKernEntry;
static MpsKernEntry g_mps_kerns[256];
static int g_nmps_kerns = 0;

typedef struct { const void *ptr; size_t bytes; __strong id<MTLBuffer> buf; int zero_copy; } MpsBufEntry;
static MpsBufEntry g_mps_bufs[128];
static int g_nmps_bufs = 0;

static void mps_init(void) {
    if (g_mtl_dev) return;
    g_mtl_dev = MTLCreateSystemDefaultDevice();
    if (!g_mtl_dev) { fprintf(stderr, "[mps] MTLCreateSystemDefaultDevice FAILED — no Metal device\n"); return; }
    g_mtl_queue = [g_mtl_dev newCommandQueue];
    if (!g_mtl_queue) { fprintf(stderr, "[mps] newCommandQueue FAILED\n"); g_mtl_dev = nil; return; }
    memset(g_mps_kerns, 0, sizeof(g_mps_kerns));
    memset(g_mps_bufs, 0, sizeof(g_mps_bufs));
    g_nmps_kerns = 0;
    g_nmps_bufs = 0;
}

static MPSMatrixMultiplication *mps_kernel(int ic, int oc, int seq) {
    for (int i = 0; i < g_nmps_kerns; i++)
        if (g_mps_kerns[i].ic==ic && g_mps_kerns[i].oc==oc && g_mps_kerns[i].seq==seq)
            return g_mps_kerns[i].kern;
    if (g_nmps_kerns >= (int)(sizeof(g_mps_kerns)/sizeof(g_mps_kerns[0]))) {
        fprintf(stderr, "[mps] kernel cache overflow (%d shapes)\n", g_nmps_kerns); abort();
    }
    // C = alpha * A^T @ B. A=W[ic,oc], B=x[ic,seq], C=y[oc,seq].
    // M=oc (result rows), N=seq (result cols), K=ic (interior).
    MPSMatrixMultiplication *k = [[MPSMatrixMultiplication alloc] initWithDevice:g_mtl_dev
                transposeLeft:YES transposeRight:NO
                resultRows:oc resultColumns:seq interiorColumns:ic
                alpha:1.0f beta:0.0f];
    if (!k) { fprintf(stderr, "[mps] MPSMatrixMultiplication init FAILED ic=%d oc=%d seq=%d\n", ic, oc, seq); abort(); }
    g_mps_kerns[g_nmps_kerns] = (MpsKernEntry){ic, oc, seq, k};
    g_nmps_kerns++;
    return k;
}

// Get or create a zero-copy MTLBuffer for a host pointer. The trunk reuses the
// same float* buffers every forward (CActs/CLayer), so this cache hits after the
// first forward. Zero-copy: the MTLBuffer IS the host memory (shared mode) — no
// copy, no coherency sync. Falls back to a managed buffer (copy) if the ptr
// isn't page-aligned (posix_memalign in fmalloc should prevent this).
// GROWTH: the same host buffer is used at different seq values (B=1 selfcheck
// warmup seq=96, then B=64 bench seq=6144). The cache tracks the max bytes seen
// per ptr and (re)creates the MTLBuffer when a larger request arrives. Zero-copy
// recreation is free (same host ptr, larger length); managed recreation
// reallocates. is_zero_copy is set to 1 for zero-copy, 0 for managed.
static id<MTLBuffer> mps_get_buf(const void *ptr, size_t bytes, int *is_zero_copy) {
    for (int i = 0; i < g_nmps_bufs; i++)
        if (g_mps_bufs[i].ptr == ptr) {
            if (g_mps_bufs[i].bytes >= bytes) {
                *is_zero_copy = g_mps_bufs[i].zero_copy;
                return g_mps_bufs[i].buf;
            }
            // Cached buffer too small — recreate at the larger size.
            // Zero-copy: the host allocation is large enough (fmalloc sized for
            // maxS), so we just wrap the same ptr with a larger length. Managed:
            // drop the old buffer and allocate a bigger one.
            size_t pg = 4096;
            size_t buf_bytes = (bytes + pg - 1) & ~(pg - 1);
            id<MTLBuffer> buf = nil;
            int zc = 0;
            if (((uintptr_t)ptr & (pg - 1)) == 0) {
                buf = [g_mtl_dev newBufferWithBytesNoCopy:(void *)ptr length:buf_bytes
                            options:MTLResourceStorageModeShared deallocator:nil];
                if (buf) zc = 1;
            }
            if (!buf) {
                buf = [g_mtl_dev newBufferWithLength:buf_bytes options:MTLResourceStorageModeShared];
                if (!buf) { fprintf(stderr, "[mps] buffer realloc FAILED (%zu bytes)\n", buf_bytes); abort(); }
            }
            g_mps_bufs[i].buf = buf;
            g_mps_bufs[i].bytes = bytes;
            g_mps_bufs[i].zero_copy = zc;
            *is_zero_copy = zc;
            return buf;
        }
    if (g_nmps_bufs >= (int)(sizeof(g_mps_bufs)/sizeof(g_mps_bufs[0]))) {
        fprintf(stderr, "[mps] buffer cache overflow (%d bufs)\n", g_nmps_bufs); abort();
    }
    // Round length up to page boundary (newBufferWithBytesNoCopy requires it).
    size_t pg = 4096;
    size_t buf_bytes = (bytes + pg - 1) & ~(pg - 1);
    id<MTLBuffer> buf = nil;
    int zc = 0;
    // Try zero-copy first (requires page-aligned address).
    if (((uintptr_t)ptr & (pg - 1)) == 0) {
        buf = [g_mtl_dev newBufferWithBytesNoCopy:(void *)ptr length:buf_bytes
                    options:MTLResourceStorageModeShared deallocator:nil];
        if (buf) zc = 1;
    }
    if (!buf) {
        // Fallback: managed buffer (allocates; caller copies in/out per call).
        buf = [g_mtl_dev newBufferWithLength:buf_bytes options:MTLResourceStorageModeShared];
        if (!buf) { fprintf(stderr, "[mps] buffer alloc FAILED (%zu bytes)\n", buf_bytes); abort(); }
    }
    g_mps_bufs[g_nmps_bufs] = (MpsBufEntry){ptr, bytes, buf, zc};
    g_nmps_bufs++;
    *is_zero_copy = zc;
    return buf;
}

static void mps_matmul(int ic, int oc, int seq, const float *x, const float *W, float *y) {
    @autoreleasepool {
        MPSMatrixMultiplication *kern = mps_kernel(ic, oc, seq);
        size_t bytesW = (size_t)ic * oc * 4;
        size_t bytesX = (size_t)ic * seq * 4;
        size_t bytesY = (size_t)oc * seq * 4;
        int zcW, zcX, zcY;
        id<MTLBuffer> bufW = mps_get_buf(W, bytesW, &zcW);
        id<MTLBuffer> bufX = mps_get_buf(x, bytesX, &zcX);
        id<MTLBuffer> bufY = mps_get_buf(y, bytesY, &zcY);
        // Managed (non-zero-copy) inputs: copy current data in.
        if (!zcW) memcpy(bufW.contents, W, bytesW);
        if (!zcX) memcpy(bufX.contents, x, bytesX);

        // Create fresh MPSMatrix objects from the CURRENT MTLBuffers each call.
        // (The previous MPSMatrix-by-ptr cache was UNSAFE: when mps_get_buf recreates
        // a MTLBuffer for a reused host ptr at a different size, the cached MPSMatrix
        // still referenced the OLD MTLBuffer while mps_matmul copied data to the NEW
        // one — the GPU read stale/freed memory and produced NaN. The backward's
        // temporary transposed-weight buffers trigger this: they're fmalloc'd/freed
        // each step, and malloc reuses ptrs. Creating fresh MPSMatrix objects is
        // ~2.6x overhead vs the cache but is correct, and this path is learner-only.)
        MPSMatrix *mW = nil, *mX = nil, *mY = nil;
        {
            MPSMatrixDescriptor *dW = [MPSMatrixDescriptor matrixDescriptorWithRows:ic columns:oc
                                        rowBytes:(oc*4) dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor *dX = [MPSMatrixDescriptor matrixDescriptorWithRows:ic columns:seq
                                        rowBytes:(seq*4) dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor *dY = [MPSMatrixDescriptor matrixDescriptorWithRows:oc columns:seq
                                        rowBytes:(seq*4) dataType:MPSDataTypeFloat32];
            mW = [[MPSMatrix alloc] initWithBuffer:bufW descriptor:dW];
            mX = [[MPSMatrix alloc] initWithBuffer:bufX descriptor:dX];
            mY = [[MPSMatrix alloc] initWithBuffer:bufY descriptor:dY];
        }

        uint64_t _t = 0;
        if (g_trunk_prof) _t = mach_absolute_time();
        id<MTLCommandBuffer> cb = [g_mtl_queue commandBuffer];
        [kern encodeToCommandBuffer:cb leftMatrix:mW rightMatrix:mX resultMatrix:mY];
        [cb commit]; [cb waitUntilCompleted];
        if (g_trunk_prof) g_trunk_ane_s += tb_ms(mach_absolute_time() - _t) * 1e-3;
        // Managed output: copy GPU result back to host. Zero-copy: already written.
        if (!zcY) memcpy(y, bufY.contents, bytesY);
    }
}
#endif // LILBRO_HAS_MPS

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
// CLayer.Wqkv/W13 are FUSED forward-only weights (QKV 3->1, W1/W3 2->1) that cut the
// ANE eval count 14->8/forward; NULL until chess_net_build_fused, rebuilt after every
// optimizer step. The canonical Wq/Wk/Wv/W1/W3 stay the source of truth (checkpoint +
// optimizer + backward use THEM), so the fusion is forward-only and touches neither
// the checkpoint format nor the backward. [iter 6]
// ============================================================================
typedef struct { float *Wq,*Wk,*Wv,*Wo,*W1,*W2,*W3,*rms_att,*rms_ffn; float *Wqkv, *W13; } CLayer;
// CActs.qkv/h13 are the fused activation storage: Q/K/V are contiguous row-slices of qkv
// ([Q_DIM+2*KV_DIM, S]); h1/h3 are contiguous row-slices of h13 ([2*HIDDEN, S]). The Q/K/V/
// h1/h3 pointers are set into them by cacts_alloc, so all existing indexing (attn, silu,
// backward) works unchanged. [iter 6]
typedef struct { float *layer_in,*xnorm,*qkv,*Q,*K,*V,*attn,*x2,*x2norm,*h13,*h1,*h3,*gate; } CActs;

// Page-aligned alloc for MPS zero-copy (newBufferWithBytesNoCopy requires
// page-aligned address + length). posix_memalign(4096) ensures all trunk
// buffers (weights + activations) qualify. The extra bytes from length
// rounding are zero and unused by the matmul.
static float *fmalloc(size_t n) {
    void *p = NULL; size_t bytes = ((n*4 + 4095) & ~4095);
    if (posix_memalign(&p, 4096, bytes)) { fprintf(stderr, "[fmalloc] OOM (%zu)\n", bytes); abort(); }
    return (float*)p;
}
static float *fcalloc(size_t n) {
    float *p = fmalloc(n); memset(p, 0, n*4); return p;
}

static void clayer_alloc(CLayer *w) {
    w->Wq=fmalloc(DIM*Q_DIM); w->Wk=fmalloc(DIM*KV_DIM); w->Wv=fmalloc(DIM*KV_DIM);
    w->Wo=fmalloc(Q_DIM*DIM); w->W1=fmalloc(DIM*HIDDEN); w->W2=fmalloc(HIDDEN*DIM);
    w->W3=fmalloc(DIM*HIDDEN); w->rms_att=fmalloc(DIM); w->rms_ffn=fmalloc(DIM);
    w->Wqkv=NULL; w->W13=NULL;
}
static void clayer_calloc(CLayer *g) {
    g->Wq=fcalloc(DIM*Q_DIM); g->Wk=fcalloc(DIM*KV_DIM); g->Wv=fcalloc(DIM*KV_DIM);
    g->Wo=fcalloc(Q_DIM*DIM); g->W1=fcalloc(DIM*HIDDEN); g->W2=fcalloc(HIDDEN*DIM);
    g->W3=fcalloc(DIM*HIDDEN); g->rms_att=fcalloc(DIM); g->rms_ffn=fcalloc(DIM);
    g->Wqkv=NULL; g->W13=NULL;
}
// Activations sized for up to `maxS` packed tokens (= maxB*SEQ). B=1 path uses maxS=SEQ.
// Q/K/V are slices of qkv; h1/h3 are slices of h13 (see CActs comment). [iter 6]
static void cacts_alloc(CActs *a, int maxS) {
    a->layer_in=fmalloc((size_t)DIM*maxS); a->xnorm=fmalloc((size_t)DIM*maxS);
    a->qkv=fmalloc((size_t)(Q_DIM+2*KV_DIM)*maxS);
    a->Q = a->qkv; a->K = a->qkv + (size_t)Q_DIM*maxS; a->V = a->qkv + (size_t)(Q_DIM+KV_DIM)*maxS;
    a->attn=fmalloc((size_t)Q_DIM*maxS); a->x2=fmalloc((size_t)DIM*maxS); a->x2norm=fmalloc((size_t)DIM*maxS);
    a->h13=fmalloc((size_t)2*HIDDEN*maxS);
    a->h1 = a->h13; a->h3 = a->h13 + (size_t)HIDDEN*maxS;
    a->gate=fmalloc((size_t)HIDDEN*maxS);
}

// Ensure the cpu_ops RMSNorm scratch is sized for `maxS` columns (it is lazily sized to
// the FIRST S it sees; the batched path must pre-size it before any rmsnorm call).
static void chess_net_init_rmstmp(int maxS) {
    if (g_rms_tmp) { free(g_rms_tmp); }
    g_rms_tmp = (float*)malloc((size_t)maxS*4);
}

// Parallel RMSNorm over S-column chunks. The per-column reduction over `d` (DIM) is serial
// within each chunk and in the SAME order as the serial rmsnorm (vDSP_vadd accumulates row
// by row), so every output column is bit-identical and the bench checksum is preserved.
// Each chunk needs its own ss/tmp scratch (the serial rmsnorm's g_rms_tmp is shared => would
// race across cores), hence the per-chunk malloc. Small S (B=1 selfcheck) falls back to the
// serial rmsnorm unchanged. [iter 5]
static void chess_rmsnorm_par(float *out, const float *x, const float *w, int d, int S) {
    if (S < 2048) { rmsnorm(out, x, w, d, S); return; }
    long nc = S / 512; if (nc < 1) nc = 1; if (nc > 24) nc = 24;
    if (nc <= 1) { rmsnorm(out, x, w, d, S); return; }
    dispatch_queue_t dq = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
    dispatch_apply(nc, dq, ^(size_t ci) {
        long lo = (long)ci * S / nc, hi = (long)(ci + 1) * S / nc;
        vDSP_Length cs = (vDSP_Length)(hi - lo);
        float *ss = (float*)malloc((size_t)cs * 4), *tmp = (float*)malloc((size_t)cs * 4);
        memset(ss, 0, (size_t)cs * 4);
        for (int i = 0; i < d; i++) {
            vDSP_vmul(x + (size_t)i*S + lo, 1, x + (size_t)i*S + lo, 1, tmp, 1, cs);
            vDSP_vadd(tmp, 1, ss, 1, ss, 1, cs);
        }
        float invd = 1.0f / d, eps = 1e-5f;
        vDSP_vsmsa(ss, 1, &invd, &eps, ss, 1, cs);
        int n = (int)cs; vvrsqrtf(ss, ss, &n);   // ss -> rrms
        for (int i = 0; i < d; i++) {
            vDSP_vmul(x + (size_t)i*S + lo, 1, ss, 1, out + (size_t)i*S + lo, 1, cs);
            vDSP_vsmul(out + (size_t)i*S + lo, 1, &w[i], out + (size_t)i*S + lo, 1, cs);
        }
        free(ss); free(tmp);
    });
}

// ============================================================================
// Batched causal attention (per-position, channel stride B*seqp). Chess v1 has no
// attention-sink / qk-norm, so this is the plain causal core of attn_cpu_forward; for
// B=1 it is byte-identical to attn_cpu_forward(...,NULL,NULL,NULL,SEQ). GQA-aware
// (kv-head = h % KV_HEADS); chess_g0 is MHA so kv-head == head.
//
// PARALLEL + NEON [iter 2/4]: the B positions are independent (disjoint output regions,
// read-only Q/K/V), so the b-loop runs over all P-cores via dispatch_apply. Per (b,h) the
// strided [channel, S] Q/K/V slice (d-stride = S, hostile to NEON) is transposed once into
// a contiguous [seqp, HD] tile; the O(seqp^2*HD) QK^T dot and the AV weighted sum then run
// as 2-way-ILP NEON fmla over HD. The softmax stays scalar (small, separate cost).
//
// FP-order note: the NEON reductions (2 partial accumulators + horizontal sum) change the
// per-(b,h) rounding vs the old serial scalar loop, so the bench CHECKSUM CHANGES. Run-to-
// run determinism holds (NEON + a fixed tile order are deterministic), and the selfcheck
// stays green because both the batched and single-position paths use this SAME function
// (per-position results are identical regardless of B). [iter 4]
// ============================================================================
static void attn_cpu_forward_batched(float *attn_out, const float *Q, const float *K,
                                     const float *V, int B, int seqp) {
    int S = B*seqp;
    float scale = 1.0f/sqrtf((float)HD);
    dispatch_queue_t dq = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
    dispatch_apply(B, dq, ^(size_t bb) {
        int b = (int)bb;
        int base = b*seqp;
        size_t tn = (size_t)seqp * HD;
        float *buf = (float*)malloc((4 * tn + (size_t)seqp) * 4);
        float *qt = buf, *kt = buf + tn, *vt = buf + 2*tn, *ot = buf + 3*tn, *sc = buf + 4*tn;
        for (int h = 0; h < HEADS; h++) {
            int kvh = h % KV_HEADS;
            // build contiguous [seqp, HD] tiles from the [HD, S] strided slice (scalar strided
            // read; O(seqp*HD) << O(seqp^2*HD) compute, ~2% of the work).
            for (int t = 0; t < seqp; t++) {
                const float *qp = Q + (size_t)(h*HD)*S + base + t;
                const float *kp = K + (size_t)(kvh*HD)*S + base + t;
                const float *vp = V + (size_t)(kvh*HD)*S + base + t;
                float *qo = qt + (size_t)t*HD, *ko = kt + (size_t)t*HD, *vo = vt + (size_t)t*HD;
                for (int d = 0; d < HD; d++) { qo[d] = qp[(size_t)d*S]; ko[d] = kp[(size_t)d*S]; vo[d] = vp[(size_t)d*S]; }
            }
            for (int q = 0; q < seqp; q++) {
                const float *qr = qt + (size_t)q*HD;
                float m = -1e30f;
                for (int j = 0; j <= q; j++) {
                    const float *kr = kt + (size_t)j*HD;
                    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
                    for (int d = 0; d < HD; d += 8) {
                        a0 = vmlaq_f32(a0, vld1q_f32(qr + d),     vld1q_f32(kr + d));
                        a1 = vmlaq_f32(a1, vld1q_f32(qr + d + 4), vld1q_f32(kr + d + 4));
                    }
                    float dot = vaddvq_f32(vaddq_f32(a0, a1));
                    sc[j] = dot * scale; if (sc[j] > m) m = sc[j];
                }
                float Z = 0.0f;
                { uint64_t _t = g_trunk_prof ? mach_absolute_time() : 0;
                  float32x4_t z4 = vdupq_n_f32(0.0f); float32x4_t vm = vdupq_n_f32(m);
                  int j = 0;
                  for (; j + 4 <= q + 1; j += 4) {
                      float32x4_t s = vsubq_f32(vld1q_f32(&sc[j]), vm);
                      s = neon_expf(s);
                      vst1q_f32(&sc[j], s);
                      z4 = vaddq_f32(z4, s);
                  }
                  Z = vaddvq_f32(z4);
                  for (; j <= q; j++) { sc[j] = expf(sc[j] - m); Z += sc[j]; }
                  if (g_trunk_prof) g_trunk_softmax_s += tb_ms(mach_absolute_time() - _t) * 1e-3; }
                float inv = 1.0f / Z;
                float *orc = ot + (size_t)q*HD;
                for (int dd = 0; dd < HD; dd += 8) {
                    float32x4_t o0 = vdupq_n_f32(0.0f), o1 = vdupq_n_f32(0.0f);
                    for (int j = 0; j <= q; j++) {
                        float32x4_t wv = vdupq_n_f32(sc[j] * inv);
                        const float *vr = vt + (size_t)j*HD + dd;
                        o0 = vmlaq_f32(o0, wv, vld1q_f32(vr));
                        o1 = vmlaq_f32(o1, wv, vld1q_f32(vr + 4));
                    }
                    vst1q_f32(orc + dd, o0);
                    vst1q_f32(orc + dd + 4, o1);
                }
            }
            // scatter [seqp, HD] tile back to the [HD, S] strided output
            for (int q = 0; q < seqp; q++)
                for (int d = 0; d < HD; d++)
                    attn_out[(size_t)(h*HD + d)*S + base + q] = ot[(size_t)q*HD + d];
        }
        free(buf);
    });
}
// Backward of attn_cpu_forward_batched. da[Q_DIM,S] -> dQ[Q_DIM,S], dK/dV[KV_DIM,S]
// (accumulated, GQA-reduced). Recomputes the softmax. B=1 == attn_cpu_backward (no knobs).
//
// PARALLEL + NEON [the measured G2 bottleneck fix]: the B positions are independent (each b
// writes DISJOINT columns [b*seqp, (b+1)*seqp) of dQ/dK/dV), so the b-loop runs over all
// P-cores via dispatch_apply — the same seam as the forward. Per (b,h) the strided Q/K/V/da
// slices are transposed once into contiguous [seqp, HD] tiles; the O(seqp^2*HD) QK^T dot, the
// dp dot, and the dQ/dK/dV accumulations run as 2-way-ILP NEON fmla. dkt/dvt persist across
// the q-loop (accumulate over q); dqt is a per-q reduction. Scatter-ADD to strided dQ/dK/dV
// after the q-loop (MHA: each h its own region; GQA: multiple h scatter-add to the same kvh
// region serially within a b -> correct). FP-order changes vs the old serial loop (NEON
// partial sums), but deterministic and selfcheck tests the FORWARD, not the backward. [iter 9]
static void attn_cpu_backward_batched(const float *da, const float *Q, const float *K,
                                      const float *V, int B, int seqp,
                                      float *dQ, float *dK, float *dV) {
    int S = B*seqp;
    float scale = 1.0f/sqrtf((float)HD);
    memset(dQ, 0, (size_t)Q_DIM*S*4); memset(dK, 0, (size_t)KV_DIM*S*4); memset(dV, 0, (size_t)KV_DIM*S*4);
    dispatch_queue_t dq = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
    dispatch_apply(B, dq, ^(size_t bb) {
        int b = (int)bb;
        int base = b*seqp;
        size_t tn = (size_t)seqp * HD;
        float *buf = (float*)malloc((7 * tn + (size_t)3*seqp) * 4);
        float *qt = buf, *kt = buf + tn, *vt = buf + 2*tn, *dat = buf + 3*tn;
        float *dqt = buf + 4*tn, *dkt = buf + 5*tn, *dvt = buf + 6*tn;
        float *sc = buf + 7*tn, *p = sc + seqp, *dp = p + seqp;
        for (int h = 0; h < HEADS; h++) {
            int kvh = h % KV_HEADS;
            // build contiguous [seqp, HD] tiles from the [HD, S] strided slices (O(seqp*HD),
            // << O(seqp^2*HD) compute). da is [Q_DIM, S]; its h-th head slice transposes too.
            for (int t = 0; t < seqp; t++) {
                const float *qp = Q  + (size_t)(h*HD)*S   + base + t;
                const float *kp = K  + (size_t)(kvh*HD)*S + base + t;
                const float *vp = V  + (size_t)(kvh*HD)*S + base + t;
                const float *dap= da + (size_t)(h*HD)*S   + base + t;
                float *qo = qt + (size_t)t*HD, *ko = kt + (size_t)t*HD, *vo = vt + (size_t)t*HD, *dao = dat + (size_t)t*HD;
                for (int d = 0; d < HD; d++) { qo[d]=qp[(size_t)d*S]; ko[d]=kp[(size_t)d*S]; vo[d]=vp[(size_t)d*S]; dao[d]=dap[(size_t)d*S]; }
            }
            memset(dkt, 0, (size_t)tn*4);
            memset(dvt, 0, (size_t)tn*4);
            memset(dqt, 0, (size_t)tn*4);
            for (int q = 0; q < seqp; q++) {
                const float *qr = qt + (size_t)q*HD;
                const float *dar = dat + (size_t)q*HD;
                float m = -1e30f;
                for (int j = 0; j <= q; j++) {
                    const float *kr = kt + (size_t)j*HD;
                    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
                    for (int d = 0; d < HD; d += 8) {
                        a0 = vmlaq_f32(a0, vld1q_f32(qr + d),     vld1q_f32(kr + d));
                        a1 = vmlaq_f32(a1, vld1q_f32(qr + d + 4), vld1q_f32(kr + d + 4));
                    }
                    float dot = vaddvq_f32(vaddq_f32(a0, a1));
                    sc[j] = dot * scale; if (sc[j] > m) m = sc[j];
                }
                float Z = 0.0f;
                {
                  float32x4_t z4 = vdupq_n_f32(0.0f); float32x4_t vm = vdupq_n_f32(m);
                  int j = 0;
                  for (; j + 4 <= q + 1; j += 4) {
                      float32x4_t s = vsubq_f32(vld1q_f32(&sc[j]), vm);
                      s = neon_expf(s);
                      vst1q_f32(&sc[j], s);
                      z4 = vaddq_f32(z4, s);
                  }
                  Z = vaddvq_f32(z4);
                  for (; j <= q; j++) { sc[j] = expf(sc[j] - m); Z += sc[j]; }
                }
                float inv = 1.0f / Z; for (int j = 0; j <= q; j++) p[j] = sc[j] * inv;
                // dp[j] = dat[q] · vt[j]  (NEON dot); dvt[j] += p[j] * dat[q]  (NEON fmla)
                for (int j = 0; j <= q; j++) {
                    const float *vr = vt + (size_t)j*HD;
                    float *dvr = dvt + (size_t)j*HD;
                    float w = p[j];
                    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
                    float32x4_t wv = vdupq_n_f32(w);
                    for (int d = 0; d < HD; d += 8) {
                        a0 = vmlaq_f32(a0, vld1q_f32(dar + d),     vld1q_f32(vr + d));
                        a1 = vmlaq_f32(a1, vld1q_f32(dar + d + 4), vld1q_f32(vr + d + 4));
                        vst1q_f32(dvr + d,     vmlaq_f32(vld1q_f32(dvr + d),     wv, vld1q_f32(dar + d)));
                        vst1q_f32(dvr + d + 4, vmlaq_f32(vld1q_f32(dvr + d + 4), wv, vld1q_f32(dar + d + 4)));
                    }
                    dp[j] = vaddvq_f32(vaddq_f32(a0, a1));
                }
                float g = 0; for (int j = 0; j <= q; j++) g += p[j]*dp[j];
                // dqt[q] += dscore[j] * kt[j]  (NEON fmla, accumulates over j into q's row);
                // dkt[j] += dscore[j] * qt[q]  (NEON fmla, accumulates across q)
                float *dqrr = dqt + (size_t)q*HD;
                for (int j = 0; j <= q; j++) {
                    float dscore = p[j]*(dp[j]-g)*scale;
                    const float *kr = kt + (size_t)j*HD;
                    float *dkrr = dkt + (size_t)j*HD;
                    float32x4_t ds = vdupq_n_f32(dscore);
                    for (int d = 0; d < HD; d += 8) {
                        vst1q_f32(dqrr + d,     vmlaq_f32(vld1q_f32(dqrr + d),     ds, vld1q_f32(kr + d)));
                        vst1q_f32(dqrr + d + 4, vmlaq_f32(vld1q_f32(dqrr + d + 4), ds, vld1q_f32(kr + d + 4)));
                        vst1q_f32(dkrr + d,     vmlaq_f32(vld1q_f32(dkrr + d),     ds, vld1q_f32(qr + d)));
                        vst1q_f32(dkrr + d + 4, vmlaq_f32(vld1q_f32(dkrr + d + 4), ds, vld1q_f32(qr + d + 4)));
                    }
                }
            }
            // scatter-add the per-(b,h) tiles back to the [HD, S] strided grads (disjoint b cols;
            // GQA: multiple h serially scatter-add to the same kvh region within a b -> correct)
            for (int q = 0; q < seqp; q++)
                for (int d = 0; d < HD; d++)
                    dQ[(size_t)(h*HD + d)*S + base + q] += dqt[(size_t)q*HD + d];
            for (int j = 0; j < seqp; j++)
                for (int d = 0; d < HD; d++) {
                    dK[(size_t)(kvh*HD + d)*S + base + j] += dkt[(size_t)j*HD + d];
                    dV[(size_t)(kvh*HD + d)*S + base + j] += dvt[(size_t)j*HD + d];
                }
        }
        free(buf);
    });
}

// ============================================================================
// MPSGraph whole-forward backend (Phase 2 — the pipeline-bubble-elimination win).
// The ENTIRE trunk (8 matmuls + attention + rms + silu + residual + final rmsnorm)
// as ONE MPSGraph in ONE command buffer — zero CPU<->GPU sync between ops, no
// pipeline bubbles. probe_mps_graph measured 5.3-6.0x vs ANE+CPU (3.2ms vs 16.9ms
// at B=64). Eval-only (save_acts==0): the learner (save_acts==1) needs all
// intermediates for the backward, so it stays on the ANE+CPU path. Generation +
// the eval ladder are save_acts==0 — the dominant cost (the bench is 94-95% trunk).
//
// Graph caching: shapes are baked into the graph at build time (S = B*SEQ). The eval
// path pads B to a multiple of 32, so only a handful of distinct S values appear
// (96, 3072, 6144, 9216, 12288, 15360). Each gets its own compiled graph, cached
// by S. Buffer caching: reuses the Phase 1 mps_get_buf (ptr-keyed, zero-copy
// MTLBuffers wrapping the fmalloc'd page-aligned host memory). Weights are
// persistent (same ptr every forward; the optimizer writes to the host ptr -> the
// GPU sees updated values via shared memory). x_in is CPU-written (embed+posenc)
// then GPU-read; x_final is GPU-written then CPU-read (readout) — both zero-copy.
// ============================================================================
#if LILBRO_HAS_MPS
typedef struct {
    __strong MPSGraph *graph;
    int S;   // packed S = P*SEQ (shapes baked in at build)
    __strong MPSGraphTensor *x_in_ph, *rms_final_ph;
    __strong MPSGraphTensor *Wqkv_ph[NLAYERS], *Wo_ph[NLAYERS], *W13_ph[NLAYERS], *W2_ph[NLAYERS];
    __strong MPSGraphTensor *rms_att_ph[NLAYERS], *rms_ffn_ph[NLAYERS];
    __strong MPSGraphTensor *x_out;
} MpsGraphTrunk;

#define MAX_MPS_GRAPHS 16
static MpsGraphTrunk g_mg[MAX_MPS_GRAPHS];
static int g_nmg = 0;

// matmul: y[oc,seq] = W[ic,oc]^T @ x[ic,seq]. W is [ic,oc], x is [ic,seq].
static MPSGraphTensor *mg_matmul(MPSGraph *g, MPSGraphTensor *W, MPSGraphTensor *x, NSString *nm) {
    MPSGraphTensor *Wt = [g transposeTensor:W dimension:0 withDimension:1 name:[nm stringByAppendingString:@"_Wt"]];
    return [g matrixMultiplicationWithPrimaryTensor:Wt secondaryTensor:x name:nm];
}

// RMSNorm: out = x * rsqrt(mean(x^2, axis=0) + eps) * weight. x: [DIM, S], w: [DIM].
static MPSGraphTensor *mg_rmsnorm(MPSGraph *g, MPSGraphTensor *x, MPSGraphTensor *w,
                                  int dim, int S, NSString *nm) {
    MPSGraphTensor *x2 = [g multiplicationWithPrimaryTensor:x secondaryTensor:x name:[nm stringByAppendingString:@"_x2"]];
    MPSGraphTensor *sum = [g reductionSumWithTensor:x2 axis:0 name:[nm stringByAppendingString:@"_sum"]];
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

// Causal MHA: Q/K/V [Q_DIM, S] -> attn_out [Q_DIM, S]. mask [SEQ,SEQ] is a graph constant.
static MPSGraphTensor *mg_attention(MPSGraph *g, MPSGraphTensor *Q, MPSGraphTensor *K,
                                    MPSGraphTensor *V, MPSGraphTensor *mask, int B, int S,
                                    NSString *nm) {
    MPSGraphTensor *Q4 = [g reshapeTensor:Q withShape:@[@(HEADS), @(HD), @(B), @(SEQ)] name:[nm stringByAppendingString:@"_Q4"]];
    MPSGraphTensor *Qb = [g transposeTensor:Q4 permutation:@[@2, @0, @3, @1] name:[nm stringByAppendingString:@"_Qb"]];
    MPSGraphTensor *K4 = [g reshapeTensor:K withShape:@[@(HEADS), @(HD), @(B), @(SEQ)] name:[nm stringByAppendingString:@"_K4"]];
    MPSGraphTensor *Kb = [g transposeTensor:K4 permutation:@[@2, @0, @3, @1] name:[nm stringByAppendingString:@"_Kb"]];
    MPSGraphTensor *V4 = [g reshapeTensor:V withShape:@[@(HEADS), @(HD), @(B), @(SEQ)] name:[nm stringByAppendingString:@"_V4"]];
    MPSGraphTensor *Vb = [g transposeTensor:V4 permutation:@[@2, @0, @3, @1] name:[nm stringByAppendingString:@"_Vb"]];
    MPSGraphTensor *Kt = [g transposeTensor:Kb dimension:2 withDimension:3 name:[nm stringByAppendingString:@"_Kt"]];
    MPSGraphTensor *sc = [g matrixMultiplicationWithPrimaryTensor:Qb secondaryTensor:Kt name:[nm stringByAppendingString:@"_sc"]];
    float scale = 1.0f/sqrtf((float)HD);
    MPSGraphTensor *sc_c = [g constantWithScalar:(double)scale dataType:MPSDataTypeFloat32];
    MPSGraphTensor *scs = [g multiplicationWithPrimaryTensor:sc secondaryTensor:sc_c name:[nm stringByAppendingString:@"_scs"]];
    MPSGraphTensor *mb = [g reshapeTensor:mask withShape:@[@1, @1, @(SEQ), @(SEQ)] name:[nm stringByAppendingString:@"_mb"]];
    MPSGraphTensor *msk = [g additionWithPrimaryTensor:scs secondaryTensor:mb name:[nm stringByAppendingString:@"_msk"]];
    MPSGraphTensor *pr = [g softMaxWithTensor:msk axis:3 name:[nm stringByAppendingString:@"_sm"]];
    MPSGraphTensor *at = [g matrixMultiplicationWithPrimaryTensor:pr secondaryTensor:Vb name:[nm stringByAppendingString:@"_at"]];
    MPSGraphTensor *at_t = [g transposeTensor:at permutation:@[@1, @3, @0, @2] name:[nm stringByAppendingString:@"_att"]];
    return [g reshapeTensor:at_t withShape:@[@(Q_DIM), @(S)] name:nm];
}

static void mg_build(MpsGraphTrunk *t, int S) {
    int B = S / SEQ;
    t->graph = [[MPSGraph alloc] init];
    t->S = S;
    MPSGraph *g = t->graph;
    t->x_in_ph = [g placeholderWithShape:@[@(DIM), @(S)] dataType:MPSDataTypeFloat32 name:@"x_in"];
    t->rms_final_ph = [g placeholderWithShape:@[@(DIM)] dataType:MPSDataTypeFloat32 name:@"rms_final"];
    // Causal mask [SEQ, SEQ]: 0 below diag, -1e9 above (baked as a graph constant).
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
    MPSGraphTensor *x = t->x_in_ph;
    for (int L = 0; L < NLAYERS; L++) {
        NSString *pfx = [NSString stringWithFormat:@"L%d_", L];
        MPSGraphTensor *xn = mg_rmsnorm(g, x, t->rms_att_ph[L], DIM, S, [pfx stringByAppendingString:@"rms_att"]);
        MPSGraphTensor *qkv = mg_matmul(g, t->Wqkv_ph[L], xn, [pfx stringByAppendingString:@"qkv"]);
        MPSGraphTensor *Q = [g sliceTensor:qkv dimension:0 start:0 length:Q_DIM name:[pfx stringByAppendingString:@"Q"]];
        MPSGraphTensor *K = [g sliceTensor:qkv dimension:0 start:Q_DIM length:KV_DIM name:[pfx stringByAppendingString:@"K"]];
        MPSGraphTensor *V = [g sliceTensor:qkv dimension:0 start:Q_DIM+KV_DIM length:KV_DIM name:[pfx stringByAppendingString:@"V"]];
        MPSGraphTensor *at = mg_attention(g, Q, K, V, mask, B, S, [pfx stringByAppendingString:@"attn"]);
        MPSGraphTensor *o = mg_matmul(g, t->Wo_ph[L], at, [pfx stringByAppendingString:@"wo"]);
        MPSGraphTensor *ao = [g multiplicationWithPrimaryTensor:o secondaryTensor:alpha_c name:[pfx stringByAppendingString:@"ao"]];
        MPSGraphTensor *x2 = [g additionWithPrimaryTensor:x secondaryTensor:ao name:[pfx stringByAppendingString:@"x2"]];
        MPSGraphTensor *x2n = mg_rmsnorm(g, x2, t->rms_ffn_ph[L], DIM, S, [pfx stringByAppendingString:@"rms_ffn"]);
        MPSGraphTensor *h13 = mg_matmul(g, t->W13_ph[L], x2n, [pfx stringByAppendingString:@"h13"]);
        MPSGraphTensor *h1 = [g sliceTensor:h13 dimension:0 start:0 length:HIDDEN name:[pfx stringByAppendingString:@"h1"]];
        MPSGraphTensor *h3 = [g sliceTensor:h13 dimension:0 start:HIDDEN length:HIDDEN name:[pfx stringByAppendingString:@"h3"]];
        MPSGraphTensor *sig = [g sigmoidWithTensor:h1 name:[pfx stringByAppendingString:@"sig"]];
        MPSGraphTensor *silu = [g multiplicationWithPrimaryTensor:h1 secondaryTensor:sig name:[pfx stringByAppendingString:@"silu"]];
        MPSGraphTensor *gate = [g multiplicationWithPrimaryTensor:silu secondaryTensor:h3 name:[pfx stringByAppendingString:@"gate"]];
        MPSGraphTensor *ffn = mg_matmul(g, t->W2_ph[L], gate, [pfx stringByAppendingString:@"w2"]);
        MPSGraphTensor *af = [g multiplicationWithPrimaryTensor:ffn secondaryTensor:alpha_c name:[pfx stringByAppendingString:@"af"]];
        x = [g additionWithPrimaryTensor:x2 secondaryTensor:af name:[pfx stringByAppendingString:@"x"]];
    }
    t->x_out = mg_rmsnorm(g, x, t->rms_final_ph, DIM, S, @"rms_final");
}

static MpsGraphTrunk *mg_get(int S) {
    for (int i = 0; i < g_nmg; i++)
        if (g_mg[i].S == S) return &g_mg[i];
    if (g_nmg >= MAX_MPS_GRAPHS) {
        fprintf(stderr, "[mps_graph] graph cache overflow (%d shapes)\n", g_nmg); abort();
    }
    mg_build(&g_mg[g_nmg], S);
    return &g_mg[g_nmg++];
}

// Execute the whole-trunk graph: x_in -> x_final (zero-copy, in-place into x_final).
// Eval-only (save_acts==0 callers). x_pre_final is NOT written — no save_acts==0
// caller reads it (verified: net_eval_batched and selfcheck's single-position reference
// both pass x_pre but only read x_final). Weights are the FUSED forward-only Wqkv/W13.
static void mps_graph_forward(CLayer *W, const float *x_in, int B,
                              float *x_final, const float *rms_final) {
    int S = B*SEQ;
    MpsGraphTrunk *t = mg_get(S);
    @autoreleasepool {
        NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
        int zc;
        id<MTLBuffer> bx = mps_get_buf(x_in, (size_t)DIM*S*4, &zc);
        feeds[t->x_in_ph] = [[MPSGraphTensorData alloc] initWithMTLBuffer:bx shape:@[@(DIM), @(S)] dataType:MPSDataTypeFloat32];
        id<MTLBuffer> brf = mps_get_buf(rms_final, (size_t)DIM*4, &zc);
        feeds[t->rms_final_ph] = [[MPSGraphTensorData alloc] initWithMTLBuffer:brf shape:@[@(DIM)] dataType:MPSDataTypeFloat32];
        for (int L = 0; L < NLAYERS; L++) {
            CLayer *w = &W[L];
            #define FEED2D(ph, ptr, r, c) do { \
                id<MTLBuffer> _b = mps_get_buf(ptr, (size_t)(r)*(c)*4, &zc); \
                feeds[t->ph[L]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:_b shape:@[@(r), @(c)] dataType:MPSDataTypeFloat32]; \
            } while(0)
            #define FEED1D(ph, ptr, n) do { \
                id<MTLBuffer> _b = mps_get_buf(ptr, (size_t)(n)*4, &zc); \
                feeds[t->ph[L]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:_b shape:@[@(n)] dataType:MPSDataTypeFloat32]; \
            } while(0)
            FEED2D(Wqkv_ph, w->Wqkv, DIM, Q_DIM+2*KV_DIM);
            FEED2D(Wo_ph,   w->Wo,   Q_DIM, DIM);
            FEED2D(W13_ph,  w->W13,  DIM, 2*HIDDEN);
            FEED2D(W2_ph,   w->W2,   HIDDEN, DIM);
            FEED1D(rms_att_ph, w->rms_att, DIM);
            FEED1D(rms_ffn_ph, w->rms_ffn, DIM);
            #undef FEED2D
            #undef FEED1D
        }
        id<MTLBuffer> bout = mps_get_buf(x_final, (size_t)DIM*S*4, &zc);
        MPSGraphTensorData *out_td = [[MPSGraphTensorData alloc] initWithMTLBuffer:bout shape:@[@(DIM), @(S)] dataType:MPSDataTypeFloat32];
        NSMutableDictionary *results = [NSMutableDictionary dictionary];
        results[t->x_out] = out_td;
        id<MTLCommandBuffer> cb = [g_mtl_queue commandBuffer];
        MPSCommandBuffer *mps_cb = [MPSCommandBuffer commandBufferWithCommandBuffer:cb];
        [t->graph encodeToCommandBuffer:mps_cb feeds:feeds targetOperations:nil resultsDictionary:results executionDescriptor:nil];
        [mps_cb commit]; [mps_cb waitUntilCompleted];
    }
}
#endif // LILBRO_HAS_MPS

// ============================================================================
// Trunk forward: x_in[DIM, B*SEQ] (embed+posenc already summed) -> x_final[DIM, B*SEQ].
// save_acts=1 -> acts is NLAYERS-long (kept for backward); =0 -> acts[0] reused (eval).
// B=1 is byte-identical to the pre-#18 chess_forward.
// ============================================================================
static void chess_trunk_forward(CLayer *W, CActs *acts, const float *x_in, int B,
                                float *x_pre_final, float *x_final, const float *rms_final,
                                float res_alpha, int save_acts) {
#if LILBRO_HAS_MPS
    // Phase 2: MPSGraph whole-forward — the entire trunk as one GPU graph (5.3-6.0x
    // vs ANE+CPU, probe_mps_graph). Eval-only (save_acts==0): the learner needs all
    // intermediates for the backward, so save_acts==1 falls through to the ANE+CPU
    // path. x_pre_final is NOT written on this path — no save_acts==0 caller reads it.
    if (g_use_mps_graph && !save_acts && g_mtl_dev) {
        mps_graph_forward(W, x_in, B, x_final, rms_final);
        return;
    }
#endif
    int S = B*SEQ;
    float *x = fmalloc((size_t)DIM*S); memcpy(x, x_in, (size_t)DIM*S*4);
    float *o = fmalloc((size_t)DIM*S), *ffn = fmalloc((size_t)DIM*S);
    for (int L = 0; L < NLAYERS; L++) {
        CActs *ac = save_acts ? &acts[L] : &acts[0];
        CLayer *w = &W[L];
        if (save_acts) memcpy(ac->layer_in, x, (size_t)DIM*S*4);   // only the backward reads layer_in [iter 7]
        { uint64_t _t = g_trunk_prof ? mach_absolute_time() : 0;
          chess_rmsnorm_par(ac->xnorm, x, w->rms_att, DIM, S);
          if (g_trunk_prof) g_trunk_rms_s += tb_ms(mach_absolute_time() - _t) * 1e-3; }
        // Fused QKV vs separate: ADAPTIVE. The fused matmul (3 evals -> 1) wins when the ANE
        // is dispatch-bound (small S, floor ~0.2ms/eval dominates); it is neutral-or-worse
        // when compute-bound (large S, B=160: same FLOPs, bigger oc surfaces => more io). The
        // probe puts the crossover near B~128 (S~12288). Both paths write the SAME ac->qkv
        // buffer (Q/K/V are contiguous row-slices), so the output is bit-identical either way.
        // [iter 6]
        if (S <= 12288) {
            ane_matmul(DIM, Q_DIM + 2*KV_DIM, S, ac->xnorm, w->Wqkv, ac->Q);
        } else {
            ane_matmul(DIM, Q_DIM,    S, ac->xnorm, w->Wq, ac->Q);
            ane_matmul(DIM, KV_DIM,   S, ac->xnorm, w->Wk, ac->K);
            ane_matmul(DIM, KV_DIM,   S, ac->xnorm, w->Wv, ac->V);
        }
        {
            uint64_t _ta = g_trunk_prof ? mach_absolute_time() : 0;
            attn_cpu_forward_batched(ac->attn, ac->Q, ac->K, ac->V, B, SEQ);
            if (g_trunk_prof) g_trunk_attn_s += tb_ms(mach_absolute_time() - _ta) * 1e-3;
        }
        ane_matmul(Q_DIM, DIM, S, ac->attn, w->Wo, o);
        { float ra = res_alpha; float *x2 = ac->x2, *xx = x, *oo = o; long N = (long)DIM*S;
          chess_parallel_for(N, ^(long lo, long hi){ for (long i = lo; i < hi; i++) x2[i] = xx[i] + ra*oo[i]; }); }
        { uint64_t _t = g_trunk_prof ? mach_absolute_time() : 0;
          chess_rmsnorm_par(ac->x2norm, ac->x2, w->rms_ffn, DIM, S);
          if (g_trunk_prof) g_trunk_rms_s += tb_ms(mach_absolute_time() - _t) * 1e-3; }
        // Fused W1/W3 vs separate: ADAPTIVE (same dispatch-bound vs compute-bound crossover
        // as QKV above, S <= 12288). Both paths write the SAME ac->h13 buffer. [iter 6]
        if (S <= 12288) {
            ane_matmul(DIM, 2*HIDDEN, S, ac->x2norm, w->W13, ac->h1);
        } else {
            ane_matmul(DIM, HIDDEN, S, ac->x2norm, w->W1, ac->h1);
            ane_matmul(DIM, HIDDEN, S, ac->x2norm, w->W3, ac->h3);
        }
        { float *h1 = ac->h1, *h3 = ac->h3, *gate = ac->gate; long N = (long)HIDDEN*S;
          uint64_t _t = g_trunk_prof ? mach_absolute_time() : 0;
          chess_parallel_for(N, ^(long lo, long hi){
            for (long i = lo; i < hi; i++) { float sig = 1.0f/(1.0f+expf(-h1[i])); gate[i] = (h1[i]*sig)*h3[i]; }
          });
          if (g_trunk_prof) g_trunk_silu_s += tb_ms(mach_absolute_time() - _t) * 1e-3; }
        ane_matmul(HIDDEN, DIM, S, ac->gate, w->W2, ffn);
        { float ra = res_alpha; float *xx = x, *x2 = ac->x2, *ff = ffn; long N = (long)DIM*S;
          chess_parallel_for(N, ^(long lo, long hi){ for (long i = lo; i < hi; i++) xx[i] = x2[i] + ra*ff[i]; }); }
    }
    if (save_acts) memcpy(x_pre_final, x, (size_t)DIM*S*4);   // only the backward reads x_pre_final [iter 7]
    { uint64_t _t = g_trunk_prof ? mach_absolute_time() : 0;
      chess_rmsnorm_par(x_final, x, rms_final, DIM, S);
      if (g_trunk_prof) g_trunk_rms_s += tb_ms(mach_absolute_time() - _t) * 1e-3; }
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
//
// FORWARD is parallelized over B (each b writes disjoint x_in columns, reads read-only
// embeddings) => bit-identical, checksum preserved. BACKWARD stays serial: it accumulates
// into shared d_tok/d_rank/d_file/d_misc (many positions share a token), so racing those
// across cores would need atomics. Backward is learner-only, not the bench hot path. [iter 3]
// ============================================================================
static void chess_embed_posenc_batched(float *x_in, int B, const uint16_t *tokens,
                                       const float *tok_emb, const float *rank_emb,
                                       const float *file_emb, const float *misc_emb) {
    int S = B*SEQ;
    dispatch_queue_t dq = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
    dispatch_apply(B, dq, ^(size_t bb) {
        int b = (int)bb;
        const uint16_t *tk = tokens + b*SEQ;
        for (int t = 0; t < SEQ; t++) {
            int tok = tk[t], col = b*SEQ + t;
            for (int d = 0; d < DIM; d++) x_in[d*S+col] = tok_emb[tok*DIM+d];
            if (t < NBOARD) { int rk = t>>3, fl = t&7;
                for (int d = 0; d < DIM; d++) x_in[d*S+col] += rank_emb[rk*DIM+d] + file_emb[fl*DIM+d]; }
            else { int mi = t - NBOARD;
                for (int d = 0; d < DIM; d++) x_in[d*S+col] += misc_emb[mi*DIM+d]; }
        }
    });
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

// Build the FUSED forward-only weights from the canonical ones: Wqkv[DIM, Q_DIM+2*KV_DIM] =
// [Wq | Wk | Wv] concatenated along the output-channel axis; W13[DIM, 2*HIDDEN] = [W1 | W3].
// The fused matmul produces [Q; K; V] (resp. [h1; h3]) in ONE ANE eval, so Q/K/V (h1/h3) are
// contiguous row-slices of the output -- exactly how CActs.qkv (h13) is laid out. Call once
// after init/load (bench) and after every optimizer step (training); free with _free_fused.
// Forward-only: the canonical weights remain the checkpoint/optimizer/backward source of
// truth. [iter 6]
static void chess_layer_build_fused(CLayer *w) {
    int qkv_oc = Q_DIM + 2*KV_DIM;
    if (!w->Wqkv) w->Wqkv = fmalloc((size_t)DIM*qkv_oc);
    for (int i = 0; i < DIM; i++) {
        memcpy(w->Wqkv + (size_t)i*qkv_oc,                 w->Wq + (size_t)i*Q_DIM,    (size_t)Q_DIM*4);
        memcpy(w->Wqkv + (size_t)i*qkv_oc + Q_DIM,         w->Wk + (size_t)i*KV_DIM,   (size_t)KV_DIM*4);
        memcpy(w->Wqkv + (size_t)i*qkv_oc + Q_DIM+KV_DIM,  w->Wv + (size_t)i*KV_DIM,   (size_t)KV_DIM*4);
    }
    int w13_oc = 2*HIDDEN;
    if (!w->W13) w->W13 = fmalloc((size_t)DIM*w13_oc);
    for (int i = 0; i < DIM; i++) {
        memcpy(w->W13 + (size_t)i*w13_oc,           w->W1 + (size_t)i*HIDDEN, (size_t)HIDDEN*4);
        memcpy(w->W13 + (size_t)i*w13_oc + HIDDEN,  w->W3 + (size_t)i*HIDDEN, (size_t)HIDDEN*4);
    }
}
static void chess_net_build_fused(ChessNet *n) {
    for (int L = 0; L < NLAYERS; L++) chess_layer_build_fused(&n->W[L]);
}
static void chess_net_free_fused(ChessNet *n) {
    for (int L = 0; L < NLAYERS; L++) {
        if (n->W[L].Wqkv) { free(n->W[L].Wqkv); n->W[L].Wqkv = NULL; }
        if (n->W[L].W13)  { free(n->W[L].W13);  n->W[L].W13  = NULL; }
    }
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
