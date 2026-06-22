// probe_mps.m — Tracer bullet: does MPS/Metal beat the ANE dispatch floor for
// the chess trunk's matmuls? The speedup doc (results/chess_speedup.md iter 8)
// names the ANE matmul path (ane=23% + io=16% = 40% of gen wall) as the single
// remaining big lever, and hypothesizes MPS/Metal "could approach 100x" via
// zero-copy MTLBuffers + fused command buffers. That is a HYPOTHESIS — this
// probe measures it before we commit to a rewrite.
//
// The chess trunk forward is 8 ane_matmul evals (fused QKV + Wo + fused W1W3 +
// W2, ×2 layers). Each ane_matmul pays: fp32->fp16 convert + IOSurfaceLock +
// ~0.2-0.4ms ANE dispatch floor + IOSurfaceLock + fp16->fp32 convert. MPS on
// Apple Silicon unified memory can skip ALL of that: newBufferWithBytesNoCopy
// wraps an existing float* in a MTLBuffer with zero copy, MPSDataTypeFloat32
// needs no conversion, and N matmuls can encode into ONE command buffer that
// dispatches once.
//
// What this measures, at the real shapes (DIM=256 HIDDEN=512 SEQ=96, B=64/96/160):
//   (1) CORRECTNESS: MPS matmul vs cblas_sgemm — max abs err, mean abs err.
//   (2) SINGLE-EVAL LATENCY: one MPS matmul (cold + warm) vs one ANE matmul.
//       This isolates the dispatch floor, the doc's #1 lever.
//   (3) ZERO-COPY vs MANAGED: newBufferWithBytesNoCopy (no copy) vs
//       newBufferWithLength + memcpy. Quantifies the io (16%) bucket.
//   (4) FUSED COMMAND BUFFER: 8 matmuls (one layer pair) in ONE cmd buffer vs
//       8 separate cmd buffers. Quantifies the fusion win (the 100x hypothesis).
//
// If (1) passes (err < 1e-4) AND (2)+(4) beat ANE, integrate as mps_matmul
// replacing ane_matmul at the same seam. If MPS dispatch is ALSO slow, attack
// a different angle (the doc names MPSGraph whole-forward fusion as the
// frontier; this probe uses MPSMatrixMultiplication to isolate the matmul).
//
// Build: see Makefile target probe_mps (Metal + MPS + MPSMatrix frameworks).
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <Accelerate/Accelerate.h>
#include <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Chess net dims (must match models/chess_g0.h / config.h for the real shapes).
#define DIM     256
#define HIDDEN  512
#define HEADS    8
#define HD      32
#define Q_DIM  (HEADS*HD)      // 256 (MHA)
#define KV_DIM (HEADS*HD)      // 256
#define SEQ     96
#define NLAYERS 2

static double tb_ms(uint64_t t) {
    static double s = -1; if (s < 0) { mach_timebase_info_data_t b; mach_timebase_info(&b);
        s = (double)b.numer / b.denom / 1e6; }
    return (double)t * s;
}

// The 4 distinct matmul shapes per layer (fused forward, 8 evals/forward = 4×2):
//   qkv : DIM -> Q_DIM+2*KV_DIM = 256 -> 768
//   wo  : Q_DIM -> DIM           = 256 -> 256
//   w13 : DIM -> 2*HIDDEN        = 256 -> 1024
//   w2  : HIDDEN -> DIM          = 512 -> 256
// ane_matmul computes y[oc,seq] = W[ic,oc]^T @ x[ic,seq].
typedef struct { const char *name; int ic, oc; } MMShape;
static const MMShape SHAPES[4] = {
    {"qkv", DIM, Q_DIM + 2*KV_DIM},
    {"wo",  Q_DIM, DIM},
    {"w13", DIM, 2*HIDDEN},
    {"w2",  HIDDEN, DIM},
};

// Fill with bounded random data (deterministic seed).
static void fill_randn(float *p, size_t n, unsigned *seed) {
    for (size_t i = 0; i < n; i++) {
        // cheap Box-Muller-ish: sum of 12 uniforms ~ N(0,1)
        float s = 0;
        for (int k = 0; k < 12; k++) { *seed = *seed * 1103515245u + 12345u; s += (float)(*seed >> 8) / 16777216.0f; }
        p[i] = (s - 6.0f) * 0.1f;   // small values, typical of post-norm activations
    }
}

// Reference: y = W^T @ x via cblas_sgemm. W[ic,oc], x[ic,seq], y[oc,seq].
static void ref_matmul(const float *W, const float *x, float *y, int ic, int oc, int seq) {
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                oc, seq, ic, 1.0f, W, oc, x, seq, 0.0f, y, seq);
}

// ---- MPS matmul harness ----
// One shape's persistent state: the MPS kernel + reusable MTLBuffers (sized for
// the largest seq we'll run). zero_copy=1 uses newBufferWithBytesNoCopy on the
// caller's float*; zero_copy=0 uses a managed buffer (newBufferWithLength).
typedef struct {
    MPSMatrixMultiplication *kern;
    MPSMatrix *A, *B, *C;       // W, x, y
    id<MTLBuffer> bufA, bufB, bufC;
    int ic, oc, seq;
    int zero_copy;
} MpsMM;

static void mpsmm_free(MpsMM *m) {
    if (!m->kern) return;
    m->kern = nil;
    m->A = m->B = m->C = nil;
    m->bufA = m->bufB = m->bufC = nil;
}

// Create (or resize) the MPS state for shape (ic,oc) at seq S. If a matching
// state already exists, reuses it (the kernel is shape-fixed at init; buffers
// can be rebound). Returns 0 on success.
static int mpsmm_setup(MpsMM *m, id<MTLDevice> dev, id<MTLCommandQueue> q,
                       int ic, int oc, int seq, int zero_copy,
                       float *W_host, float *x_host, float *y_host) {
    @autoreleasepool {
        // (Re)create kernel if shape changed.
        if (m->kern == nil || m->ic != ic || m->oc != oc) {
            mpsmm_free(m);
            // C = alpha * A^T @ B. A=W[ic,oc], B=x[ic,seq], C=y[oc,seq].
            // M=oc (result rows), N=seq (result cols), K=ic (interior).
            m->kern = [[MPSMatrixMultiplication alloc] initWithDevice:dev
                                                         transposeLeft:YES transposeRight:NO
                                                         resultRows:oc resultColumns:seq interiorColumns:ic
                                                         alpha:1.0f beta:0.0f];
            if (!m->kern) { fprintf(stderr, "  [mps] MPSMatrixMultiplication init FAILED ic=%d oc=%d seq=%d\n", ic, oc, seq); return -1; }
            m->ic = ic; m->oc = oc;
        }
        m->seq = seq;
        m->zero_copy = zero_copy;

        // (Re)allocate buffers if seq grew. W is [ic,oc], x is [ic,seq], y is [oc,seq].
        size_t bytesA = (size_t)ic * oc * 4;
        size_t bytesB = (size_t)ic * seq * 4;
        size_t bytesC = (size_t)oc * seq * 4;
        if (m->bufA == nil || m->bufA.length < bytesA) {
            m->bufA = nil;
            if (zero_copy) m->bufA = [dev newBufferWithBytesNoCopy:W_host length:bytesA options:MTLResourceStorageModeShared deallocator:nil];
            else           m->bufA = [dev newBufferWithLength:bytesA options:MTLResourceStorageModeShared];
            if (!m->bufA) { fprintf(stderr, "  [mps] bufA alloc FAILED\n"); return -1; }
        }
        if (m->bufB == nil || m->bufB.length < bytesB) {
            m->bufB = nil;
            if (zero_copy) m->bufB = [dev newBufferWithBytesNoCopy:x_host length:bytesB options:MTLResourceStorageModeShared deallocator:nil];
            else           m->bufB = [dev newBufferWithLength:bytesB options:MTLResourceStorageModeShared];
            if (!m->bufB) { fprintf(stderr, "  [mps] bufB alloc FAILED\n"); return -1; }
        }
        if (m->bufC == nil || m->bufC.length < bytesC) {
            m->bufC = nil;
            if (zero_copy) m->bufC = [dev newBufferWithBytesNoCopy:y_host length:bytesC options:MTLResourceStorageModeShared deallocator:nil];
            else           m->bufC = [dev newBufferWithLength:bytesC options:MTLResourceStorageModeShared];
            if (!m->bufC) { fprintf(stderr, "  [mps] bufC alloc FAILED\n"); return -1; }
        }

        // Managed path: copy host data in. Zero-copy: the buffer IS the host memory.
        if (!zero_copy) {
            memcpy(m->bufA.contents, W_host, bytesA);
            memcpy(m->bufB.contents, x_host, bytesB);
        }

        // MPSMatrix descriptors. rowBytes = cols*4 (fp32, row-major, packed).
        MPSMatrixDescriptor *dA = [MPSMatrixDescriptor matrixDescriptorWithRows:ic columns:oc
                                                    rowBytes:oc*4 dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *dB = [MPSMatrixDescriptor matrixDescriptorWithRows:ic columns:seq
                                                    rowBytes:seq*4 dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *dC = [MPSMatrixDescriptor matrixDescriptorWithRows:oc columns:seq
                                                    rowBytes:seq*4 dataType:MPSDataTypeFloat32];
        m->A = [[MPSMatrix alloc] initWithBuffer:m->bufA descriptor:dA];
        m->B = [[MPSMatrix alloc] initWithBuffer:m->bufB descriptor:dB];
        m->C = [[MPSMatrix alloc] initWithBuffer:m->bufC descriptor:dC];
        if (!m->A || !m->B || !m->C) { fprintf(stderr, "  [mps] MPSMatrix init FAILED\n"); return -1; }
        return 0;
    }
}

// Encode one matmul into the given command buffer (no commit/wait — caller controls).
static void mpsmm_encode(MpsMM *m, id<MTLCommandBuffer> cb) {
    [m->kern encodeToCommandBuffer:cb leftMatrix:m->A rightMatrix:m->B resultMatrix:m->C];
}

// ---- Measurements ----

// (1) Correctness: MPS vs cblas_sgemm at one shape. Returns max abs err.
static double measure_correctness(id<MTLDevice> dev, id<MTLCommandQueue> q,
                                  const MMShape *sh, int seq, unsigned seed) {
    int ic = sh->ic, oc = sh->oc;
    size_t nW = (size_t)ic*oc, nx = (size_t)ic*seq, ny = (size_t)oc*seq;
    float *W = (float*)malloc(nW*4), *x = (float*)malloc(nx*4);
    float *y_mps = (float*)malloc(ny*4), *y_ref = (float*)malloc(ny*4);
    fill_randn(W, nW, &seed); fill_randn(x, nx, &seed);
    ref_matmul(W, x, y_ref, ic, oc, seq);

    // Zero-copy: y_mps's buffer IS the MTLBuffer, so after commit+wait the host
    // pointer is written. We must align the malloc to page size for NoCopy.
    // For correctness, use the managed (copy) path to avoid alignment edge cases.
    MpsMM m = {0};
    if (mpsmm_setup(&m, dev, q, ic, oc, seq, /*zero_copy=*/0, W, x, y_mps) != 0) {
        free(W); free(x); free(y_mps); free(y_ref); return -1.0;
    }
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        mpsmm_encode(&m, cb);
        [cb commit]; [cb waitUntilCompleted];
        // Managed output: copy back.
        memcpy(y_mps, m.bufC.contents, ny*4);
    }
    double maxerr = 0, sumerr = 0;
    for (size_t i = 0; i < ny; i++) {
        double e = fabs((double)y_mps[i] - y_ref[i]);
        if (e > maxerr) maxerr = e;
        sumerr += e;
    }
    mpsmm_free(&m);
    free(W); free(x); free(y_mps); free(y_ref);
    return maxerr;
}

// (2) Single-eval latency: one matmul per command buffer, warm + timed.
// Returns mean ms over `iters` warm runs, or -1 on failure.
static double measure_single_latency(id<MTLDevice> dev, id<MTLCommandQueue> q,
                                     const MMShape *sh, int seq, int iters,
                                     int zero_copy, unsigned seed) {
    int ic = sh->ic, oc = sh->oc;
    size_t nW = (size_t)ic*oc, nx = (size_t)ic*seq, ny = (size_t)oc*seq;
    float *W = (float*)malloc(nW*4), *x = (float*)malloc(nx*4), *y = (float*)malloc(ny*4);
    fill_randn(W, nW, &seed); fill_randn(x, nx, &seed);

    // For zero-copy, align allocations to page size (required by newBufferWithBytesNoCopy).
    MpsMM m = {0};
    if (zero_copy) {
        // Re-allocate page-aligned (newBufferWithBytesNoCopy requires page-aligned
        // address AND length on some configs; use posix_memalign).
        free(W); free(x); free(y);
        size_t pg = 4096;
        posix_memalign((void**)&W, pg, nW*4);
        posix_memalign((void**)&x, pg, nx*4);
        posix_memalign((void**)&y, pg, ny*4);
        fill_randn(W, nW, &seed); fill_randn(x, nx, &seed);
    }
    if (mpsmm_setup(&m, dev, q, ic, oc, seq, zero_copy, W, x, y) != 0) {
        free(W); free(x); free(y); return -1.0;
    }
    @autoreleasepool {
        // Warmup (first few evals compile/allocate caches).
        for (int i = 0; i < 20; i++) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            mpsmm_encode(&m, cb);
            [cb commit]; [cb waitUntilCompleted];
        }
        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < iters; i++) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            mpsmm_encode(&m, cb);
            [cb commit]; [cb waitUntilCompleted];
        }
        double ms = tb_ms(mach_absolute_time() - t0) / iters;
        mpsmm_free(&m);
        free(W); free(x); free(y);
        return ms;
    }
}

// (4) Fused command buffer: 8 matmuls (one layer pair = 4 shapes × 2 layers)
// encoded into ONE command buffer, one commit/wait. vs 8 separate cmd buffers.
// Returns mean ms over `iters` warm runs.
static double measure_fused_8(id<MTLDevice> dev, id<MTLCommandQueue> q,
                              int seq, int iters, int zero_copy, int fused,
                              unsigned seed) {
    // Allocate persistent state for all 4 shapes (×2 layers reuse the same
    // kernels; weights differ per layer but for timing we use the same buffers).
    MpsMM mms[4] = {{0}};
    float *Wbuf[4] = {0}, *xbuf[4] = {0}, *ybuf[4] = {0};
    size_t pg = 4096;
    for (int s = 0; s < 4; s++) {
        int ic = SHAPES[s].ic, oc = SHAPES[s].oc;
        size_t nW = (size_t)ic*oc, nx = (size_t)ic*seq, ny = (size_t)oc*seq;
        if (zero_copy) {
            posix_memalign((void**)&Wbuf[s], pg, nW*4);
            posix_memalign((void**)&xbuf[s], pg, nx*4);
            posix_memalign((void**)&ybuf[s], pg, ny*4);
        } else {
            Wbuf[s] = (float*)malloc(nW*4);
            xbuf[s] = (float*)malloc(nx*4);
            ybuf[s] = (float*)malloc(ny*4);
        }
        fill_randn(Wbuf[s], nW, &seed);
        fill_randn(xbuf[s], nx, &seed);
        if (mpsmm_setup(&mms[s], dev, q, ic, oc, seq, zero_copy, Wbuf[s], xbuf[s], ybuf[s]) != 0) {
            for (int j = 0; j <= s; j++) { free(Wbuf[j]); free(xbuf[j]); free(ybuf[j]); }
            return -1.0;
        }
    }
    @autoreleasepool {
        // Warmup.
        for (int w = 0; w < 10; w++) {
            if (fused) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                for (int L = 0; L < NLAYERS; L++)
                    for (int s = 0; s < 4; s++) mpsmm_encode(&mms[s], cb);
                [cb commit]; [cb waitUntilCompleted];
            } else {
                for (int L = 0; L < NLAYERS; L++)
                    for (int s = 0; s < 4; s++) {
                        id<MTLCommandBuffer> cb = [q commandBuffer];
                        mpsmm_encode(&mms[s], cb);
                        [cb commit]; [cb waitUntilCompleted];
                    }
            }
        }
        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < iters; i++) {
            if (fused) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                for (int L = 0; L < NLAYERS; L++)
                    for (int s = 0; s < 4; s++) mpsmm_encode(&mms[s], cb);
                [cb commit]; [cb waitUntilCompleted];
            } else {
                for (int L = 0; L < NLAYERS; L++)
                    for (int s = 0; s < 4; s++) {
                        id<MTLCommandBuffer> cb = [q commandBuffer];
                        mpsmm_encode(&mms[s], cb);
                        [cb commit]; [cb waitUntilCompleted];
                    }
            }
        }
        double ms = tb_ms(mach_absolute_time() - t0) / iters;
        for (int s = 0; s < 4; s++) mpsmm_free(&mms[s]);
        for (int s = 0; s < 4; s++) { free(Wbuf[s]); free(xbuf[s]); free(ybuf[s]); }
        return ms;
    }
}

int main(int argc, char **argv) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "[probe_mps] no Metal device\n"); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        if (!q) { fprintf(stderr, "[probe_mps] no command queue\n"); return 1; }

        printf("# probe_mps — MPS/Metal matmul vs ANE dispatch floor\n");
        printf("# device: %s\n", [[dev name] UTF8String]);
        printf("# shapes: DIM=%d HIDDEN=%d SEQ=%d NLAYERS=%d (8 matmuls/forward)\n",
               DIM, HIDDEN, SEQ, NLAYERS);
        printf("# matmul: y[oc,seq] = W[ic,oc]^T @ x[ic,seq]  (fp32, matches ane_matmul)\n\n");

        int Bs[] = {64, 96, 160};
        int nB = (int)(sizeof(Bs)/sizeof(Bs[0]));
        unsigned seed = 42;

        // ---- (1) CORRECTNESS ----
        printf("## (1) Correctness: MPS vs cblas_sgemm (max abs err, fp32)\n");
        printf("# shape      ic   oc    seq   max_err      verdict\n");
        int B = 64; int seq = B * SEQ;
        for (int s = 0; s < 4; s++) {
            double e = measure_correctness(dev, q, &SHAPES[s], seq, seed);
            const char *v = (e >= 0 && e < 1e-4) ? "PASS" : (e >= 0 ? "FAIL" : "ERR");
            printf("  %-8s  %4d %4d  %5d  %.3e   %s\n", SHAPES[s].name, SHAPES[s].ic, SHAPES[s].oc, seq, e, v);
        }
        printf("\n");

        // ---- (2) SINGLE-EVAL LATENCY ----
        printf("## (2) Single-eval latency: 1 matmul / cmd buffer (mean ms over 200 iters)\n");
        printf("# B    seq    shape   managed_ms  zerocopy_ms  zc_speedup\n");
        for (int bi = 0; bi < nB; bi++) {
            B = Bs[bi]; seq = B * SEQ;
            for (int s = 0; s < 4; s++) {
                double ms_man = measure_single_latency(dev, q, &SHAPES[s], seq, 200, 0, seed);
                double ms_zc  = measure_single_latency(dev, q, &SHAPES[s], seq, 200, 1, seed);
                double sp = (ms_zc > 0 && ms_man > 0) ? ms_man / ms_zc : 0;
                printf("  %3d  %5d  %-8s  %8.4f    %8.4f    %.2fx\n",
                       B, seq, SHAPES[s].name, ms_man, ms_zc, sp);
            }
        }
        printf("\n");

        // ---- (4) FUSED COMMAND BUFFER ----
        printf("## (4) Fused command buffer: 8 matmuls (1 layer pair) — 1 cmd buf vs 8\n");
        printf("# B    seq    path             ms/forward   matmul_ms   vs_separate\n");
        for (int bi = 0; bi < nB; bi++) {
            B = Bs[bi]; seq = B * SEQ;
            // managed, separate
            double ms_man_sep = measure_fused_8(dev, q, seq, 100, 0, 0, seed);
            // managed, fused
            double ms_man_fus = measure_fused_8(dev, q, seq, 100, 0, 1, seed);
            // zerocopy, separate
            double ms_zc_sep  = measure_fused_8(dev, q, seq, 100, 1, 0, seed);
            // zerocopy, fused (the hypothesized winner)
            double ms_zc_fus  = measure_fused_8(dev, q, seq, 100, 1, 1, seed);
            printf("  %3d  %5d  man,separate      %8.4f     %8.4f     1.00x\n",  B, seq, ms_man_sep, ms_man_sep/8);
            printf("  %3d  %5d  man,fused         %8.4f     %8.4f     %.2fx\n",  B, seq, ms_man_fus, ms_man_fus/8, ms_man_sep/ms_man_fus);
            printf("  %3d  %5d  zc, separate      %8.4f     %8.4f     %.2fx\n",  B, seq, ms_zc_sep,  ms_zc_sep/8,  ms_man_sep/ms_zc_sep);
            printf("  %3d  %5d  zc, fused         %8.4f     %8.4f     %.2fx\n",  B, seq, ms_zc_fus,  ms_zc_fus/8,  ms_man_sep/ms_zc_fus);
        }
        printf("\n");

        // ---- Verdict ----
        printf("## Verdict (compare to ANE: 8 evals/forward, ~0.2-0.4ms dispatch floor each = 1.6-3.2ms + io)\n");
        printf("# Read the zc,fused ms/forward column above. If it's < the ANE forward's ane+io time\n");
        printf("# (~6.4s wall * 40%% = ~2.5ms/forward at B=64 from the speedup doc), MPS wins. Then\n");
        printf("# integrate as mps_matmul at the ane_matmul seam (chess_net.h:101).\n");
        return 0;
    }
}
