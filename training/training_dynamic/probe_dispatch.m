// probe_dispatch.m — Is the ~51ms ane_fwd+ane_bwd dominated by per-eval ANE
// DISPATCH overhead (→ kernel fusion is the lever) or by COMPUTE (→ hard floor)?
// (1) Sweeps square matmul kernels at fixed seq=256: per-eval time vs FLOPs.
//     Flat low end → dispatch-bound; linear-in-FLOPs → compute-bound.
// (2) Fusion PoC: 2 matmuls in ONE eval vs 1 matmul/eval. If 2-for-1 ≈ cost of
//     1, fusing halves per-matmul dispatch cost — the lever, demonstrated.
// Standalone; LOCAL non-aborting eval so it can probe sizes the ANE may reject.
#include "mil_dynamic.h"  // pulls io.h + config.h

static BOOL try_eval(Kern *k) {
    id mdl = (__bridge id)k->model; id req = (__bridge id)k->request; NSError *e = nil;
    return ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
}
static double time_kern(Kern *k, int iters) {
    for (int i = 0; i < 20; i++) if (!try_eval(k)) return -1;   // warmup + validity
    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++) try_eval(k);
    return tb_ms(mach_absolute_time() - t0) / iters;
}
static double per_eval_ms(int n, int seq, int iters) {
    @autoreleasepool {
        Kern *k = compile_kern_mil_w(gen_dyn_matmul_mil(n, n, seq), @{},
                                     n*(seq+n)*2, n*seq*2);
        if (!k) return -2;
        float *inbuf = (float*)calloc((size_t)n*(seq+n), 4);
        io_write_fp16(k->ioIn, inbuf, n, seq+n);
        double ms = time_kern(k, iters);
        free(inbuf); free_kern(k);
        return ms;
    }
}
// One kernel, TWO matmuls (act@W1 + act@W2): input [1,n,1,seq+n+n], one output.
static NSString *gen_fused2_mil(int n, int seq) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", n, seq + n + n];
    gen_dyn_matmul(m, "f1", n, n, seq, 0, seq,     "x");   // W1 at spatial offset seq
    // gen_dyn_matmul hardcodes one un-prefixed const "bF"; a 2nd instance would
    // duplicate that name (InvalidMILProgram). Rename it in the f2 block.
    NSMutableString *f2 = [NSMutableString string];
    gen_dyn_matmul(f2, "f2", n, n, seq, 0, seq + n, "x");  // W2 at offset seq+n
    [f2 replaceOccurrencesOfString:@"bF" withString:@"bFb" options:0 range:NSMakeRange(0, f2.length)];
    [m appendString:f2];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> fy = add(x=f1_y, y=f2_y)[name=string(\"fy\")];\n", n, seq];
    [m appendString:@"    } -> (fy);\n}\n"];
    return m;
}

int main(void) {
    @autoreleasepool {
        ane_init();
        mach_timebase_info(&g_tb);
        int N = 400, seq = 256;
        int sizes[] = {32, 64, 128, 256, 512, 768, 1024, 2048};
        printf("\n[1] per-eval ms, square ic=oc=n @ seq=%d (%d iters), FLOPs ~ 2*n*n*seq:\n", seq, N);
        for (int i = 0; i < (int)(sizeof(sizes)/sizeof(int)); i++) {
            int n = sizes[i]; double ms = per_eval_ms(n, seq, N);
            if (ms < 0) { printf("  n=%-4d  REJECTED\n", n); continue; }
            printf("  n=%-4d  %.4f ms   (FLOPs %.3fx of 768)\n", n, ms, (double)n*n/(768.0*768.0));
        }
        // [2] Fusion PoC at n=768.
        int n = 768;
        double single = per_eval_ms(n, seq, N);
        double fused = -1;
        @autoreleasepool {
            Kern *kf = compile_kern_mil_w(gen_fused2_mil(n, seq), @{},
                                          n*(seq+n+n)*2, n*seq*2);
            if (kf) {
                float *inbuf = (float*)calloc((size_t)n*(seq+n+n), 4);
                io_write_fp16(kf->ioIn, inbuf, n, seq+n+n);
                fused = time_kern(kf, N);
                free(inbuf); free_kern(kf);
            }
        }
        printf("\n[2] FUSION PoC (n=768, seq=256):\n");
        printf("  1 matmul / eval          = %.4f ms\n", single);
        if (fused < 0) { printf("  2 matmuls / eval (fused) = REJECTED\n"); }
        else {
            printf("  2 matmuls / eval (fused) = %.4f ms\n", fused);
            printf("  → 2 separate evals would be ~%.4f ms; fused saves ~%.4f ms (%.0f%%)\n",
                   2*single, 2*single - fused, 100.0*(2*single - fused)/(2*single));
        }
        printf("\nVerdict: flat sweep + fused≈single ⇒ dispatch-bound ⇒ fewer evals (fusion) is the lever.\n");
    }
    return 0;
}
