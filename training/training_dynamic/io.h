// io.h — IOSurface helpers, NEON conversion, kernel compile/eval
// Updated for GQA (Qwen3-0.6B): Q_DIM != DIM, separate KV heads
#pragma once
#include "config.h"

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

// Blob builders for const weights (mask, rms)
static NSData *build_blob(const float *w, int rows, int cols) {
    int ws=rows*cols*2, tot=128+ws;
    uint8_t *b=(uint8_t*)calloc(tot,1);
    b[0]=1;b[4]=2;b[64]=0xEF;b[65]=0xBE;b[66]=0xAD;b[67]=0xDE;b[68]=1;
    *(uint32_t*)(b+72)=ws;*(uint32_t*)(b+80)=128;
    _Float16 *fp16=(_Float16*)(b+128);
    for(int i=0;i<rows*cols;i++) fp16[i]=(_Float16)w[i];
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}
static NSData *build_blob_fp16(_Float16 *d, int cnt) {
    int ws=cnt*2, tot=128+ws;
    uint8_t *b=(uint8_t*)calloc(tot,1);
    b[0]=1;b[4]=2;b[64]=0xEF;b[65]=0xBE;b[66]=0xAD;b[67]=0xDE;b[68]=1;
    *(uint32_t*)(b+72)=ws;*(uint32_t*)(b+80)=128;
    memcpy(b+128,d,ws);
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}

// NEON vectorized conversion
static void cvt_f16_f32(float *dst, const _Float16 *src, int n) {
    int i = 0;
    for (; i+7 < n; i += 8) {
        float16x8_t h = vld1q_f16((const __fp16*)(src+i));
        vst1q_f32(dst+i,   vcvt_f32_f16(vget_low_f16(h)));
        vst1q_f32(dst+i+4, vcvt_f32_f16(vget_high_f16(h)));
    }
    for (; i < n; i++) dst[i] = (float)src[i];
}
static void cvt_f32_f16(_Float16 *dst, const float *src, int n) {
    int i = 0;
    for (; i+7 < n; i += 8) {
        float16x8_t h = vcombine_f16(vcvt_f16_f32(vld1q_f32(src+i)),
                                      vcvt_f16_f32(vld1q_f32(src+i+4)));
        vst1q_f16((__fp16*)(dst+i), h);
    }
    for (; i < n; i++) dst[i] = (_Float16)src[i];
}

// IOSurface I/O (channel-first [C,S] layout, fp16 on surface)
static void io_write_fp16(IOSurfaceRef s, const float *data, int channels, int sp) {
    IOSurfaceLock(s, 0, NULL);
    cvt_f32_f16((_Float16*)IOSurfaceGetBaseAddress(s), data, channels * sp);
    IOSurfaceUnlock(s, 0, NULL);
}
static void io_read_fp16(IOSurfaceRef s, float *data, int ch_off, int channels, int sp) {
    IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
    cvt_f16_f32(data, (_Float16*)IOSurfaceGetBaseAddress(s) + ch_off * sp, channels * sp);
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
}
static void io_copy(IOSurfaceRef dst, int dst_ch, IOSurfaceRef src, int src_ch, int channels, int sp) {
    IOSurfaceLock(dst, 0, NULL);
    IOSurfaceLock(src, kIOSurfaceLockReadOnly, NULL);
    memcpy((_Float16*)IOSurfaceGetBaseAddress(dst) + dst_ch*sp,
           (_Float16*)IOSurfaceGetBaseAddress(src) + src_ch*sp,
           channels * sp * sizeof(_Float16));
    IOSurfaceUnlock(src, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceUnlock(dst, 0, NULL);
}
static void io_write_fp16_at(IOSurfaceRef s, int ch_off, const float *data, int channels, int sp) {
    IOSurfaceLock(s, 0, NULL);
    cvt_f32_f16((_Float16*)IOSurfaceGetBaseAddress(s) + ch_off * sp, data, channels * sp);
    IOSurfaceUnlock(s, 0, NULL);
}

// fp16 IOSurface I/O (for dynamic matmul kernels with fp16 input/output)
static void io_write_dyn(IOSurfaceRef s, const float *act, int ic, int seq,
                         const float *W, int oc) {
    int sp = seq + oc;
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < ic; d++) {
        cvt_f32_f16(buf + d*sp, act + d*seq, seq);
        cvt_f32_f16(buf + d*sp + seq, W + d*oc, oc);
    }
    IOSurfaceUnlock(s, 0, NULL);
}

// Read output from dynamic matmul kernel: [1, OC, 1, SEQ]
static void io_read_dyn(IOSurfaceRef s, float *out, int oc, int seq) {
    IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
    cvt_f16_f32(out, (_Float16*)IOSurfaceGetBaseAddress(s), oc * seq);
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
}

// Compile MIL to ANE kernel
static Kern *compile_kern_mil_w(NSString *mil, NSDictionary *weights, int ic_bytes, int oc_bytes) {
    @autoreleasepool {
    NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:), md, weights, nil);
    if (!desc) { printf("  [compile] desc=NULL\n"); return NULL; }
    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
    [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    for (NSString *path in weights) {
        NSString *rel = [path stringByReplacingOccurrencesOfString:@"@model_path/" withString:@""];
        [weights[path][@"data"] writeToFile:[td stringByAppendingPathComponent:rel] atomically:YES];
    }
    NSError *e = nil;
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
        printf("  [compile] FAIL: %s\n", e ? [[e description] UTF8String] : "no error"); return NULL;
    }
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
        printf("  [compile] load FAIL\n"); return NULL;
    }
    __sync_fetch_and_add(&g_compile_count, 1);
    Kern *k = (Kern*)calloc(1, sizeof(Kern));
    k->model = (void*)CFBridgingRetain(mdl);
    k->ioIn = make_surface(ic_bytes);
    k->ioOut = make_surface(oc_bytes);
    id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k->ioIn);
    id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k->ioOut);
    k->request = (void*)CFBridgingRetain(((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wI], @[@0], @[wO], @[@0], nil, nil, @0));
    k->tmpDir = (void*)CFBridgingRetain(td);
    return k;
    }
}
// Two-input variant: compile a MIL func main(x, w) and bind both input
// surfaces (indices 0,1) in the template request. Additive — single-input
// compile_kern_mil_w is untouched. Used by the function-parameter weight path.
static Kern *compile_kern_mil_2in(NSString *mil, int ic0_bytes, int ic1_bytes, int oc_bytes) {
    Kern *k = compile_kern_mil_w(mil, @{}, ic0_bytes, oc_bytes);
    if (!k) return NULL;
    @autoreleasepool {
        CFRelease(k->request);  // drop the single-input template request
        k->ioIn1 = make_surface(ic1_bytes);
        id wI0 = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k->ioIn);
        id wI1 = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k->ioIn1);
        id wO  = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k->ioOut);
        k->request = (void*)CFBridgingRetain(((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wI0, wI1], @[@0, @1], @[wO], @[@0], nil, nil, @0));
    }
    return k;
}
// Per-layer request binding two input surfaces (act @0, weight @1) to one output.
static void *make_request_2in(Kern *k, IOSurfaceRef in0, IOSurfaceRef in1) {
    id wI0 = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), in0);
    id wI1 = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), in1);
    id wO  = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k->ioOut);
    id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wI0, wI1], @[@0, @1], @[wO], @[@0], nil, nil, @0);
    return (void*)CFBridgingRetain(req);
}
static void free_kern(Kern *k) {
    if (!k) return;
    id mdl = (__bridge id)k->model; NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &e);
    CFRelease(k->ioIn); CFRelease(k->ioOut);
    if (k->ioIn1) CFRelease(k->ioIn1);
    [[NSFileManager defaultManager] removeItemAtPath:(__bridge id)k->tmpDir error:nil];
    CFRelease(k->model); CFRelease(k->request); CFRelease(k->tmpDir);
    free(k);
}
// A failed ANE eval returns NO and leaves the output surface untouched (stale /
// zero). Silently discarding that BOOL is how a broken kernel masquerades as
// "bitwise-correct" — the eval never runs, the output stays zero, and a naive
// equivalence check passes. Fail LOUD: print the kernel + NSError and abort, so
// "silently wrong gradients" can never hide behind a green gate again.
static void ane_eval_check(id mdl, id req) {
    NSError *e = nil;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
    if (!ok) {
        fprintf(stderr, "\n[ANE EVAL FAILED] evaluateWithQoS returned NO — output "
                "surface is STALE/ZERO, not a real result.\n  error: %s\n",
                e ? [[e description] UTF8String] : "(nil)");
        abort();
    }
}
static void ane_eval(Kern *k) {
    ane_eval_check((__bridge id)k->model, (__bridge id)k->request);
}
static void ane_eval_req(Kern *k, void *request) {
    ane_eval_check((__bridge id)k->model, (__bridge id)request);
}
static void *make_request(Kern *k, IOSurfaceRef ioIn) {
    id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
    id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k->ioOut);
    id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wI], @[@0], @[wO], @[@0], nil, nil, @0);
    return (void*)CFBridgingRetain(req);
}

// ===== Per-layer weight staging for GQA =====
// sdpaFwd: [1, DIM, 1, SEQ + Q_DIM + KV_DIM + KV_DIM] fp16 — no Wo (separate kernel)
//   Wq: [DIM, Q_DIM], Wk: [DIM, KV_DIM], Wv: [DIM, KV_DIM]
#define SDPA_FWD_SP (SEQ + Q_DIM + KV_DIM + KV_DIM)
static void stage_sdpa_fwd_weights(IOSurfaceRef s, const float *Wq, const float *Wk, const float *Wv) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < DIM; d++) {
        cvt_f32_f16(buf + d*SDPA_FWD_SP + SEQ,                   Wq + d*Q_DIM, Q_DIM);
        cvt_f32_f16(buf + d*SDPA_FWD_SP + SEQ+Q_DIM,             Wk + d*KV_DIM, KV_DIM);
        cvt_f32_f16(buf + d*SDPA_FWD_SP + SEQ+Q_DIM+KV_DIM,     Wv + d*KV_DIM, KV_DIM);
    }
    IOSurfaceUnlock(s, 0, NULL);
}
static void write_sdpa_fwd_acts(IOSurfaceRef s, const float *xnorm) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < DIM; d++)
        cvt_f32_f16(buf + d*SDPA_FWD_SP, xnorm + d*SEQ, SEQ);
    IOSurfaceUnlock(s, 0, NULL);
}

// woFwd: [1, Q_DIM, 1, SEQ + DIM] fp16 — Wo: [Q_DIM, DIM]
#define WO_FWD_SP (SEQ + DIM)
static void stage_wo_fwd_weights(IOSurfaceRef s, const float *Wo) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < Q_DIM; d++)
        cvt_f32_f16(buf + d*WO_FWD_SP + SEQ, Wo + d*DIM, DIM);
    IOSurfaceUnlock(s, 0, NULL);
}
static void write_wo_fwd_acts(IOSurfaceRef s, const float *attn_out) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < Q_DIM; d++)
        cvt_f32_f16(buf + d*WO_FWD_SP, attn_out + d*SEQ, SEQ);
    IOSurfaceUnlock(s, 0, NULL);
}
static void copy_wo_fwd_acts_fp16_from_sdpa(IOSurfaceRef s, const _Float16 *attn_out_fp16) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < Q_DIM; d++) {
#if WO_FUNCPARAM
        memcpy(buf + d*SEQ, attn_out_fp16 + d*SEQ, SEQ * sizeof(_Float16));
#else
        memcpy(buf + d*WO_FWD_SP, attn_out_fp16 + d*SEQ, SEQ * sizeof(_Float16));
#endif
    }
    IOSurfaceUnlock(s, 0, NULL);
}
// Function-parameter woFwd (WO_FUNCPARAM): the act surface holds only attn_out
// [1,Q_DIM,1,SEQ] (no packed weight); Wo^T arrives as its own [1,1,Q_DIM,DIM]
// surface, already in matmul shape, so the kernel skips the weight slice/reshape.
static void write_wo_fwd_acts_fp(IOSurfaceRef s, const float *attn_out) {
    io_write_fp16_at(s, 0, attn_out, Q_DIM, SEQ);   // contiguous [Q_DIM, SEQ]
}
static void stage_wo_fwd_w_fp(IOSurfaceRef s, const float *Wot) {
    io_write_fp16_at(s, 0, Wot, Q_DIM, DIM);        // contiguous [Q_DIM, DIM] = Wo^T
}

// ffnFused: [1, DIM, 1, 2*SEQ+3*HIDDEN] fp16
#define FFN_FUSED_SP (2*SEQ + 3*HIDDEN)
static void stage_ffn_fused_weights(IOSurfaceRef s,
                                     const float *W1t, const float *W3t, const float *W2_orig) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < DIM; d++) {
        cvt_f32_f16(buf + d*FFN_FUSED_SP + 2*SEQ,          W1t + d*HIDDEN, HIDDEN);
        cvt_f32_f16(buf + d*FFN_FUSED_SP + 2*SEQ+HIDDEN,   W3t + d*HIDDEN, HIDDEN);
        cvt_f32_f16(buf + d*FFN_FUSED_SP + 2*SEQ+2*HIDDEN, W2_orig + d*HIDDEN, HIDDEN);
    }
    IOSurfaceUnlock(s, 0, NULL);
}
static void write_ffn_fused_acts(IOSurfaceRef s, const float *x2norm, const float *x2) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < DIM; d++) {
        cvt_f32_f16(buf + d*FFN_FUSED_SP,       x2norm + d*SEQ, SEQ);
        cvt_f32_f16(buf + d*FFN_FUSED_SP + SEQ, x2 + d*SEQ, SEQ);
    }
    IOSurfaceUnlock(s, 0, NULL);
}

// ffnBwdW2t: [1, DIM, 1, SEQ+HIDDEN] fp16
#define FFN_BWD_W2T_SP (SEQ + HIDDEN)
static void stage_ffn_bwd_w2t_weights(IOSurfaceRef s, const float *W2) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < DIM; d++)
        cvt_f32_f16(buf + d*FFN_BWD_W2T_SP + SEQ, W2 + d*HIDDEN, HIDDEN);
    IOSurfaceUnlock(s, 0, NULL);
}
static void write_ffn_bwd_w2t_acts(IOSurfaceRef s, const float *dffn) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < DIM; d++)
        cvt_f32_f16(buf + d*FFN_BWD_W2T_SP, dffn + d*SEQ, SEQ);
    IOSurfaceUnlock(s, 0, NULL);
}

// ffnBwdW13t: [1, HIDDEN, 1, 2*SEQ+2*DIM] fp16
#define FFN_BWD_W13T_SP (2*SEQ + 2*DIM)
static void stage_ffn_bwd_w13t_weights(IOSurfaceRef s, const float *W1, const float *W3) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < HIDDEN; d++) {
        cvt_f32_f16(buf + d*FFN_BWD_W13T_SP + 2*SEQ,       W1 + d*DIM, DIM);
        cvt_f32_f16(buf + d*FFN_BWD_W13T_SP + 2*SEQ + DIM, W3 + d*DIM, DIM);
    }
    IOSurfaceUnlock(s, 0, NULL);
}
static void write_ffn_bwd_w13t_acts(IOSurfaceRef s, const float *dh1, const float *dh3) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < HIDDEN; d++) {
        cvt_f32_f16(buf + d*FFN_BWD_W13T_SP,       dh1 + d*SEQ, SEQ);
        cvt_f32_f16(buf + d*FFN_BWD_W13T_SP + SEQ, dh3 + d*SEQ, SEQ);
    }
    IOSurfaceUnlock(s, 0, NULL);
}

// FUSE_SILU_BWD: ffnBwdW13t input grows to [dsilu | h1 | h3 | W1t | W3t] —
// SP = 3*SEQ + 2*DIM (one extra SEQ-wide region vs FFN_BWD_W13T_SP). Same
// spatial-strided layout. dsilu is filled by copy_dsilu_into_w13t_fused; h1,h3
// by write_ffn_bwd_w13t_fused_h; W1,W3 (weights) by the staging fn below.
#define FFN_BWD_W13T_FUSED_SP (3*SEQ + 2*DIM)
static void stage_ffn_bwd_w13t_fused_weights(IOSurfaceRef s, const float *W1, const float *W3) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < HIDDEN; d++) {
        cvt_f32_f16(buf + d*FFN_BWD_W13T_FUSED_SP + 3*SEQ,       W1 + d*DIM, DIM);
        cvt_f32_f16(buf + d*FFN_BWD_W13T_FUSED_SP + 3*SEQ + DIM, W3 + d*DIM, DIM);
    }
    IOSurfaceUnlock(s, 0, NULL);
}
static void write_ffn_bwd_w13t_fused_h(IOSurfaceRef s, const float *h1, const float *h3) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < HIDDEN; d++) {
        cvt_f32_f16(buf + d*FFN_BWD_W13T_FUSED_SP + SEQ,   h1 + d*SEQ, SEQ);
        cvt_f32_f16(buf + d*FFN_BWD_W13T_FUSED_SP + 2*SEQ, h3 + d*SEQ, SEQ);
    }
    IOSurfaceUnlock(s, 0, NULL);
}
// Strided ANE->ANE copy: dsilu lives in ffnBwdW2t's OUTPUT as contiguous
// [HIDDEN,SEQ] (stride SEQ); place it at offset 0 of each channel of the fused
// W13t input (stride FFN_BWD_W13T_FUSED_SP). fp16->fp16, no conversion, no CPU
// round-trip — the silu-bwd math never leaves the ANE.
static void copy_dsilu_into_w13t_fused(IOSurfaceRef dst, IOSurfaceRef src) {
    IOSurfaceLock(dst, 0, NULL);
    IOSurfaceLock(src, kIOSurfaceLockReadOnly, NULL);
    _Float16 *db = (_Float16*)IOSurfaceGetBaseAddress(dst);
    _Float16 *sb = (_Float16*)IOSurfaceGetBaseAddress(src);
    for (int d = 0; d < HIDDEN; d++)
        memcpy(db + d*FFN_BWD_W13T_FUSED_SP, sb + d*SEQ, SEQ * sizeof(_Float16));
    IOSurfaceUnlock(src, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceUnlock(dst, 0, NULL);
}

// wotBwd: [1, DIM, 1, SEQ+Q_DIM] fp16 — Wo is [DIM, Q_DIM], matmul gives Wo^T @ dy
#define WOT_BWD_SP (SEQ + Q_DIM)
static void stage_wot_bwd_weights(IOSurfaceRef s, const float *Wo) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < DIM; d++)
        cvt_f32_f16(buf + d*WOT_BWD_SP + SEQ, Wo + d*Q_DIM, Q_DIM);
    IOSurfaceUnlock(s, 0, NULL);
}
static void write_wot_bwd_acts(IOSurfaceRef s, const float *dy) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < DIM; d++)
        cvt_f32_f16(buf + d*WOT_BWD_SP, dy + d*SEQ, SEQ);
    IOSurfaceUnlock(s, 0, NULL);
}
#if CONV1IN == 2
// Step-B staging (PRD #26): write the weight region pre-transposed so the conv
// MIL (gen_conv_1in_mil_B) reshapes the weight slice [1,IC,1,OC] straight to the
// conv kernel [OC,IC,1,1] = W[o,i]=M[i,o] with NO in-MIL transpose. Wo is M
// [DIM=IC, Q_DIM=OC]. The reshape preserves logical-flat order, so conv-kernel
// flat p=o*IC+i must hold M[i,o], i.e. sliced element [0,p/OC,0,p%OC] =
// buf[(p/OC)*WOT_BWD_SP + SEQ + (p%OC)]. (IC==OC here, but written generally.)
static void stage_wot_bwd_weights_convB(IOSurfaceRef s, const float *Wo) {
    const int IC = DIM, OC = Q_DIM;
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int o = 0; o < OC; o++)
        for (int i = 0; i < IC; i++) {
            int p = o*IC + i;            // conv-kernel logical-flat index
            buf[(p/OC)*WOT_BWD_SP + SEQ + (p%OC)] = (_Float16)Wo[i*OC + o];  // M[i,o]
        }
    IOSurfaceUnlock(s, 0, NULL);
}
#endif

// qBwd: [1, Q_DIM, 1, SEQ+DIM] fp16 — Wq is [Q_DIM, DIM], matmul gives Wq^T @ dq
#define Q_BWD_SP (SEQ + DIM)
static void stage_q_bwd_weights(IOSurfaceRef s, const float *Wq) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < Q_DIM; d++)
        cvt_f32_f16(buf + d*Q_BWD_SP + SEQ, Wq + d*DIM, DIM);
    IOSurfaceUnlock(s, 0, NULL);
}
static void write_q_bwd_acts(IOSurfaceRef s, const float *dq) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < Q_DIM; d++)
        cvt_f32_f16(buf + d*Q_BWD_SP, dq + d*SEQ, SEQ);
    IOSurfaceUnlock(s, 0, NULL);
}

// kvBwd: [1, KV_DIM, 1, 2*SEQ+2*DIM] fp16 — dk @ Wk + dv @ Wv → dx_kv
#define KV_BWD_SP (2*SEQ + 2*DIM)
static void stage_kv_bwd_weights(IOSurfaceRef s, const float *Wk, const float *Wv) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < KV_DIM; d++) {
        cvt_f32_f16(buf + d*KV_BWD_SP + 2*SEQ,       Wk + d*DIM, DIM);
        cvt_f32_f16(buf + d*KV_BWD_SP + 2*SEQ + DIM, Wv + d*DIM, DIM);
    }
    IOSurfaceUnlock(s, 0, NULL);
}
static void write_kv_bwd_acts(IOSurfaceRef s, const float *dk, const float *dv) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < KV_DIM; d++) {
        cvt_f32_f16(buf + d*KV_BWD_SP,       dk + d*SEQ, SEQ);
        cvt_f32_f16(buf + d*KV_BWD_SP + SEQ, dv + d*SEQ, SEQ);
    }
    IOSurfaceUnlock(s, 0, NULL);
}

#if FUSE_QKVBWD
// qkvBwd (fused): one packed Q_DIM-channel surface =
// [dq | dk | dv | Wq | Wk | Wv]. Under GQA, K/V occupy only the first KV_DIM
// channels of their slices; the MIL slices those narrower channel ranges. The
// staged layouts match the split qBwd/kvBwd paths, so the math is byte-identical
// to dx_q + dx_kv, summed in-kernel.
#define QKV_BWD_SP (3*SEQ + 3*DIM)
static void stage_qkv_bwd_weights(IOSurfaceRef s, const float *Wq,
                                  const float *Wk, const float *Wv) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < Q_DIM; d++)
        cvt_f32_f16(buf + d*QKV_BWD_SP + 3*SEQ, Wq + d*DIM, DIM);
    for (int d = 0; d < KV_DIM; d++) {
        cvt_f32_f16(buf + d*QKV_BWD_SP + 3*SEQ + DIM,   Wk + d*DIM, DIM);
        cvt_f32_f16(buf + d*QKV_BWD_SP + 3*SEQ + 2*DIM, Wv + d*DIM, DIM);
    }
    IOSurfaceUnlock(s, 0, NULL);
}
static void write_qkv_bwd_acts(IOSurfaceRef s, const float *dq,
                               const float *dk, const float *dv) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < Q_DIM; d++)
        cvt_f32_f16(buf + d*QKV_BWD_SP, dq + d*SEQ, SEQ);
    for (int d = 0; d < KV_DIM; d++) {
        cvt_f32_f16(buf + d*QKV_BWD_SP + SEQ,   dk + d*SEQ, SEQ);
        cvt_f32_f16(buf + d*QKV_BWD_SP + 2*SEQ, dv + d*SEQ, SEQ);
    }
    IOSurfaceUnlock(s, 0, NULL);
}
#endif

// SDPA backward hot-path packing/unpacking. These keep GQA expansion/reduction at
// the fp16 IOSurface boundary instead of materializing full Q_DIM K/V tensors in
// fp32 scratch buffers for every layer.
static void cvt_gqa_kv_f32_to_f16_tiled(_Float16 *dst, const float *src, int seq) {
#if GQA_RATIO == 1
    cvt_f32_f16(dst, src, KV_DIM * seq);
#else
    const int chunk = HD * seq;
    for (int r = 0; r < GQA_RATIO; r++) {
        for (int kv = 0; kv < KV_HEADS; kv++) {
            int q_head = r * KV_HEADS + kv;
            cvt_f32_f16(dst + q_head * chunk, src + kv * chunk, chunk);
        }
    }
#endif
}

static void reduce_gqa_fp16_to_f32(float *out, const _Float16 *in, int seq) {
#if GQA_RATIO == 1
    cvt_f16_f32(out, in, KV_DIM * seq);
#else
    const int chunk = HD * seq;
    float tmp[HD * SEQ];
    for (int kv = 0; kv < KV_HEADS; kv++) {
        float *dst = out + kv * chunk;
        cvt_f16_f32(dst, in + kv * chunk, chunk);
        for (int r = 1; r < GQA_RATIO; r++) {
            int q_head = r * KV_HEADS + kv;
            cvt_f16_f32(tmp, in + q_head * chunk, chunk);
            vDSP_vadd(dst, 1, tmp, 1, dst, 1, (vDSP_Length)chunk);
        }
    }
#endif
}

static void write_sdpa_bwd1_acts_gqa(IOSurfaceRef s, const float *q,
                                     const float *k, const float *v,
                                     const float *da) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    cvt_f32_f16(buf, q, Q_DIM * SEQ);
    cvt_gqa_kv_f32_to_f16_tiled(buf + Q_DIM * SEQ, k, SEQ);
    cvt_gqa_kv_f32_to_f16_tiled(buf + 2 * Q_DIM * SEQ, v, SEQ);
    cvt_f32_f16(buf + 3 * Q_DIM * SEQ, da, Q_DIM * SEQ);
    IOSurfaceUnlock(s, 0, NULL);
}

static void write_sdpa_bwd_v_acts_gqa(IOSurfaceRef s, const float *q,
                                      const float *k, const float *da) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    cvt_f32_f16(buf, q, Q_DIM * SEQ);
    cvt_gqa_kv_f32_to_f16_tiled(buf + Q_DIM * SEQ, k, SEQ);
    cvt_f32_f16(buf + 2 * Q_DIM * SEQ, da, Q_DIM * SEQ);
    IOSurfaceUnlock(s, 0, NULL);
}

static void write_sdpa_bwd2_input_gqa(IOSurfaceRef dst, IOSurfaceRef bwd1_out,
                                      const float *q, const float *k) {
    IOSurfaceLock(dst, 0, NULL);
    IOSurfaceLock(bwd1_out, kIOSurfaceLockReadOnly, NULL);
    _Float16 *db = (_Float16*)IOSurfaceGetBaseAddress(dst);
    _Float16 *sb = (_Float16*)IOSurfaceGetBaseAddress(bwd1_out);
    memcpy(db, sb + Q_DIM * SEQ, 2 * SCORE_CH * SEQ * sizeof(_Float16));
    cvt_f32_f16(db + 2 * SCORE_CH * SEQ, q, Q_DIM * SEQ);
    cvt_gqa_kv_f32_to_f16_tiled(db + (2 * SCORE_CH + Q_DIM) * SEQ, k, SEQ);
    IOSurfaceUnlock(bwd1_out, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceUnlock(dst, 0, NULL);
}

static void read_sdpa_bwd_outputs_gqa(IOSurfaceRef bwd2_out, IOSurfaceRef bwd1_out,
                                      float *dq, float *dk, float *dv) {
    IOSurfaceLock(bwd2_out, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceLock(bwd1_out, kIOSurfaceLockReadOnly, NULL);
    _Float16 *b2 = (_Float16*)IOSurfaceGetBaseAddress(bwd2_out);
    _Float16 *b1 = (_Float16*)IOSurfaceGetBaseAddress(bwd1_out);
    cvt_f16_f32(dq, b2, Q_DIM * SEQ);
    reduce_gqa_fp16_to_f32(dk, b2 + Q_DIM * SEQ, SEQ);
    reduce_gqa_fp16_to_f32(dv, b1, SEQ);
    IOSurfaceUnlock(bwd1_out, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceUnlock(bwd2_out, kIOSurfaceLockReadOnly, NULL);
}

static void read_sdpa_bwd_fused_outputs_gqa(IOSurfaceRef s,
                                            float *dq, float *dk, float *dv) {
    IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    cvt_f16_f32(dq, buf, Q_DIM * SEQ);
    reduce_gqa_fp16_to_f32(dk, buf + Q_DIM * SEQ, SEQ);
    reduce_gqa_fp16_to_f32(dv, buf + 2 * Q_DIM * SEQ, SEQ);
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
}

// Free per-layer surfaces and requests
static void free_per_layer(PerLayerSurfaces *pls, PerLayerRequests *plr) {
    for (int L = 0; L < NLAYERS; L++) {
        CFRelease(pls[L].sdpaFwd_in); CFRelease(pls[L].woFwd_in); CFRelease(pls[L].ffnFused_in);
#if WO_FUNCPARAM
        CFRelease(pls[L].woFwd_w);
#endif
#if W2T_FUNCPARAM
        CFRelease(pls[L].ffnBwdW2t_w);
#endif
#if CONV_DATAPATH
        CFRelease(pls[L].wotBwd_w); CFRelease(pls[L].qBwd_w);
#endif
        CFRelease(pls[L].ffnBwdW2t_in); CFRelease(pls[L].ffnBwdW13t_in);
        CFRelease(pls[L].wotBwd_in);
        CFRelease(plr[L].sdpaFwd); CFRelease(plr[L].woFwd); CFRelease(plr[L].ffnFused);
        CFRelease(plr[L].ffnBwdW2t); CFRelease(plr[L].ffnBwdW13t);
        CFRelease(plr[L].wotBwd);
#if FUSE_QKVBWD
        CFRelease(pls[L].qkvBwd_in);
        CFRelease(plr[L].qkvBwd);
#else
        CFRelease(pls[L].qBwd_in); CFRelease(pls[L].kvBwd_in);
        CFRelease(plr[L].qBwd); CFRelease(plr[L].kvBwd);
#endif
    }
}

// GQA helpers: tile KV from KV_HEADS to HEADS, and reduce HEADS to KV_HEADS.
// CONVENTION: q-head q maps to kv-head (q % KV_HEADS) — i.e. the tiled head
// order is [kv0..kvN-1, kv0..kvN-1, ...] (GQA_RATIO whole copies). This MUST
// match the forward kernel, which tiles with concat(interleave=false) over
// GQA_RATIO copies of k_rope (mil_dynamic.h). The earlier block layout
// (q_head = kv*GQA_RATIO + r) disagreed with the forward and produced silently
// wrong K/V gradients for any GQA_RATIO>1 model (e.g. qwen3_06b); caught by R1.
// tile_kv: input [KV_DIM, SEQ], output [Q_DIM, SEQ]
static void gqa_tile_kv(float *out, const float *in, int seq) {
    for (int r = 0; r < GQA_RATIO; r++) {
        for (int kv = 0; kv < KV_HEADS; kv++) {
            int q_head = r * KV_HEADS + kv;
            memcpy(out + q_head * HD * seq, in + kv * HD * seq, HD * seq * sizeof(float));
        }
    }
}
// reduce_kv: input [Q_DIM, SEQ], output [KV_DIM, SEQ]
// Sum contributions from Q heads sharing each KV head (interleaved: q % KV_HEADS)
static void gqa_reduce_kv(float *out, const float *in, int seq) {
    memset(out, 0, KV_DIM * seq * sizeof(float));
    for (int r = 0; r < GQA_RATIO; r++) {
        for (int kv = 0; kv < KV_HEADS; kv++) {
            int q_head = r * KV_HEADS + kv;
            const float *src = in + q_head * HD * seq;
            float *dst = out + kv * HD * seq;
            for (int i = 0; i < HD * seq; i++)
                dst[i] += src[i];
        }
    }
}
