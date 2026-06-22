// test_heads_cpu.c — finite-difference gate for the NEW chess heads' backward
// (issue #16 acceptance criterion: "analytic-vs-numeric check on the new heads'
// backward"). Pure C / fp32, the same pattern as test_attn_cpu.c: run the analytic
// backward, then central-difference every input and assert the max disagreement is
// tiny. This is the per-kernel correctness discipline the project demands — a
// silently-wrong head backward is exactly what G0 (and this test) exist to catch.
//
// It exercises chess_heads.h, the SAME code train_chess.m runs, at small dims.
//
// Build: cc -O2 -std=c11 -o test_heads test_heads_cpu.c -lm   (see Makefile target)

#include "chess_heads.h"
#include <stdio.h>
#include <stdlib.h>

static int g_fail = 0;
static float fr(void) { return 2.0f*((float)rand()/(float)RAND_MAX) - 1.0f; }

// Central-difference every entry of `arr` (n long) against analytic `grad`,
// recomputing the scalar loss via the LOSS() expression. Tracks the worst abs error.
#define CHK(arr, n, grad, LOSS) do { \
    for (int _i = 0; _i < (n); _i++) { \
        float _sv = (arr)[_i]; \
        (arr)[_i] = _sv + eps; float _lp = (LOSS); \
        (arr)[_i] = _sv - eps; float _lm = (LOSS); \
        (arr)[_i] = _sv; \
        float _num = (_lp - _lm) / (2*eps); \
        float _e = fabsf(_num - (grad)[_i]); \
        if (_e > worst) worst = _e; \
    } \
} while (0)

static void report(const char *name, float worst, float tol) {
    int ok = worst < tol;
    printf("  %-34s max|analytic-numeric| = %.2e  %s\n", name, worst, ok ? "OK" : "*** FAIL ***");
    if (!ok) g_fail++;
}

// ---- policy head FD check ----------------------------------------------------
static void test_policy(void) {
    printf("[policy head: masked-softmax CE backward]\n");
    srand(11);
    const int dim = 6, seq = 10, nboard = 4, planes = 5, N = nboard*planes;
    float x[6*10], W[6*5];
    for (int i = 0; i < dim*seq; i++) x[i] = fr();
    for (int i = 0; i < dim*planes; i++) W[i] = 0.5f*fr();
    // A plausible legal set (a handful of squares/planes) + a one-hot target on one legal index.
    uint8_t mask[20]; float tgt[20];
    for (int i = 0; i < N; i++) { mask[i] = 0; tgt[i] = 0.0f; }
    int legal[] = {0*planes+1, 0*planes+3, 1*planes+0, 2*planes+4, 3*planes+2, 3*planes+0};
    for (int j = 0; j < (int)(sizeof(legal)/sizeof(int)); j++) mask[legal[j]] = 1;
    tgt[legal[2]] = 1.0f;  // one-hot target -> loss floor is 0 (the G0 shape)

    float dx[6*10] = {0}, dW[6*5] = {0};
    chess_policy_loss(x, W, dim, seq, nboard, planes, mask, tgt, dx, dW);

    float eps = 1e-3f, worst = 0;
    CHK(x, dim*seq, dx, chess_policy_loss(x, W, dim, seq, nboard, planes, mask, tgt, NULL, NULL));
    report("d(loss)/d(x)", worst, 5e-3f);
    worst = 0;
    CHK(W, dim*planes, dW, chess_policy_loss(x, W, dim, seq, nboard, planes, mask, tgt, NULL, NULL));
    report("d(loss)/d(W_pol)", worst, 5e-3f);
}

// ---- value head FD check ----------------------------------------------------
static void test_value(void) {
    printf("[value head: WDL softmax CE backward]\n");
    srand(23);
    const int dim = 6, seq = 10, nreal = 8, nwdl = 3;
    float x[6*10], W[6*3], tgt[3] = {0,0,0};
    for (int i = 0; i < dim*seq; i++) x[i] = fr();
    for (int i = 0; i < dim*nwdl; i++) W[i] = 0.5f*fr();
    tgt[1] = 1.0f;  // one-hot "draw" -> loss floor 0

    float dx[6*10] = {0}, dW[6*3] = {0};
    chess_value_loss(x, W, dim, seq, nreal, nwdl, tgt, dx, dW);

    float eps = 1e-3f, worst = 0;
    CHK(x, dim*seq, dx, chess_value_loss(x, W, dim, seq, nreal, nwdl, tgt, NULL, NULL));
    report("d(loss)/d(x)", worst, 5e-3f);
    worst = 0;
    CHK(W, dim*nwdl, dW, chess_value_loss(x, W, dim, seq, nreal, nwdl, tgt, NULL, NULL));
    report("d(loss)/d(W_val)", worst, 5e-3f);
}

// ---- 2D posenc FD check + rank/file additivity ------------------------------
// posenc has no loss of its own; probe its backward through L = sum(da * x_post).
static const float *g_da;  // [dim*seq]
static int g_dim, g_seq, g_nboard;
static float posenc_probe_loss(const float *x0, const float *rk, const float *fl, const float *ms) {
    float *x = (float*)malloc((size_t)g_dim*g_seq*sizeof(float));
    memcpy(x, x0, (size_t)g_dim*g_seq*sizeof(float));
    chess_posenc_forward(x, rk, fl, ms, g_dim, g_seq, g_nboard);
    float l = 0; for (int i = 0; i < g_dim*g_seq; i++) l += g_da[i]*x[i];
    free(x); return l;
}
static void test_posenc(void) {
    printf("[2D posenc: rank+file+misc backward]\n");
    srand(37);
    const int dim = 4, seq = 72, nboard = 64, nmisc = seq - nboard;
    float x0[4*72], da[4*72];
    float rk[8*4], fl[8*4], ms[8*4];
    for (int i = 0; i < dim*seq; i++) { x0[i] = fr(); da[i] = fr(); }
    for (int i = 0; i < 8*dim; i++) { rk[i] = fr(); fl[i] = fr(); ms[i] = fr(); }
    g_da = da; g_dim = dim; g_seq = seq; g_nboard = nboard;

    float drk[8*4] = {0}, dfl[8*4] = {0}, dms[8*4] = {0};
    chess_posenc_backward(da, drk, dfl, dms, dim, seq, nboard);

    float eps = 1e-3f, worst = 0;
    CHK(rk, 8*dim, drk, posenc_probe_loss(x0, rk, fl, ms));
    report("d(L)/d(rank_emb)", worst, 5e-3f);
    worst = 0;
    CHK(fl, 8*dim, dfl, posenc_probe_loss(x0, rk, fl, ms));
    report("d(L)/d(file_emb)", worst, 5e-3f);
    worst = 0;
    CHK(ms, nmisc*dim, dms, posenc_probe_loss(x0, rk, fl, ms));
    report("d(L)/d(misc_emb)", worst, 5e-3f);

    // --- rank/file additivity unit check (issue #16 AC) ---
    // posenc(square) must decompose EXACTLY as rank_emb[rank] + file_emb[file].
    float zero[4*72] = {0};
    chess_posenc_forward(zero, rk, fl, ms, dim, seq, nboard);
    float decomp_err = 0, add_err = 0;
    for (int sq = 0; sq < nboard; sq++) {
        int rank = sq >> 3, file = sq & 7;
        for (int d = 0; d < dim; d++) {
            float pe = zero[d*seq + sq];
            float want = rk[rank*dim + d] + fl[file*dim + d];
            float e = fabsf(pe - want); if (e > decomp_err) decomp_err = e;
        }
    }
    // Additive geometry => pe(a1)+pe(h8) - pe(a8) - pe(h1) == 0 for every channel
    // (a1=0,h1=7,a8=56,h8=63): the rank and file contributions cancel.
    for (int d = 0; d < dim; d++) {
        float v = zero[d*seq+0] + zero[d*seq+63] - zero[d*seq+56] - zero[d*seq+7];
        if (fabsf(v) > add_err) add_err = fabsf(v);
    }
    report("posenc == rank+file (decomp)", decomp_err, 1e-6f);
    report("rank/file additivity (a1+h8=a8+h1)", add_err, 1e-5f);
}

// ---- combined AZ dx check ----------------------------------------------------
// The trainer sums policy + value gradients into ONE dx[dim,seq]; verify that
// combined dx (the gradient handed to the trunk) matches central differences.
static int gc_dim, gc_seq, gc_nboard, gc_planes, gc_nreal, gc_nwdl;
static const uint8_t *gc_mask; static const float *gc_tp, *gc_tv, *gc_Wp, *gc_Wv;
static float gc_vw;
static float az_loss(const float *x) {
    float lp = chess_policy_loss(x, gc_Wp, gc_dim, gc_seq, gc_nboard, gc_planes, gc_mask, gc_tp, NULL, NULL);
    float lv = chess_value_loss (x, gc_Wv, gc_dim, gc_seq, gc_nreal, gc_nwdl, gc_tv, NULL, NULL);
    return lp + gc_vw*lv;
}
static void test_combined(void) {
    printf("[AZ loss: combined policy+value dx into trunk]\n");
    srand(53);
    const int dim = 6, seq = 12, nboard = 4, planes = 5, nreal = 10, nwdl = 3, N = nboard*planes;
    static float x[6*12], Wp[6*5], Wv[6*3];
    static uint8_t mask[20]; static float tp[20], tv[3] = {0,0,0};
    for (int i = 0; i < dim*seq; i++) x[i] = fr();
    for (int i = 0; i < dim*planes; i++) Wp[i] = 0.5f*fr();
    for (int i = 0; i < dim*nwdl; i++) Wv[i] = 0.5f*fr();
    for (int i = 0; i < N; i++) { mask[i] = 0; tp[i] = 0; }
    int legal[] = {1, 3, planes+0, 2*planes+4, 3*planes+2};
    for (int j = 0; j < (int)(sizeof(legal)/sizeof(int)); j++) mask[legal[j]] = 1;
    tp[legal[1]] = 1.0f; tv[2] = 1.0f;
    float vw = 1.0f;

    float dx[6*12] = {0}, dWp[6*5] = {0}, dWv[6*3] = {0};
    chess_policy_loss(x, Wp, dim, seq, nboard, planes, mask, tp, dx, dWp);
    // value grad scaled by vw, same dx accumulator (mirrors the trainer)
    float dxv[6*12] = {0};
    chess_value_loss(x, Wv, dim, seq, nreal, nwdl, tv, dxv, dWv);
    for (int i = 0; i < dim*seq; i++) dx[i] += vw*dxv[i];

    gc_dim=dim; gc_seq=seq; gc_nboard=nboard; gc_planes=planes; gc_nreal=nreal; gc_nwdl=nwdl;
    gc_mask=mask; gc_tp=tp; gc_tv=tv; gc_Wp=Wp; gc_Wv=Wv; gc_vw=vw;
    float eps = 1e-3f, worst = 0;
    CHK(x, dim*seq, dx, az_loss(x));
    report("d(policy+value)/d(x)", worst, 5e-3f);
}

int main(void) {
    test_policy();
    test_value();
    test_posenc();
    test_combined();
    printf("\n%s (%d failure%s)\n", g_fail ? "TESTS FAILED" : "ALL HEAD FD CHECKS PASSED",
           g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
