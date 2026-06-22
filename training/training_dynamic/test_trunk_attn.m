// test_attn_cpu.m — finite-difference gate for the trunk attention backward
// (issue #36): analytic-vs-numeric check on attn_cpu_backward_batched's dQ/dK/dV.
//
// The heads have test_heads_cpu.c; the trunk attention backward had NO FD gate —
// which is exactly how the dQ NEON fold bug (HD>8) slipped through. This is the
// per-kernel gate the project demands, exercised on the REAL static functions in
// chess_net.h at production chess_g0 dims (HD=32 triggers the bug).
//
// Mirrors test_heads_cpu.c's CHK pattern: run the analytic backward, central-
// difference every input through the forward, assert max disagreement is tiny.
//
// Build: make test_trunk_attn   (see Makefile target; mirrors probe_autodiff's build).

#include "mil_dynamic.h"
#include "cpu_ops.h"
#include "chess/chess.h"
#include "chess/chess_heads.h"
#include "chess/chess_net.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int g_fail = 0;
static float fr(unsigned *seed) { *seed = *seed * 1103515245u + 12345u;
    return 2.0f*((float)((*seed >> 8) & 0xffffff) / 16777216.0f) - 1.0f; }

// Linear-probe loss = <da, attn_out(Q,K,V)> — the VJP source whose gradient the
// hand-written backward computes. Recomputes attn_out via the real forward.
static float attn_loss(const float *Q, const float *K, const float *V, const float *da,
                       float *attn_out_scratch, int B, int seqp) {
    attn_cpu_forward_batched(attn_out_scratch, Q, K, V, B, seqp);
    size_t n = (size_t)Q_DIM * B * seqp;
    double s = 0;
    for (size_t i = 0; i < n; i++) s += (double)da[i] * attn_out_scratch[i];
    return (float)s;
}

#define CHK(arr, n, grad, LOSS_EXPR) do { \
    for (size_t _i = 0; _i < (size_t)(n); _i++) { \
        float _sv = (arr)[_i]; \
        (arr)[_i] = _sv + eps; float _lp = (LOSS_EXPR); \
        (arr)[_i] = _sv - eps; float _lm = (LOSS_EXPR); \
        (arr)[_i] = _sv; \
        float _num = (_lp - _lm) / (2*eps); \
        float _e = fabsf(_num - (grad)[_i]); \
        if (_e > worst) worst = _e; \
    } \
} while (0)

static void report(const char *name, float worst, float tol) {
    int ok = worst < tol;
    printf("  %-34s max|analytic-numeric| = %.3e  %s\n", name, worst, ok ? "OK" : "*** FAIL ***");
    if (!ok) g_fail++;
}

static void test_attn_backward(void) {
    printf("[trunk attention backward: causal MHA dQ/dK/dV vs central-difference]\n");
    const int B = 1, seqp = 8;
    const int S = B * seqp;
    unsigned seed = 7;

    float *Q  = fmalloc((size_t)Q_DIM*S);
    float *K  = fmalloc((size_t)KV_DIM*S);
    float *V  = fmalloc((size_t)KV_DIM*S);
    float *da = fmalloc((size_t)Q_DIM*S);
    float *dQ = fmalloc((size_t)Q_DIM*S);
    float *dK = fmalloc((size_t)KV_DIM*S);
    float *dV = fmalloc((size_t)KV_DIM*S);
    float *scratch = fmalloc((size_t)Q_DIM*S);
    for (size_t i = 0; i < (size_t)Q_DIM*S;  i++) Q[i]  = 0.5f*fr(&seed);
    for (size_t i = 0; i < (size_t)KV_DIM*S; i++) K[i]  = 0.5f*fr(&seed);
    for (size_t i = 0; i < (size_t)KV_DIM*S; i++) V[i]  = 0.5f*fr(&seed);
    for (size_t i = 0; i < (size_t)Q_DIM*S;  i++) da[i] = fr(&seed);

    printf("  dims: HEADS=%d KV_HEADS=%d HD=%d Q_DIM=%d KV_DIM=%d  (B=%d seqp=%d, HD>8 triggers %s)\n",
           HEADS, KV_HEADS, HD, Q_DIM, KV_DIM, B, seqp, (HD > 8) ? "the bug" : "no bug (HD<=8)");

    attn_cpu_backward_batched(da, Q, K, V, B, seqp, dQ, dK, dV);

    float eps = 1e-3f, worst = 0;
    CHK(Q, Q_DIM*S, dQ, attn_loss(Q, K, V, da, scratch, B, seqp));
    report("d(loss)/d(Q)", worst, 5e-3f);
    worst = 0;
    CHK(K, KV_DIM*S, dK, attn_loss(Q, K, V, da, scratch, B, seqp));
    report("d(loss)/d(K)", worst, 5e-3f);
    worst = 0;
    CHK(V, KV_DIM*S, dV, attn_loss(Q, K, V, da, scratch, B, seqp));
    report("d(loss)/d(V)", worst, 5e-3f);

    free(Q);free(K);free(V);free(da);free(dQ);free(dK);free(dV);free(scratch);
}

int main(void) {
    test_attn_backward();
    printf("\n%s (%d failure%s)\n", g_fail ? "TESTS FAILED" : "ALL ATTN FD CHECKS PASSED",
           g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
