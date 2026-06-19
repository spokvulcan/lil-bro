// mhc_sinkhorn_spike.c — issue #5: de-risk the hardest mHC numeric before the
// full residual integration (#11). Unrolled log-domain Sinkhorn-Knopp over a 4x4
// matrix, t_max=20, forward + backward, with an fp16 doubly-stochasticity check.
//
// "ANE CPU path": this is the CPU-side numeric that will land alongside the
// trainer's existing CPU ops (RMSNorm, GQA tile/reduce, RoPE-bwd). No MLX/torch,
// no gradient oracle — correctness is (a) the doubly-stochasticity forward
// property, (b) a finite-difference self-check of the unrolled backward, and
// (c) a behavioral overfit of an isolated mHC mixing block.
//
//   cc -O2 -o mhc_sinkhorn_spike mhc_sinkhorn_spike.c -lm && ./mhc_sinkhorn_spike
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define N      4      // n_hc streams (4x4 residual map B)
#define TMAX   20     // Sinkhorn iterations (V4 §4.2.1)

// ---- log-domain Sinkhorn forward (double), taping the per-iter softmaxes ----
// B = exp(L + f_i + g_j), L = Btilde/tau, with alternating dual updates:
//   f_i = -logsumexp_j(L_ij + g_j)   (rows -> sum 1)
//   g_j = -logsumexp_i(L_ij + f_i)   (cols -> sum 1)
// Tapes P[t] = row-softmax(L + g^{t-1}) and Q[t] = col-softmax(L + f^t).
typedef struct {
    double L[N][N], f[N], g[N], B[N][N];
    double P[TMAX][N][N], Q[TMAX][N][N];   // taped softmaxes for the backward
    double tau;
} Tape;

static double logsumexp(const double *x, int n) {
    double m = x[0];
    for (int i = 1; i < n; i++) if (x[i] > m) m = x[i];
    double s = 0;
    for (int i = 0; i < n; i++) s += exp(x[i] - m);
    return m + log(s);
}

static void sinkhorn_fwd(const double Btilde[N][N], double tau, Tape *T) {
    T->tau = tau;
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) T->L[i][j] = Btilde[i][j] / tau;
    for (int i = 0; i < N; i++) { T->f[i] = 0; T->g[i] = 0; }
    for (int t = 0; t < TMAX; t++) {
        // row update: f_i = -lse_j(L_ij + g_j); tape P = softmax over j
        for (int i = 0; i < N; i++) {
            double row[N];
            for (int j = 0; j < N; j++) row[j] = T->L[i][j] + T->g[j];
            double lse = logsumexp(row, N);
            T->f[i] = -lse;
            for (int j = 0; j < N; j++) T->P[t][i][j] = exp(row[j] - lse);
        }
        // col update: g_j = -lse_i(L_ij + f_i); tape Q = softmax over i
        for (int j = 0; j < N; j++) {
            double col[N];
            for (int i = 0; i < N; i++) col[i] = T->L[i][j] + T->f[i];
            double lse = logsumexp(col, N);
            T->g[j] = -lse;
            for (int i = 0; i < N; i++) T->Q[t][i][j] = exp(col[i] - lse);
        }
    }
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++)
        T->B[i][j] = exp(T->L[i][j] + T->f[i] + T->g[j]);
}

// ---- unrolled backward: dB -> dBtilde, reversing the taped iterations ----
static void sinkhorn_bwd(const Tape *T, const double dB[N][N], double dBtilde[N][N]) {
    double dL[N][N] = {{0}}, df[N] = {0}, dg[N] = {0};
    // B reverse: dB_ij*B_ij flows to L_ij, f_i, g_j
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
        double t = dB[i][j] * T->B[i][j];
        dL[i][j] += t; df[i] += t; dg[j] += t;
    }
    for (int t = TMAX - 1; t >= 0; t--) {
        // reverse col update g_j = -lse_i(L_ij + f_i), Q = col-softmax
        double df_new[N] = {0};
        for (int j = 0; j < N; j++) for (int i = 0; i < N; i++) {
            double c = -dg[j] * T->Q[t][i][j];
            dL[i][j] += c; df_new[i] += c;
        }
        for (int i = 0; i < N; i++) { df[i] += df_new[i]; }
        for (int j = 0; j < N; j++) dg[j] = 0;
        // reverse row update f_i = -lse_j(L_ij + g_j), P = row-softmax
        double dg_new[N] = {0};
        for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
            double c = -df[i] * T->P[t][i][j];
            dL[i][j] += c; dg_new[j] += c;
        }
        for (int j = 0; j < N; j++) dg[j] = dg_new[j];
        for (int i = 0; i < N; i++) df[i] = 0;
    }
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) dBtilde[i][j] = dL[i][j] / T->tau;
}

// ---- fp16 forward: does B stay doubly-stochastic in fp16 at t_max=20? ----
static double sinkhorn_fwd_fp16_n(const double Btilde[N][N], double tau, int iters,
                                  double *max_row_err, double *max_col_err) {
    _Float16 L[N][N], f[N], g[N], B[N][N];
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) L[i][j] = (_Float16)(Btilde[i][j] / tau);
    for (int i = 0; i < N; i++) { f[i] = 0; g[i] = 0; }
    for (int t = 0; t < iters; t++) {
        for (int i = 0; i < N; i++) {
            _Float16 row[N], m = (_Float16)-1e4f;
            for (int j = 0; j < N; j++) { row[j] = (_Float16)(L[i][j] + g[j]); if (row[j] > m) m = row[j]; }
            float s = 0; for (int j = 0; j < N; j++) s += expf((float)(row[j] - m));
            f[i] = (_Float16)(-((float)m + logf(s)));
        }
        for (int j = 0; j < N; j++) {
            _Float16 col[N], m = (_Float16)-1e4f;
            for (int i = 0; i < N; i++) { col[i] = (_Float16)(L[i][j] + f[i]); if (col[i] > m) m = col[i]; }
            float s = 0; for (int i = 0; i < N; i++) s += expf((float)(col[i] - m));
            g[j] = (_Float16)(-((float)m + logf(s)));
        }
    }
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) B[i][j] = (_Float16)expf((float)(L[i][j] + f[i] + g[j]));
    double mr = 0, mc = 0;
    for (int i = 0; i < N; i++) { double r = 0; for (int j = 0; j < N; j++) r += (double)B[i][j]; if (fabs(r-1) > mr) mr = fabs(r-1); }
    for (int j = 0; j < N; j++) { double c = 0; for (int i = 0; i < N; i++) c += (double)B[i][j]; if (fabs(c-1) > mc) mc = fabs(c-1); }
    *max_row_err = mr; *max_col_err = mc;
    return mr > mc ? mr : mc;
}
static double sinkhorn_fwd_fp16(const double Btilde[N][N], double tau,
                                double *max_row_err, double *max_col_err) {
    return sinkhorn_fwd_fp16_n(Btilde, tau, TMAX, max_row_err, max_col_err);
}

static double frand(void) { return 2.0 * ((double)rand() / RAND_MAX) - 1.0; }

int main(void) {
    srand(7);
    printf("=== mHC Sinkhorn spike (N=%d, t_max=%d) ===\n\n", N, TMAX);

    // (1) fp16 doubly-stochasticity sweep over tau, on random logit matrices.
    printf("(1) fp16 doubly-stochasticity (max |rowsum-1|, |colsum-1| over 200 random 4x4):\n");
    double taus[] = {1.0, 0.5, 0.1, 0.05};
    for (int ti = 0; ti < 4; ti++) {
        double tau = taus[ti], worst = 0, wr = 0, wc = 0;
        for (int k = 0; k < 200; k++) {
            double Bt[N][N];
            for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) Bt[i][j] = frand();
            double r, c, e = sinkhorn_fwd_fp16(Bt, tau, &r, &c);
            if (e > worst) { worst = e; wr = r; wc = c; }
        }
        printf("   tau=%.2f : worst row_err=%.2e col_err=%.2e  %s\n",
               tau, wr, wc, worst < 1e-2 ? "OK (<1e-2)" : "NOT doubly-stochastic");
    }

    // (1b) For the stiff community default tau=0.05, how many iterations does fp16
    //      need to reach doubly-stochastic? (ADR open Q1: "report the needed t_max").
    printf("\n(1b) t_max needed at tau=0.05 (fp16, worst over 200 random 4x4):\n");
    int iters_list[] = {20, 40, 80, 160};
    for (int ii = 0; ii < 4; ii++) {
        int it = iters_list[ii]; double worst = 0;
        for (int k = 0; k < 200; k++) {
            double Bt[N][N];
            for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) Bt[i][j] = frand();
            double r, c, e = sinkhorn_fwd_fp16_n(Bt, 0.05, it, &r, &c);
            if (e > worst) worst = e;
        }
        printf("   t_max=%3d : worst err=%.2e  %s\n", it, worst,
               worst < 1e-2 ? "OK (<1e-2)" : "not yet");
    }

    // (2) unrolled backward vs central finite differences (double; loss = sum(W*B)).
    printf("\n(2) unrolled backward vs finite differences (loss = sum(W .* B), tau=0.1):\n");
    double Bt[N][N], W[N][N];
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) { Bt[i][j] = frand(); W[i][j] = frand(); }
    Tape T; sinkhorn_fwd(Bt, 0.1, &T);
    double dBtilde[N][N]; sinkhorn_bwd(&T, W, dBtilde);   // dLoss/dB = W
    double max_abs = 0, eps = 1e-5;
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
        double save = Bt[i][j];
        Bt[i][j] = save + eps; Tape Tp; sinkhorn_fwd(Bt, 0.1, &Tp);
        double lp = 0; for (int a = 0; a < N; a++) for (int b = 0; b < N; b++) lp += W[a][b] * Tp.B[a][b];
        Bt[i][j] = save - eps; Tape Tm; sinkhorn_fwd(Bt, 0.1, &Tm);
        double lm = 0; for (int a = 0; a < N; a++) for (int b = 0; b < N; b++) lm += W[a][b] * Tm.B[a][b];
        Bt[i][j] = save;
        double num = (lp - lm) / (2 * eps), d = fabs(num - dBtilde[i][j]);
        if (d > max_abs) max_abs = d;
    }
    printf("   max |analytic - numerical| = %.2e  %s\n", max_abs,
           max_abs < 1e-4 ? "OK (backward correct)" : "FAIL");

    // (3) behavioral overfit of an isolated mHC mixing block: learn Btilde so that
    //     Y = B(Btilde) @ X matches a reachable target B* @ X. Loss -> 0 confirms
    //     gradients flow correctly through the unrolled Sinkhorn.
    printf("\n(3) isolated mHC block overfit (Y = Sinkhorn(Btilde) @ X -> target):\n");
    srand(11);   // independent, reproducible instance
    int d = 3;
    double X[N][3], Bstar_t[N][N], target[N][3], Btr[N][N];
    for (int i = 0; i < N; i++) for (int k = 0; k < d; k++) X[i][k] = frand();
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) { Bstar_t[i][j] = frand(); Btr[i][j] = 0.1 * frand(); }
    Tape Ts; sinkhorn_fwd(Bstar_t, 0.1, &Ts);              // target B*
    for (int i = 0; i < N; i++) for (int k = 0; k < d; k++) {
        double s = 0; for (int j = 0; j < N; j++) s += Ts.B[i][j] * X[j][k]; target[i][k] = s;
    }
    double lr = 0.1, loss0 = 0, loss = 0;
    for (int step = 0; step < 20000; step++) {
        Tape Tt; sinkhorn_fwd(Btr, 0.1, &Tt);
        double Y[N][3], dB[N][N] = {{0}}; loss = 0;
        for (int i = 0; i < N; i++) for (int k = 0; k < d; k++) {
            double s = 0; for (int j = 0; j < N; j++) s += Tt.B[i][j] * X[j][k];
            Y[i][k] = s; double e = s - target[i][k]; loss += e * e;
            for (int j = 0; j < N; j++) dB[i][j] += 2 * e * X[j][k];   // dLoss/dB
        }
        if (step == 0) loss0 = loss;
        double dBt[N][N]; sinkhorn_bwd(&Tt, dB, dBt);
        for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) Btr[i][j] -= lr * dBt[i][j];
    }
    printf("   loss %.4e -> %.4e (%.0fx)  %s\n", loss0, loss, loss0 / (loss + 1e-30),
           loss < 1e-5 ? "OK (overfit; unrolled backward trains)" : "did not collapse");

    return 0;
}
