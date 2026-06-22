// elo.c — Bradley-Terry MLE (MM algorithm) for self-anchored Elo. See elo.h.
#include "elo.h"
#include <math.h>
#include <stdlib.h>

double chess_elo_expected(double Ra, double Rb) {
    return 1.0 / (1.0 + pow(10.0, (Rb - Ra) / 400.0));
}

// MM update (Hunter 2004): gamma_i <- W_i / sum_{j!=i} N_ij/(gamma_i+gamma_j), where gamma
// = 10^(Elo/400), W_i = total effective wins of i, N_ij = total games i vs j. Monotonically
// increases the Bradley-Terry likelihood; we normalize by the geometric mean each sweep to
// stop the overall scale drifting, and shift to anchor==0 at the end.
int chess_elo_fit(int K, const double *wins, const double *games,
                  int anchor, double *elo_out, int max_iters, double tol) {
    if (K < 2 || !wins || !games || !elo_out || anchor < 0 || anchor >= K) return -1;
    if (max_iters < 1) max_iters = 1;
    if (tol <= 0.0) tol = 1e-9;

    const double GAMMA_FLOOR = 1e-9;   // a player with 0 wins -> ~ -3600 Elo (clamped, not -inf)
    double *gamma = (double*)malloc((size_t)K * sizeof(double));
    double *ng    = (double*)malloc((size_t)K * sizeof(double));
    double *W     = (double*)malloc((size_t)K * sizeof(double));
    if (!gamma || !ng || !W) { free(gamma); free(ng); free(W); return -1; }

    for (int i = 0; i < K; i++) {
        gamma[i] = 1.0;
        double wi = 0.0;
        for (int j = 0; j < K; j++) if (j != i) wi += wins[i * K + j];
        W[i] = wi;
    }

    int iter = 0;
    for (; iter < max_iters; iter++) {
        for (int i = 0; i < K; i++) {
            double denom = 0.0;
            for (int j = 0; j < K; j++) {
                if (j == i) continue;
                double nij = games[i * K + j];
                if (nij > 0.0) denom += nij / (gamma[i] + gamma[j]);
            }
            double g = (W[i] > 0.0 && denom > 0.0) ? (W[i] / denom) : GAMMA_FLOOR;
            if (g < GAMMA_FLOOR) g = GAMMA_FLOOR;
            ng[i] = g;
        }
        // normalize by the geometric mean (sum of logs == 0) to pin the overall scale
        double mean_log = 0.0;
        for (int i = 0; i < K; i++) mean_log += log(ng[i]);
        mean_log /= (double)K;
        double scale = exp(-mean_log);
        double max_delta = 0.0;
        for (int i = 0; i < K; i++) {
            double v = ng[i] * scale;
            double d = fabs(log(v) - log(gamma[i]));
            if (d > max_delta) max_delta = d;
            gamma[i] = v;
        }
        if (max_delta < tol) { iter++; break; }
    }

    double anchor_elo = 400.0 * log10(gamma[anchor]);
    for (int i = 0; i < K; i++) elo_out[i] = 400.0 * log10(gamma[i]) - anchor_elo;

    free(gamma); free(ng); free(W);
    return iter;
}
