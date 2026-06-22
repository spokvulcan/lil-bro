// test_elo.c — pure-C gate for the Bradley-Terry / Elo solver (ADR 0007). No ANE, no net.
//
// Oracle: three players with Bradley-Terry strengths gamma = (9, 3, 1). That reproduces
// EXACTLY P(A>B)=9/12=0.75, P(B>C)=3/4=0.75, P(A>C)=9/10=0.90. So a correct fit must
// recover Elo = 400*log10(gamma) = (381.70, 190.85, 0) up to the anchor shift. We feed the
// solver round-robin counts consistent with those probabilities and assert it inverts them.
#include "elo.h"
#include <stdio.h>
#include <math.h>

static int fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { printf("  FAIL: " __VA_ARGS__); printf("\n"); fail = 1; } } while (0)

int main(void) {
    printf("## test_elo — Bradley-Terry MLE recovers a known Elo ladder\n");

    enum { K = 3 };
    const int A = 0, B = 1, C = 2;
    double wins[K * K]  = {0};
    double games[K * K] = {0};

    // 100 games per pair, split to match the gamma=(9,3,1) win probabilities exactly.
    // A vs B: 75/25 ; B vs C: 75/25 ; A vs C: 90/10. No draws in the oracle.
    games[A*K+B] = games[B*K+A] = 100; wins[A*K+B] = 75; wins[B*K+A] = 25;
    games[B*K+C] = games[C*K+B] = 100; wins[B*K+C] = 75; wins[C*K+B] = 25;
    games[A*K+C] = games[C*K+A] = 100; wins[A*K+C] = 90; wins[C*K+A] = 10;

    double elo[K];
    int iters = chess_elo_fit(K, wins, games, /*anchor=*/C, elo, /*max_iters=*/1000, /*tol=*/1e-12);
    printf("   converged in %d MM iters\n", iters);
    printf("   Elo: A=%.2f  B=%.2f  C=%.2f  (anchor C=0)\n", elo[A], elo[B], elo[C]);

    CHECK(iters >= 1, "solver did not run");
    CHECK(fabs(elo[C] - 0.0) < 1e-6, "anchor C must be exactly 0, got %.4f", elo[C]);
    CHECK(fabs(elo[B] - 190.85) < 1.0, "B Elo should be ~190.85, got %.4f", elo[B]);
    CHECK(fabs(elo[A] - 381.70) < 1.0, "A Elo should be ~381.70, got %.4f", elo[A]);
    CHECK(elo[A] > elo[B] && elo[B] > elo[C], "ordering A>B>C broken");

    // Round-trip: the fitted Elo must predict the input win-rates back.
    double pAB = chess_elo_expected(elo[A], elo[B]);
    double pBC = chess_elo_expected(elo[B], elo[C]);
    double pAC = chess_elo_expected(elo[A], elo[C]);
    printf("   predicted P(A>B)=%.3f P(B>C)=%.3f P(A>C)=%.3f (oracle 0.75/0.75/0.90)\n", pAB, pBC, pAC);
    CHECK(fabs(pAB - 0.75) < 0.01, "P(A>B) round-trip off: %.4f", pAB);
    CHECK(fabs(pBC - 0.75) < 0.01, "P(B>C) round-trip off: %.4f", pBC);
    CHECK(fabs(pAC - 0.90) < 0.01, "P(A>C) round-trip off: %.4f", pAC);

    // Draws-as-half-point: an all-draw round-robin must yield equal Elo (0 everywhere).
    {
        double w2[K*K] = {0}, g2[K*K] = {0};
        for (int i = 0; i < K; i++) for (int j = 0; j < K; j++) if (i != j) { g2[i*K+j] = 40; w2[i*K+j] = 20; }
        double e2[K];
        chess_elo_fit(K, w2, g2, A, e2, 1000, 1e-12);
        CHECK(fabs(e2[A]) < 1e-6 && fabs(e2[B]) < 1.0 && fabs(e2[C]) < 1.0,
              "all-draws should be ~equal Elo, got %.3f/%.3f/%.3f", e2[A], e2[B], e2[C]);
    }

    printf(fail ? "\n*** test_elo FAILED ***\n" : "\nall elo gates passed\n");
    return fail;
}
