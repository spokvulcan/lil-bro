// elo.h — self-anchored Elo from a round-robin of net checkpoints (ADR 0007).
//
// The "infinite-learning" SLOPE metric (chess CONTEXT.md): a relative Elo curve over the
// net's OWN past checkpoints. A fixed greedy bot saturates ~1000 Elo and cannot measure
// further climb; self-anchored Elo is unbounded — it always answers "is the latest net
// stronger than its past selves?". Strength is fit by Bradley-Terry maximum-likelihood
// (the standard Elo-from-results model), solved by the MM algorithm (Hunter 2004), which
// is monotone and provably convergent. Draws are scored as half a win to each side (the
// standard half-point approximation; a Davidson draw model is a later refinement).
//
// Pure C, zero deps (only -lm). Evaluator-agnostic: the round-robin that fills `wins`/
// `games` is net-vs-net (selfplay.c match_net_vs_net), but the math here knows nothing of
// chess — it is unit-tested against a hand-computed oracle (chess/test_elo.c, no ANE).
#ifndef LILBRO_CHESS_ELO_H
#define LILBRO_CHESS_ELO_H

// Fit relative Elo for K players from a round-robin.
//   wins  : K*K row-major; wins[i*K+j] = effective wins of i over j (W_ij + 0.5*D_ij).
//   games : K*K row-major; games[i*K+j] = total games played between i and j (== games[j*K+i]).
//           Diagonal entries are ignored. The caller guarantees
//           wins[i*K+j] + wins[j*K+i] == games[i*K+j].
//   anchor: the player index pinned to 0 Elo (the oldest checkpoint, so the curve climbs
//           from 0). Must be in [0,K).
//   elo_out: filled with K Elo values, elo_out[anchor] == 0.
// Returns the number of MM iterations run (>=1), or -1 on a bad argument.
int chess_elo_fit(int K, const double *wins, const double *games,
                  int anchor, double *elo_out, int max_iters, double tol);

// Expected score of player A vs player B under the Elo model: 1/(1+10^((Rb-Ra)/400)).
// (Exposed for tests + reporting.)
double chess_elo_expected(double Ra, double Rb);

#endif  // LILBRO_CHESS_ELO_H
