// perft.c — the perft correctness gate for the bitboard engine (ADR 0005, issue #15).
//
// perft enumerates the move tree to a fixed depth and counts leaf nodes. Matching the
// published reference counts EXACTLY is the gate: one wrong movegen edge case changes a
// count, so a green perft is strong evidence the rules are correct. Reference counts are
// the canonical Chess Programming Wiki "Perft Results" values.
//
// Usage:
//   ./perft                      run the full published-count gate (default)
//   ./perft quick                run a shallow, fast subset (dev iteration)
//   ./perft "<fen>" <depth>      print perft(depth) for one position
//   ./perft divide "<fen>" <d>   per-root-move subtree counts (localize a mismatch)

#include "chess.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

typedef struct { const char *name; const char *fen; int depth; uint64_t expected; } Case;

// Canonical CPW perft positions/counts. The two required by issue #15 are flagged.
static const Case FULL[] = {
    { "startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",            6, 119060324ull }, // required
    { "kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 5, 193690690ull }, // required
    { "position3","8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",                            6, 11030083ull  },
    { "position4","r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",     5, 15833292ull  },
    { "position5","rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",            5, 89941194ull  },
    { "position6","r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 5, 164075551ull },
};

static const Case QUICK[] = {
    { "startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",            5, 4865609ull },
    { "kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4085603ull },
    { "position3","8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",                            5, 674624ull  },
    { "position4","r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",     4, 422333ull  },
    { "position5","rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",            4, 2103487ull },
    { "position6","r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 4, 3894594ull },
};

static int run_suite(const Case *cases, int ncases) {
    int failed = 0;
    uint64_t total_nodes = 0;
    double total_t = 0;
    printf("%-10s %-6s %16s %16s %10s  %s\n", "position", "depth", "got", "expected", "Mnps", "result");
    for (int i = 0; i < ncases; i++) {
        Position p;
        if (!chess_from_fen(&p, cases[i].fen)) { printf("  bad FEN: %s\n", cases[i].fen); failed++; continue; }
        double t0 = now_sec();
        uint64_t got = chess_perft(&p, cases[i].depth);
        double dt = now_sec() - t0;
        total_nodes += got; total_t += dt;
        bool ok = (got == cases[i].expected);
        if (!ok) failed++;
        printf("%-10s %-6d %16llu %16llu %10.1f  %s\n", cases[i].name, cases[i].depth,
               (unsigned long long)got, (unsigned long long)cases[i].expected,
               dt > 0 ? got / dt / 1e6 : 0.0, ok ? "PASS" : "*** FAIL ***");
    }
    printf("--------------------------------------------------------------------------\n");
    printf("%llu nodes in %.2fs  (%.1f Mnps)   %s\n",
           (unsigned long long)total_nodes, total_t,
           total_t > 0 ? total_nodes / total_t / 1e6 : 0.0,
           failed ? "GATE RED" : "GATE GREEN");
    return failed;
}

// perft-divide: per-root-move subtree node counts. Sum must equal perft(depth).
static void divide(Position *p, int depth) {
    Move moves[MAX_MOVES];
    int n = chess_legal_moves(p, moves);
    uint64_t total = 0;
    char buf[8];
    for (int i = 0; i < n; i++) {
        Undo u;
        chess_make(p, moves[i], &u);
        uint64_t c = (depth <= 1) ? 1 : chess_perft(p, depth - 1);
        chess_unmake(p, moves[i], &u);
        printf("%s: %llu\n", chess_move_to_uci(moves[i], buf), (unsigned long long)c);
        total += c;
    }
    printf("\nmoves: %d\nnodes: %llu\n", n, (unsigned long long)total);
}

int main(int argc, char **argv) {
    chess_init();

    if (argc == 1)                              return run_suite(FULL, (int)(sizeof FULL / sizeof FULL[0]));
    if (argc == 2 && !strcmp(argv[1], "quick")) return run_suite(QUICK, (int)(sizeof QUICK / sizeof QUICK[0]));

    if (argc == 4 && !strcmp(argv[1], "divide")) {
        Position p;
        if (!chess_from_fen(&p, argv[2])) { fprintf(stderr, "bad FEN\n"); return 2; }
        divide(&p, atoi(argv[3]));
        return 0;
    }
    if (argc == 3) {
        Position p;
        if (!chess_from_fen(&p, argv[1])) { fprintf(stderr, "bad FEN\n"); return 2; }
        int depth = atoi(argv[2]);
        double t0 = now_sec();
        uint64_t got = chess_perft(&p, depth);
        double dt = now_sec() - t0;
        printf("perft(%d) = %llu   (%.2fs, %.1f Mnps)\n", depth, (unsigned long long)got, dt,
               dt > 0 ? got / dt / 1e6 : 0.0);
        return 0;
    }
    fprintf(stderr, "usage: %s [quick | \"<fen>\" <depth> | divide \"<fen>\" <depth>]\n", argv[0]);
    return 2;
}
