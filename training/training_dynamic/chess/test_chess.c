// test_chess.c — unit tests for the bitboard engine (issue #15).
//
// Covers the acceptance criteria that perft alone does not: make/unmake exact restore
// (bytewise, at every node), position<->token round-trip, and the move<->(8x8x73) index
// map being a bijection over legal moves. Anchored index values lock the policy-encoding
// contract that issue #16 (policy head + legal mask) will consume. Pattern follows the
// existing standalone tests (test_attn_cpu.c): print per-check OK/FAIL, return nonzero on
// any failure.

#include "chess.h"
#include <stdio.h>
#include <string.h>

static int g_fail = 0;
static void check(int cond, const char *name) {
    printf("  %-48s %s\n", name, cond ? "OK" : "*** FAIL ***");
    if (!cond) g_fail++;
}

// ---- FEN round-trip ---------------------------------------------------------
static void test_fen(void) {
    printf("[FEN round-trip]\n");
    const char *fens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    };
    char buf[96];
    for (int i = 0; i < 4; i++) {
        Position p;
        check(chess_from_fen(&p, fens[i]), "parse");
        check(strcmp(chess_to_fen(&p, buf), fens[i]) == 0, "re-emit equals input");
    }
}

// ---- make/unmake exact restore (bytewise, every node) -----------------------
static int verify_fail = 0;
static uint64_t perft_verify(Position *p, int depth) {
    if (depth == 0) return 1;
    Move moves[MAX_MOVES];
    int n = chess_legal_moves(p, moves);
    uint64_t nodes = 0;
    for (int i = 0; i < n; i++) {
        Position snap = *p;
        Undo u;
        chess_make(p, moves[i], &u);
        nodes += perft_verify(p, depth - 1);
        chess_unmake(p, moves[i], &u);
        if (memcmp(&snap, p, sizeof(Position)) != 0) verify_fail++;
    }
    return nodes;
}

static void test_make_unmake(void) {
    printf("[make/unmake exact restore + no-bulk perft cross-check]\n");
    struct { const char *fen; int depth; uint64_t expect; } cs[] = {
        { "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",            5, 4865609ull },
        { "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4085603ull },
        { "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",     4, 422333ull  },
    };
    for (int i = 0; i < 3; i++) {
        Position p;
        chess_from_fen(&p, cs[i].fen);
        verify_fail = 0;
        uint64_t got = perft_verify(&p, cs[i].depth);
        check(verify_fail == 0, "every make/unmake restored bytewise");
        check(got == cs[i].expect, "no-bulk perft equals published count");
    }
}

// ---- codec: position <-> tokens round-trip ----------------------------------
static void test_codec_position(void) {
    printf("[codec: position round-trip]\n");
    const char *fens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",  // en passant
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", // promo pawn, partial rights
        "8/8/8/8/8/8/6k1/4K2R w K - 7 42",                                  // clocks, lone rook
    };
    for (int i = 0; i < 5; i++) {
        Position p, q;
        chess_from_fen(&p, fens[i]);
        int16_t toks[CHESS_NUM_TOKENS];
        chess_encode(&p, toks);
        check(chess_decode(&q, toks), "decode ok");
        check(memcmp(&p, &q, sizeof(Position)) == 0, "decode(encode(pos)) == pos");
    }
}

// ---- codec: move <-> (8x8x73) index -----------------------------------------
static void test_move_index_bijection(void) {
    printf("[codec: move<->index bijection over legal moves]\n");
    const char *fens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",       // promotions + castle
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    };
    for (int i = 0; i < 4; i++) {
        Position p;
        chess_from_fen(&p, fens[i]);
        Move moves[MAX_MOVES];
        int n = chess_legal_moves(&p, moves);
        static uint8_t seen[CHESS_POLICY_SIZE];
        memset(seen, 0, sizeof seen);
        int in_range = 1, distinct = 1, invertible = 1;
        for (int j = 0; j < n; j++) {
            int idx = chess_move_to_index(moves[j]);
            if (idx < 0 || idx >= CHESS_POLICY_SIZE) { in_range = 0; continue; }
            if (seen[idx]) distinct = 0;
            seen[idx] = 1;
            if (chess_index_to_move(idx, moves, n) != moves[j]) invertible = 0;
        }
        check(in_range, "all indices in [0,4672)");
        check(distinct, "no two legal moves share an index");
        check(invertible, "index_to_move inverts move_to_index");
    }
}

// ---- codec: anchored index values (lock the contract for #16) ----------------
static void test_move_index_anchors(void) {
    printf("[codec: anchored policy-index values]\n");
    // queen-plane: e2e4 (N, dist 2) -> from 12, plane 1
    check(chess_move_to_index(move_make(12, 28, MF_DOUBLE_PUSH)) == 12 * 73 + 1, "e2e4 -> 877");
    // knight-planes: b1c3 (delta +2,+1 = plane 56), g1f3 (delta +2,-1 = plane 63)
    check(chess_move_to_index(move_make(1, 18, MF_QUIET)) == 1 * 73 + 56, "b1c3 -> 129");
    check(chess_move_to_index(move_make(6, 21, MF_QUIET)) == 6 * 73 + 63, "g1f3 -> 501");
    // castling via king queen-planes: O-O e1g1 (E dist2 = plane 15), O-O-O e1c1 (W dist2 = plane 43)
    check(chess_move_to_index(move_make(4, 6, MF_KCASTLE)) == 4 * 73 + 15, "O-O -> 307");
    check(chess_move_to_index(move_make(4, 2, MF_QCASTLE)) == 4 * 73 + 43, "O-O-O -> 335");
    // queen-promotion uses the queen plane: a7a8=Q -> plane 0
    check(chess_move_to_index(move_make(48, 56, MF_PROMO_Q)) == 48 * 73 + 0, "a7a8=Q -> 3504");
    // underpromotions: a7a8=N forward (u0,m0 -> plane 64); a7b8=R cap-right (u2,m2 -> plane 72)
    check(chess_move_to_index(move_make(48, 56, MF_PROMO_N)) == 48 * 73 + 64, "a7a8=N -> 3568");
    check(chess_move_to_index(move_make(48, 57, MF_PROMO_R_CAP)) == 48 * 73 + 72, "a7b8=R -> 3576");
}

// ---- shallow perft sanity (fast) --------------------------------------------
static void test_perft_shallow(void) {
    printf("[perft shallow sanity]\n");
    struct { const char *fen; int depth; uint64_t expect; } cs[] = {
        { "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",            1, 20ull },
        { "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",            3, 8902ull },
        { "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 1, 48ull },
        { "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3, 97862ull },
        { "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",                            4, 43238ull },
        { "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",            3, 62379ull },
        { "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 3, 89890ull },
    };
    char nm[64];
    for (int i = 0; i < 7; i++) {
        Position p;
        chess_from_fen(&p, cs[i].fen);
        uint64_t got = chess_perft(&p, cs[i].depth);
        snprintf(nm, sizeof nm, "perft d%d == %llu", cs[i].depth, (unsigned long long)cs[i].expect);
        check(got == cs[i].expect, nm);
    }
}

int main(void) {
    chess_init();
    test_fen();
    test_perft_shallow();
    test_make_unmake();
    test_codec_position();
    test_move_index_bijection();
    test_move_index_anchors();
    printf("\n%s (%d failure%s)\n", g_fail ? "TESTS FAILED" : "ALL TESTS PASSED",
           g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
