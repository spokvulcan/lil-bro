// chess.c — implementation of the dependency-free bitboard engine (see chess.h).
//
// Move generation is *fully legal* (not pseudo-legal + filter): we compute checkers,
// pinned pieces and their pin rays, and a king-danger map (king made transparent to
// sliders), then generate only moves that leave our king safe. En-passant — the one
// case a pin map cannot express cleanly (removing two pawns from a rank can expose a
// horizontal slider) — is validated by simulating the resulting occupancy and testing
// king safety directly. Slider attacks use classical ray scans (precomputed rays +
// bitscan): the most transparently-correct method, and perft is the external oracle.

#include "chess.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// ---- bit helpers ------------------------------------------------------------
static inline int lsb(uint64_t b)      { return __builtin_ctzll(b); }
static inline int msb(uint64_t b)      { return 63 - __builtin_clzll(b); }
static inline int popcount(uint64_t b) { return __builtin_popcountll(b); }
static inline int poplsb(uint64_t *b)  { int s = lsb(*b); *b &= *b - 1; return s; }
#define BIT(s) (1ull << (s))
// pawn "forward" delta (square index step toward promotion) for a color.
static inline int up_dir(int color) { return (color == WHITE) ? 8 : -8; }

// ---- directions (clockwise from North); (drank, dfile) ----------------------
enum { DIR_N, DIR_NE, DIR_E, DIR_SE, DIR_S, DIR_SW, DIR_W, DIR_NW };
static const int DR[8] = { +1, +1,  0, -1, -1, -1,  0, +1 };
static const int DF[8] = {  0, +1, +1, +1,  0, -1, -1, -1 };
// positive direction = square index increases (use lsb of blockers); else msb.
static const bool RAY_POS[8] = { true, true, true, false, false, false, false, true };

#define FILE_A 0x0101010101010101ull
#define FILE_H 0x8080808080808080ull
#define RANK_2 0x000000000000FF00ull
#define RANK_7 0x00FF000000000000ull
#define RANK_1 0x00000000000000FFull
#define RANK_8 0xFF00000000000000ull

// ---- precomputed tables -----------------------------------------------------
static uint64_t knight_att[64];
static uint64_t king_att[64];
static uint64_t pawn_att[2][64];
static uint64_t RAYS[8][64];
static uint64_t between_bb[64][64];   // squares strictly between a,b if colinear, else 0
static int      castle_mask[64];      // ANDed into castling rights when a square is touched
static bool     g_inited = false;

void chess_init(void) {
    if (g_inited) return;
    for (int sq = 0; sq < 64; sq++) {
        int r = sq >> 3, f = sq & 7;
        // knight
        static const int ndr[8] = { +2, +1, -1, -2, -2, -1, +1, +2 };
        static const int ndf[8] = { +1, +2, +2, +1, -1, -2, -2, -1 };
        uint64_t kn = 0;
        for (int i = 0; i < 8; i++) {
            int rr = r + ndr[i], ff = f + ndf[i];
            if (rr >= 0 && rr < 8 && ff >= 0 && ff < 8) kn |= BIT(rr * 8 + ff);
        }
        knight_att[sq] = kn;
        // king
        uint64_t kg = 0;
        for (int dr = -1; dr <= 1; dr++)
            for (int df = -1; df <= 1; df++) {
                if (!dr && !df) continue;
                int rr = r + dr, ff = f + df;
                if (rr >= 0 && rr < 8 && ff >= 0 && ff < 8) kg |= BIT(rr * 8 + ff);
            }
        king_att[sq] = kg;
        // pawn attacks
        uint64_t wp = 0, bp = 0;
        if (r + 1 < 8) { if (f - 1 >= 0) wp |= BIT((r + 1) * 8 + f - 1); if (f + 1 < 8) wp |= BIT((r + 1) * 8 + f + 1); }
        if (r - 1 >= 0) { if (f - 1 >= 0) bp |= BIT((r - 1) * 8 + f - 1); if (f + 1 < 8) bp |= BIT((r - 1) * 8 + f + 1); }
        pawn_att[WHITE][sq] = wp;
        pawn_att[BLACK][sq] = bp;
        // rays
        for (int d = 0; d < 8; d++) {
            uint64_t ray = 0;
            int rr = r + DR[d], ff = f + DF[d];
            while (rr >= 0 && rr < 8 && ff >= 0 && ff < 8) { ray |= BIT(rr * 8 + ff); rr += DR[d]; ff += DF[d]; }
            RAYS[d][sq] = ray;
        }
    }
    // between_bb: walk each ray, recording the path so far as the "between" set.
    memset(between_bb, 0, sizeof between_bb);
    for (int a = 0; a < 64; a++) {
        int r = a >> 3, f = a & 7;
        for (int d = 0; d < 8; d++) {
            uint64_t path = 0;
            int rr = r + DR[d], ff = f + DF[d];
            while (rr >= 0 && rr < 8 && ff >= 0 && ff < 8) {
                int b = rr * 8 + ff;
                between_bb[a][b] = path;     // squares strictly between a and b
                path |= BIT(b);
                rr += DR[d]; ff += DF[d];
            }
        }
    }
    // castling rights mask: rights removed when from/to touches a key square.
    for (int s = 0; s < 64; s++) castle_mask[s] = 0xF;
    castle_mask[0]  &= ~CASTLE_WQ;                 // A1
    castle_mask[7]  &= ~CASTLE_WK;                 // H1
    castle_mask[4]  &= ~(CASTLE_WK | CASTLE_WQ);   // E1
    castle_mask[56] &= ~CASTLE_BQ;                 // A8
    castle_mask[63] &= ~CASTLE_BK;                 // H8
    castle_mask[60] &= ~(CASTLE_BK | CASTLE_BQ);   // E8
    g_inited = true;
}

// ---- slider attacks (classical ray scan) ------------------------------------
static uint64_t ray_attack(int dir, int sq, uint64_t occ) {
    uint64_t att = RAYS[dir][sq];
    uint64_t blk = att & occ;
    if (blk) { int b = RAY_POS[dir] ? lsb(blk) : msb(blk); att ^= RAYS[dir][b]; }
    return att;
}
static uint64_t bishop_attacks(int sq, uint64_t occ) {
    return ray_attack(DIR_NE, sq, occ) | ray_attack(DIR_SE, sq, occ)
         | ray_attack(DIR_SW, sq, occ) | ray_attack(DIR_NW, sq, occ);
}
static uint64_t rook_attacks(int sq, uint64_t occ) {
    return ray_attack(DIR_N, sq, occ) | ray_attack(DIR_E, sq, occ)
         | ray_attack(DIR_S, sq, occ) | ray_attack(DIR_W, sq, occ);
}
static uint64_t slider_attacks(int pt, int sq, uint64_t occ) {
    if (pt == BISHOP) return bishop_attacks(sq, occ);
    if (pt == ROOK)   return rook_attacks(sq, occ);
    return bishop_attacks(sq, occ) | rook_attacks(sq, occ);   // QUEEN
}

// ---- occupancy / piece lookup ----------------------------------------------
static inline uint64_t color_occ(const Position *p, int c) {
    return p->bb[c][0] | p->bb[c][1] | p->bb[c][2] | p->bb[c][3] | p->bb[c][4] | p->bb[c][5];
}
uint64_t chess_color_occupancy(const Position *p, int c) { return color_occ(p, c); }
uint64_t chess_occupancy(const Position *p) { return color_occ(p, WHITE) | color_occ(p, BLACK); }

static int piece_on_color(const Position *p, int sq, int c) {
    uint64_t b = BIT(sq);
    for (int pt = PAWN; pt <= KING; pt++) if (p->bb[c][pt] & b) return pt;
    return NO_PIECE_TYPE;
}

// ---- attack queries ---------------------------------------------------------
static bool square_attacked(const Position *p, int sq, int by, uint64_t occ) {
    int notby = by ^ 1;
    if (pawn_att[notby][sq] & p->bb[by][PAWN])   return true;
    if (knight_att[sq]      & p->bb[by][KNIGHT]) return true;
    if (king_att[sq]        & p->bb[by][KING])   return true;
    if (bishop_attacks(sq, occ) & (p->bb[by][BISHOP] | p->bb[by][QUEEN])) return true;
    if (rook_attacks(sq, occ)   & (p->bb[by][ROOK]   | p->bb[by][QUEEN])) return true;
    return false;
}

// All squares attacked by `by`, given occupancy `occ` (caller passes king-removed occ
// for king-danger so sliders x-ray through our king's current square).
static uint64_t attack_map(const Position *p, int by, uint64_t occ) {
    uint64_t att = 0, pw = p->bb[by][PAWN];
    if (by == WHITE) att |= ((pw & ~FILE_A) << 7) | ((pw & ~FILE_H) << 9);
    else             att |= ((pw & ~FILE_A) >> 9) | ((pw & ~FILE_H) >> 7);
    uint64_t kn = p->bb[by][KNIGHT]; while (kn) att |= knight_att[poplsb(&kn)];
    att |= king_att[lsb(p->bb[by][KING])];
    uint64_t bq = p->bb[by][BISHOP] | p->bb[by][QUEEN]; while (bq) att |= bishop_attacks(poplsb(&bq), occ);
    uint64_t rq = p->bb[by][ROOK]   | p->bb[by][QUEEN]; while (rq) att |= rook_attacks(poplsb(&rq), occ);
    return att;
}

bool chess_in_check(const Position *p) {
    int us = p->side, ksq = lsb(p->bb[us][KING]);
    return square_attacked(p, ksq, us ^ 1, chess_occupancy(p));
}

// True iff our king would be in check after an en-passant capture from `from` to `ep`
// removing the pawn at `capsq`. occ2 is the post-capture occupancy.
static bool ep_exposes_king(const Position *p, int ksq, int them, uint64_t occ2, int capsq) {
    int us = them ^ 1;
    uint64_t their_pawns = p->bb[them][PAWN] & ~BIT(capsq);
    if (pawn_att[us][ksq] & their_pawns)      return true;   // (captured pawn excluded)
    if (knight_att[ksq] & p->bb[them][KNIGHT]) return true;  // occupancy-independent
    if (king_att[ksq]   & p->bb[them][KING])   return true;
    if (bishop_attacks(ksq, occ2) & (p->bb[them][BISHOP] | p->bb[them][QUEEN])) return true;
    if (rook_attacks(ksq, occ2)   & (p->bb[them][ROOK]   | p->bb[them][QUEEN])) return true;
    return false;
}

// ---- legal move generation --------------------------------------------------
static int emit_pawn(Move *out, int n, int from, int to, bool capture, bool promo) {
    if (promo) {
        int base = capture ? MF_PROMO_N_CAP : MF_PROMO_N;   // +0..+3 = N,B,R,Q
        out[n++] = move_make(from, to, base + 0);
        out[n++] = move_make(from, to, base + 1);
        out[n++] = move_make(from, to, base + 2);
        out[n++] = move_make(from, to, base + 3);
    } else {
        out[n++] = move_make(from, to, capture ? MF_CAPTURE : MF_QUIET);
    }
    return n;
}

int chess_legal_moves(const Position *p, Move *out) {
    int n = 0;
    int us = p->side, them = us ^ 1;
    uint64_t own = color_occ(p, us), opp = color_occ(p, them), occ = own | opp;
    int ksq = lsb(p->bb[us][KING]);

    // checkers attacking our king
    uint64_t checkers =
        (knight_att[ksq] & p->bb[them][KNIGHT]) |
        (pawn_att[us][ksq] & p->bb[them][PAWN]) |
        (bishop_attacks(ksq, occ) & (p->bb[them][BISHOP] | p->bb[them][QUEEN])) |
        (rook_attacks(ksq, occ)   & (p->bb[them][ROOK]   | p->bb[them][QUEEN]));
    int ncheck = popcount(checkers);

    // king moves: king transparent to sliders so it can't slide along a check ray
    uint64_t danger = attack_map(p, them, occ ^ BIT(ksq));
    uint64_t kmoves = king_att[ksq] & ~own & ~danger;
    while (kmoves) { int to = poplsb(&kmoves);
        out[n++] = move_make(ksq, to, (opp & BIT(to)) ? MF_CAPTURE : MF_QUIET); }

    // castling (only when not in check; path squares tested with real occupancy)
    if (ncheck == 0) {
        if (us == WHITE) {
            if ((p->castling & CASTLE_WK) && !(occ & (BIT(5) | BIT(6)))
                && !square_attacked(p, 5, them, occ) && !square_attacked(p, 6, them, occ))
                out[n++] = move_make(4, 6, MF_KCASTLE);
            if ((p->castling & CASTLE_WQ) && !(occ & (BIT(1) | BIT(2) | BIT(3)))
                && !square_attacked(p, 3, them, occ) && !square_attacked(p, 2, them, occ))
                out[n++] = move_make(4, 2, MF_QCASTLE);
        } else {
            if ((p->castling & CASTLE_BK) && !(occ & (BIT(61) | BIT(62)))
                && !square_attacked(p, 61, them, occ) && !square_attacked(p, 62, them, occ))
                out[n++] = move_make(60, 62, MF_KCASTLE);
            if ((p->castling & CASTLE_BQ) && !(occ & (BIT(57) | BIT(58) | BIT(59)))
                && !square_attacked(p, 59, them, occ) && !square_attacked(p, 58, them, occ))
                out[n++] = move_make(60, 58, MF_QCASTLE);
        }
    }

    if (ncheck == 2) return n;   // double check: only the king may move

    // capture/block target mask for non-king pieces
    uint64_t target;
    if (ncheck == 1) { int csq = lsb(checkers); target = between_bb[ksq][csq] | checkers; }
    else             { target = ~own; }

    // pinned pieces and their allowed-destination rays
    uint64_t pinned = 0;
    uint64_t pin_ray[64];
    uint64_t rq = p->bb[them][ROOK] | p->bb[them][QUEEN];
    uint64_t bq = p->bb[them][BISHOP] | p->bb[them][QUEEN];
    // snipers: enemy sliders aligned with king through exactly our blockers (use only
    // enemy pieces as blockers so our pieces are the potential pinned ones).
    uint64_t snipers = (rook_attacks(ksq, opp) & rq) | (bishop_attacks(ksq, opp) & bq);
    while (snipers) {
        int ssq = poplsb(&snipers);
        uint64_t betw = between_bb[ksq][ssq] & own;
        if (popcount(betw) == 1) {
            int psq = lsb(betw);
            pinned |= BIT(psq);
            pin_ray[psq] = between_bb[ksq][ssq] | BIT(ssq);   // slide along pin or capture pinner
        }
    }

    // knights (a pinned knight can never move legally)
    uint64_t knights = p->bb[us][KNIGHT] & ~pinned;
    while (knights) { int from = poplsb(&knights);
        uint64_t att = knight_att[from] & target;
        while (att) { int to = poplsb(&att);
            out[n++] = move_make(from, to, (opp & BIT(to)) ? MF_CAPTURE : MF_QUIET); } }

    // bishops / rooks / queens
    for (int pt = BISHOP; pt <= QUEEN; pt++) {
        uint64_t bbp = p->bb[us][pt];
        while (bbp) { int from = poplsb(&bbp);
            uint64_t att = slider_attacks(pt, from, occ) & target;
            if (pinned & BIT(from)) att &= pin_ray[from];
            while (att) { int to = poplsb(&att);
                out[n++] = move_make(from, to, (opp & BIT(to)) ? MF_CAPTURE : MF_QUIET); } }
    }

    // pawns
    int up = (us == WHITE) ? 8 : -8;
    uint64_t promo_rank = (us == WHITE) ? RANK_8 : RANK_1;
    uint64_t start_rank = (us == WHITE) ? RANK_2 : RANK_7;
    uint64_t empty = ~occ;
    uint64_t pawns = p->bb[us][PAWN];
    while (pawns) {
        int from = poplsb(&pawns);
        uint64_t frbit = BIT(from);
        uint64_t pr = (pinned & frbit) ? pin_ray[from] : ~0ull;
        // single / double push
        int one = from + up;
        if (empty & BIT(one)) {
            if ((target & BIT(one)) & pr)
                n = emit_pawn(out, n, from, one, false, (promo_rank & BIT(one)) != 0);
            if (start_rank & frbit) {
                int two = one + up;
                if ((empty & BIT(two)) && ((target & BIT(two)) & pr))
                    out[n++] = move_make(from, two, MF_DOUBLE_PUSH);
            }
        }
        // captures (incl. promotion-captures)
        uint64_t caps = pawn_att[us][from] & opp & target & pr;
        while (caps) { int to = poplsb(&caps);
            n = emit_pawn(out, n, from, to, true, (promo_rank & BIT(to)) != 0); }
    }

    // en passant — validated by simulating the resulting occupancy (covers the EP-pin
    // edge case and check resolution; rare, so the per-move king test is cheap).
    if (p->ep_square != NO_SQUARE) {
        int ep = p->ep_square;
        uint64_t cap = pawn_att[them][ep] & p->bb[us][PAWN];   // our pawns attacking ep
        while (cap) {
            int from = poplsb(&cap);
            int capsq = ep - up;                               // the pawn that double-pushed
            uint64_t occ2 = (occ ^ BIT(from) ^ BIT(capsq)) | BIT(ep);
            if (!ep_exposes_king(p, ksq, them, occ2, capsq))
                out[n++] = move_make(from, ep, MF_EP);
        }
    }
    return n;
}

// ---- make / unmake ----------------------------------------------------------
void chess_make(Position *p, Move m, Undo *u) {
    int from = move_from(m), to = move_to(m), fl = move_flag(m);
    int us = p->side, them = us ^ 1;
    uint64_t frbit = BIT(from), tobit = BIT(to);

    u->captured = NO_PIECE_TYPE;
    u->ep_square = p->ep_square;
    u->castling = p->castling;
    u->halfmove = p->halfmove;
    u->fullmove = p->fullmove;

    int moving = piece_on_color(p, from, us);
    p->ep_square = NO_SQUARE;
    p->halfmove++;

    // capture (non-EP): remove the enemy piece on `to`
    if (move_is_capture(m) && fl != MF_EP) {
        int cap = piece_on_color(p, to, them);
        u->captured = cap;
        p->bb[them][cap] ^= tobit;
        p->halfmove = 0;
    }

    // move the moving piece (promotion replaces pawn with the promoted piece)
    p->bb[us][moving] ^= frbit;
    if (move_is_promo(m)) { p->bb[us][move_promo_piece(m)] ^= tobit; p->halfmove = 0; }
    else                  { p->bb[us][moving] ^= tobit; }
    if (moving == PAWN) p->halfmove = 0;

    switch (fl) {
        case MF_DOUBLE_PUSH: p->ep_square = from + up_dir(us); break;
        case MF_EP: { int capsq = to - up_dir(us); p->bb[them][PAWN] ^= BIT(capsq); } break;
        case MF_KCASTLE: { int rf = (us == WHITE) ? 7 : 63, rt = (us == WHITE) ? 5 : 61;
                           p->bb[us][ROOK] ^= BIT(rf) | BIT(rt); } break;
        case MF_QCASTLE: { int rf = (us == WHITE) ? 0 : 56, rt = (us == WHITE) ? 3 : 59;
                           p->bb[us][ROOK] ^= BIT(rf) | BIT(rt); } break;
        default: break;
    }

    p->castling &= castle_mask[from] & castle_mask[to];
    if (us == BLACK) p->fullmove++;
    p->side = them;
}

void chess_unmake(Position *p, Move m, const Undo *u) {
    int us = p->side ^ 1;   // the side that made the move
    p->side = us;
    int from = move_from(m), to = move_to(m), fl = move_flag(m);
    uint64_t frbit = BIT(from), tobit = BIT(to);

    if (move_is_promo(m)) {
        p->bb[us][move_promo_piece(m)] ^= tobit;   // remove promoted piece
        p->bb[us][PAWN] ^= frbit;                  // restore the pawn
    } else {
        int moving = piece_on_color(p, to, us);
        p->bb[us][moving] ^= tobit | frbit;        // slide back to origin
    }

    if (fl == MF_EP) {
        int capsq = to - up_dir(us);
        p->bb[us ^ 1][PAWN] ^= BIT(capsq);
    } else if (u->captured != NO_PIECE_TYPE) {
        p->bb[us ^ 1][u->captured] ^= tobit;
    }

    if (fl == MF_KCASTLE)      { int rf = (us == WHITE) ? 7 : 63, rt = (us == WHITE) ? 5 : 61;
                                 p->bb[us][ROOK] ^= BIT(rf) | BIT(rt); }
    else if (fl == MF_QCASTLE) { int rf = (us == WHITE) ? 0 : 56, rt = (us == WHITE) ? 3 : 59;
                                 p->bb[us][ROOK] ^= BIT(rf) | BIT(rt); }

    p->castling = u->castling;
    p->ep_square = u->ep_square;
    p->halfmove = u->halfmove;
    p->fullmove = u->fullmove;
}

// ---- perft ------------------------------------------------------------------
uint64_t chess_perft(Position *p, int depth) {
    if (depth == 0) return 1;
    Move moves[MAX_MOVES];
    int n = chess_legal_moves(p, moves);
    if (depth == 1) return (uint64_t)n;   // bulk count
    uint64_t nodes = 0;
    Undo u;
    for (int i = 0; i < n; i++) {
        chess_make(p, moves[i], &u);
        nodes += chess_perft(p, depth - 1);
        chess_unmake(p, moves[i], &u);
    }
    return nodes;
}

// ---- FEN / start position / formatting --------------------------------------
void chess_startpos(Position *p) {
    chess_from_fen(p, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

bool chess_from_fen(Position *p, const char *fen) {
    memset(p, 0, sizeof *p);
    p->ep_square = NO_SQUARE;
    const char *s = fen;
    int rank = 7, file = 0;
    for (; *s && *s != ' '; s++) {
        char c = *s;
        if (c == '/') { rank--; file = 0; continue; }
        if (c >= '1' && c <= '8') { file += c - '0'; continue; }
        int color = (c >= 'a') ? BLACK : WHITE;
        int pt;
        switch (c) {
            case 'P': case 'p': pt = PAWN;   break;
            case 'N': case 'n': pt = KNIGHT; break;
            case 'B': case 'b': pt = BISHOP; break;
            case 'R': case 'r': pt = ROOK;   break;
            case 'Q': case 'q': pt = QUEEN;  break;
            case 'K': case 'k': pt = KING;   break;
            default: return false;
        }
        if (rank < 0 || rank > 7 || file < 0 || file > 7) return false;
        p->bb[color][pt] |= BIT(rank * 8 + file);
        file++;
    }
    while (*s == ' ') s++;
    p->side = (*s == 'b') ? BLACK : WHITE;
    while (*s && *s != ' ') s++;
    while (*s == ' ') s++;
    p->castling = 0;
    for (; *s && *s != ' '; s++) {
        switch (*s) {
            case 'K': p->castling |= CASTLE_WK; break;
            case 'Q': p->castling |= CASTLE_WQ; break;
            case 'k': p->castling |= CASTLE_BK; break;
            case 'q': p->castling |= CASTLE_BQ; break;
            default: break;   // '-' or others
        }
    }
    while (*s == ' ') s++;
    if (*s && *s != '-') {
        int f = s[0] - 'a', r = s[1] - '1';
        if (f >= 0 && f < 8 && r >= 0 && r < 8) p->ep_square = r * 8 + f;
        s += 2;
    } else if (*s == '-') s++;
    while (*s == ' ') s++;
    p->halfmove = 0; p->fullmove = 1;
    if (*s) { p->halfmove = atoi(s); while (*s && *s != ' ') s++; while (*s == ' ') s++; }
    if (*s) { p->fullmove = atoi(s); }
    if (p->fullmove < 1) p->fullmove = 1;
    return true;
}

char *chess_to_fen(const Position *p, char *buf) {
    char *o = buf;
    for (int rank = 7; rank >= 0; rank--) {
        int empty = 0;
        for (int file = 0; file < 8; file++) {
            int sq = rank * 8 + file, found = 0;
            for (int c = 0; c < 2 && !found; c++)
                for (int pt = PAWN; pt <= KING; pt++)
                    if (p->bb[c][pt] & BIT(sq)) {
                        if (empty) { *o++ = '0' + empty; empty = 0; }
                        const char *L = "PNBRQK";
                        char ch = L[pt];
                        *o++ = (c == BLACK) ? (ch + 32) : ch;
                        found = 1; break;
                    }
            if (!found) empty++;
        }
        if (empty) *o++ = '0' + empty;
        if (rank) *o++ = '/';
    }
    *o++ = ' '; *o++ = (p->side == WHITE) ? 'w' : 'b'; *o++ = ' ';
    if (!p->castling) *o++ = '-';
    else { if (p->castling & CASTLE_WK) *o++ = 'K'; if (p->castling & CASTLE_WQ) *o++ = 'Q';
           if (p->castling & CASTLE_BK) *o++ = 'k'; if (p->castling & CASTLE_BQ) *o++ = 'q'; }
    *o++ = ' ';
    if (p->ep_square == NO_SQUARE) *o++ = '-';
    else { *o++ = 'a' + (p->ep_square & 7); *o++ = '1' + (p->ep_square >> 3); }
    o += sprintf(o, " %d %d", p->halfmove, p->fullmove);
    *o = '\0';
    return buf;
}

char *chess_move_to_uci(Move m, char *buf) {
    int from = move_from(m), to = move_to(m);
    buf[0] = 'a' + (from & 7); buf[1] = '1' + (from >> 3);
    buf[2] = 'a' + (to & 7);   buf[3] = '1' + (to >> 3);
    if (move_is_promo(m)) { buf[4] = "..nbrq"[move_promo_piece(m)]; buf[5] = '\0'; }
    else buf[4] = '\0';
    return buf;
}

// ---- codec: position <-> tokens --------------------------------------------
void chess_encode(const Position *p, int16_t t[CHESS_NUM_TOKENS]) {
    for (int sq = 0; sq < 64; sq++) {
        int code = TOK_EMPTY;
        for (int c = 0; c < 2; c++)
            for (int pt = PAWN; pt <= KING; pt++)
                if (p->bb[c][pt] & BIT(sq)) code = chess_occupant_code(c, pt);
        t[sq] = (int16_t)code;
    }
    t[64] = (p->side == WHITE) ? TOK_STM_W : TOK_STM_B;
    t[65] = (p->castling & CASTLE_WK) ? TOK_FLAG_ON : TOK_FLAG_OFF;
    t[66] = (p->castling & CASTLE_WQ) ? TOK_FLAG_ON : TOK_FLAG_OFF;
    t[67] = (p->castling & CASTLE_BK) ? TOK_FLAG_ON : TOK_FLAG_OFF;
    t[68] = (p->castling & CASTLE_BQ) ? TOK_FLAG_ON : TOK_FLAG_OFF;
    if (p->ep_square == NO_SQUARE) { t[69] = TOK_EP_NONE; t[70] = TOK_EP_NONE; }
    else {
        t[69] = TOK_EP_FILE_A + (p->ep_square & 7);
        t[70] = ((p->ep_square >> 3) == 2) ? TOK_EP_RANK_3 : TOK_EP_RANK_6;
    }
    int hm = p->halfmove; if (hm < 0) hm = 0; if (hm > 999) hm = 999;
    t[71] = TOK_DIGIT_0 + (hm / 100) % 10; t[72] = TOK_DIGIT_0 + (hm / 10) % 10; t[73] = TOK_DIGIT_0 + hm % 10;
    int fm = p->fullmove; if (fm < 0) fm = 0; if (fm > 999) fm = 999;
    t[74] = TOK_DIGIT_0 + (fm / 100) % 10; t[75] = TOK_DIGIT_0 + (fm / 10) % 10; t[76] = TOK_DIGIT_0 + fm % 10;
}

bool chess_decode(Position *p, const int16_t t[CHESS_NUM_TOKENS]) {
    memset(p, 0, sizeof *p);
    for (int sq = 0; sq < 64; sq++) {
        int code = t[sq];
        if (code == TOK_EMPTY) continue;
        if (code < 1 || code > 12) return false;
        int c = (code - 1) / 6, pt = (code - 1) % 6;
        p->bb[c][pt] |= BIT(sq);
    }
    p->side = (t[64] == TOK_STM_B) ? BLACK : WHITE;
    p->castling = 0;
    if (t[65] == TOK_FLAG_ON) p->castling |= CASTLE_WK;
    if (t[66] == TOK_FLAG_ON) p->castling |= CASTLE_WQ;
    if (t[67] == TOK_FLAG_ON) p->castling |= CASTLE_BK;
    if (t[68] == TOK_FLAG_ON) p->castling |= CASTLE_BQ;
    if (t[69] == TOK_EP_NONE) p->ep_square = NO_SQUARE;
    else {
        int file = t[69] - TOK_EP_FILE_A;
        int rank = (t[70] == TOK_EP_RANK_3) ? 2 : 5;
        p->ep_square = rank * 8 + file;
    }
    p->halfmove = (t[71] - TOK_DIGIT_0) * 100 + (t[72] - TOK_DIGIT_0) * 10 + (t[73] - TOK_DIGIT_0);
    p->fullmove = (t[74] - TOK_DIGIT_0) * 100 + (t[75] - TOK_DIGIT_0) * 10 + (t[76] - TOK_DIGIT_0);
    return true;
}

// ---- codec: move <-> (8 x 8 x 73) policy index ------------------------------
static int dir_index(int dr, int df) {
    int sr = (dr > 0) - (dr < 0), sf = (df > 0) - (df < 0);
    static const int sdr[8] = { +1, +1, 0, -1, -1, -1, 0, +1 };
    static const int sdf[8] = {  0, +1, +1, +1, 0, -1, -1, -1 };
    for (int i = 0; i < 8; i++) if (sdr[i] == sr && sdf[i] == sf) return i;
    return -1;
}

int chess_move_to_index(Move m) {
    int from = move_from(m), to = move_to(m);
    int dr = (to >> 3) - (from >> 3), df = (to & 7) - (from & 7);
    int plane;
    if (move_is_promo(m) && move_promo_piece(m) != QUEEN) {
        int u = move_promo_piece(m) - KNIGHT;          // N=0,B=1,R=2
        int mm = (df == 0) ? 0 : (df < 0) ? 1 : 2;     // forward / cap-left / cap-right
        plane = 64 + u * 3 + mm;
    } else if ((abs(dr) == 1 && abs(df) == 2) || (abs(dr) == 2 && abs(df) == 1)) {
        static const int kdr[8] = { +2, +1, -1, -2, -2, -1, +1, +2 };
        static const int kdf[8] = { +1, +2, +2, +1, -1, -2, -2, -1 };
        int kn = 0;
        for (int i = 0; i < 8; i++) if (kdr[i] == dr && kdf[i] == df) { kn = i; break; }
        plane = 56 + kn;
    } else {
        int dist = (abs(dr) > abs(df)) ? abs(dr) : abs(df);
        plane = dir_index(dr, df) * 7 + (dist - 1);
    }
    return from * 73 + plane;
}

Move chess_index_to_move(int idx, const Move *legal, int n_legal) {
    for (int i = 0; i < n_legal; i++) if (chess_move_to_index(legal[i]) == idx) return legal[i];
    return MOVE_NONE;
}

void chess_legal_mask(const Move *legal, int n_legal, uint8_t mask[CHESS_POLICY_SIZE]) {
    memset(mask, 0, CHESS_POLICY_SIZE);
    for (int i = 0; i < n_legal; i++) mask[chess_move_to_index(legal[i])] = 1;
}
