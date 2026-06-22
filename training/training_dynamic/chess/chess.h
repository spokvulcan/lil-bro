// chess.h — hand-written, dependency-free C bitboard chess engine.
//
// Build-step 1 of the chess-RL-on-ANE build order (ADR 0005, issue #15). This is the
// self-play environment's *source of truth* for board state, legal moves, and the
// encodings the policy/value net consumes. Pure C, zero external dependencies — the
// "add the op, don't add a dependency" project law (CLAUDE.md). `python-chess` may be
// used as an eval-side perft oracle in tests only; it is never importable here.
//
// Correctness is proven by perft (perft.c) against published node counts. One wrong
// movegen edge case silently corrupts every downstream learning number, so perft is a
// first-class gate (ADR 0005 gate ladder): it must match published counts EXACTLY.
//
// =============================================================================
// Board geometry: Little-Endian Rank-File (LERF). square = rank*8 + file.
//   a1=0  b1=1 ... h1=7   a2=8 ...   a8=56 ... h8=63
//   file = square & 7  (a=0 .. h=7)     rank = square >> 3  (rank 1 = 0 .. rank 8 = 7)
// Bit `s` of a bitboard is set iff square `s` is occupied.
// =============================================================================

#ifndef LILBRO_CHESS_H
#define LILBRO_CHESS_H

#include <stdint.h>
#include <stdbool.h>

// ---- Colors -----------------------------------------------------------------
enum { WHITE = 0, BLACK = 1 };

// ---- Piece types ------------------------------------------------------------
enum { PAWN = 0, KNIGHT, BISHOP, ROOK, QUEEN, KING, NO_PIECE_TYPE = 6 };

// ---- Castling rights bitmask (Position.castling) ----------------------------
enum { CASTLE_WK = 1, CASTLE_WQ = 2, CASTLE_BK = 4, CASTLE_BQ = 8 };

#define NO_SQUARE (-1)

// ---- Position ---------------------------------------------------------------
// POD struct, memset-able; equality is bytewise (used by the make/unmake gate).
// Occupancy is derived on demand (chess_occupancy / per-color helpers) rather than
// cached, so there is no redundant state to drift. Zobrist hashing is intentionally
// deferred to the search step (#17/#18) — it is not needed to prove movegen here, and
// leaving it out keeps bytewise equality bulletproof.
typedef struct {
    uint64_t bb[2][6];  // bb[color][piece_type]
    int side;           // side to move: WHITE or BLACK
    int ep_square;      // en-passant target square (square the capturer lands on), or NO_SQUARE
    int castling;       // OR of CASTLE_* flags still available
    int halfmove;       // halfmove clock (plies since last pawn move/capture; 50-move rule)
    int fullmove;       // fullmove number (starts at 1, increments after Black's move)
} Position;

// ---- Move -------------------------------------------------------------------
// Packed 16-bit move (CPW encoding): bits 0..5 = from, 6..11 = to, 12..15 = flag.
typedef uint16_t Move;

enum {
    MF_QUIET       = 0,
    MF_DOUBLE_PUSH = 1,   // pawn two-square advance (sets ep_square)
    MF_KCASTLE     = 2,   // king-side castle
    MF_QCASTLE     = 3,   // queen-side castle
    MF_CAPTURE     = 4,   // bit 2 (value 4) = capture flag
    MF_EP          = 5,   // en-passant capture
    MF_PROMO_N     = 8,   // bit 3 (value 8) = promotion flag; low 2 bits select piece
    MF_PROMO_B     = 9,
    MF_PROMO_R     = 10,
    MF_PROMO_Q     = 11,
    MF_PROMO_N_CAP = 12,
    MF_PROMO_B_CAP = 13,
    MF_PROMO_R_CAP = 14,
    MF_PROMO_Q_CAP = 15,
};

#define MOVE_NONE ((Move)0)  // a1->a1 is never a legal move, so 0 is a safe sentinel

static inline Move move_make(int from, int to, int flag) {
    return (Move)(from | (to << 6) | (flag << 12));
}
static inline int move_from(Move m) { return m & 0x3f; }
static inline int move_to(Move m)   { return (m >> 6) & 0x3f; }
static inline int move_flag(Move m) { return (m >> 12) & 0xf; }
static inline bool move_is_capture(Move m) { return (move_flag(m) & MF_CAPTURE) != 0; }
static inline bool move_is_promo(Move m)   { return (move_flag(m) & 0x8) != 0; }
// Promotion target piece type (only valid when move_is_promo): N/B/R/Q.
static inline int move_promo_piece(Move m) { return (move_flag(m) & 0x3) + KNIGHT; }

#define MAX_MOVES 256  // a legal position has at most 218 moves; 256 is a safe bound

// =============================================================================
// Engine API
// =============================================================================

// Initialize precomputed attack/geometry tables. Call once before any other call.
// Idempotent and deterministic (no RNG anywhere in the engine).
void chess_init(void);

// Set up the standard starting position.
void chess_startpos(Position *p);

// Parse a FEN string into *p. Returns true on success. Pure (no globals touched).
bool chess_from_fen(Position *p, const char *fen);

// Write the FEN of *p into buf (caller provides >= 90 bytes). Returns buf.
char *chess_to_fen(const Position *p, char *buf);

// Generate all *fully legal* moves into out[] (capacity MAX_MOVES). Returns the count.
// Handles pins, single/double check evasion, en-passant (incl. the pin-on-EP edge
// case), castling (rights/empty/through-check), and promotions. Deterministic.
int chess_legal_moves(const Position *p, Move *out);

// True iff the side to move is in check.
bool chess_in_check(const Position *p);

// Apply / revert a move. unmake restores *p to its exact pre-make state (bytewise).
// `u` is opaque undo state filled by make and consumed by the matching unmake.
typedef struct { int captured; int ep_square; int castling; int halfmove; int fullmove; } Undo;
void chess_make(Position *p, Move m, Undo *u);
void chess_unmake(Position *p, Move m, const Undo *u);

// Convenience: occupancy bitboards.
uint64_t chess_occupancy(const Position *p);            // all pieces
uint64_t chess_color_occupancy(const Position *p, int color);

// Format a move as UCI (e.g. "e2e4", "e7e8q"). buf >= 6 bytes. Returns buf.
char *chess_move_to_uci(Move m, char *buf);

// =============================================================================
// Perft (perft.c uses these; declared here so tests can share them).
// =============================================================================
// Standard perft: counts leaf nodes at `depth`. Bulk-counts at depth 1 (counts the
// legal moves without making them) — the canonical perft speedup; make/unmake is still
// exercised at every interior ply.
uint64_t chess_perft(Position *p, int depth);

// =============================================================================
// CODEC — the contract issue #16 (policy head + legal mask) consumes.
// =============================================================================
//
// ---- Position -> token sequence (fixed length CHESS_NUM_TOKENS = 77) ---------
//
// Layout (DeepMind "Grandmaster-Level Chess Without Search" fixed-length FEN
// tokenization; 64 + 1 + 4 + 2 + 3 + 3 = 77). Sequence *position* carries the board
// square (so a 2-D learned rank+file positional encoding can decompose it — ADR 0005
// decision 10); token *value* carries the occupant/state. Padded to 96 (mult-of-32)
// for the ANE at the net boundary in #16; the 77 real tokens are produced here.
//
//   index  0..63 : board squares a1..h8 in LERF order. value = occupant code:
//                    0 = empty
//                    1..6  = white {P,N,B,R,Q,K}
//                    7..12 = black {p,n,b,r,q,k}
//   index 64     : side to move.            value: TOK_STM_W / TOK_STM_B
//   index 65..68 : castling WK,WQ,BK,BQ.    value: TOK_FLAG_OFF / TOK_FLAG_ON
//   index 69     : en-passant file.         value: TOK_EP_NONE, or TOK_EP_FILE_A+file
//   index 70     : en-passant rank.         value: TOK_EP_NONE, or TOK_EP_RANK_3/_6
//   index 71..73 : halfmove clock, 3 base-10 digits (hundreds,tens,ones) clamped
//                  to 0..999.               value: TOK_DIGIT_0 + d
//   index 74..76 : fullmove number, 3 base-10 digits (hundreds,tens,ones) clamped
//                  to 0..999.               value: TOK_DIGIT_0 + d
//
// Token-id ranges are disjoint, so #16 may use a single embedding table; the
// sequence position disambiguates which field a (possibly shared) id belongs to.
// `chess_decode` inverts `chess_encode` exactly for halfmove,fullmove in [0,999].
#define CHESS_NUM_TOKENS 77
#define CHESS_PAD_TOKENS 96  // mult-of-32 padding length used at the ANE boundary (#16)

enum {
    TOK_EMPTY = 0,
    // 1..12 occupant codes (color*6 + piece_type + 1)
    TOK_STM_W = 13, TOK_STM_B = 14,
    TOK_FLAG_OFF = 15, TOK_FLAG_ON = 16,
    TOK_EP_NONE = 17,
    TOK_EP_FILE_A = 18,  // ..18+7 = file h
    TOK_EP_RANK_3 = 26, TOK_EP_RANK_6 = 27,
    TOK_DIGIT_0 = 28,    // ..28+9 = digit 9
    CHESS_VOCAB = 38,    // total distinct token ids
};

// Occupant code for a piece (color,type) — the value used for board tokens 0..63.
static inline int chess_occupant_code(int color, int type) { return color * 6 + type + 1; }

void chess_encode(const Position *p, int16_t tokens[CHESS_NUM_TOKENS]);
// Reconstruct a Position from tokens. Returns true on success (well-formed tokens).
bool chess_decode(Position *p, const int16_t tokens[CHESS_NUM_TOKENS]);

// ---- move <-> (8 x 8 x 73) index map (AlphaZero policy encoding) -------------
//
// index = from_square * 73 + plane,   range [0, CHESS_POLICY_SIZE = 4672).
// from_square = from_rank*8 + from_file gives the 8x8; 73 planes per square:
//
//   planes  0..55 : "queen" moves = direction d (0..7) * 7 + (distance-1), distance 1..7.
//                   direction order (clockwise from North), (drank,dfile):
//                     0 N (+1, 0)   1 NE(+1,+1)  2 E (0,+1)  3 SE(-1,+1)
//                     4 S (-1, 0)   5 SW(-1,-1)  6 W (0,-1)  7 NW(+1,-1)
//                   covers rook/bishop/queen/king slides, pawn pushes & captures,
//                   double pushes, castling (king 2 sq horizontally), and QUEEN
//                   promotions (a pawn reaching the last rank via a queen plane).
//   planes 56..63 : 8 knight moves, fixed (drank,dfile) order:
//                     56 (+2,+1) 57 (+1,+2) 58 (-1,+2) 59 (-2,+1)
//                     60 (-2,-1) 61 (-1,-2) 62 (+1,-2) 63 (+2,-1)
//   planes 64..72 : 9 underpromotions = piece u (N=0,B=1,R=2) * 3 + dir m, where
//                   dir is by FILE DELTA (color-independent): forward dfile=0 -> m=0,
//                   capture-left dfile=-1 -> m=1, capture-right dfile=+1 -> m=2.
//                   (Queen promotions use the queen planes, not these.)
//
// This map is a bijection over the LEGAL moves of a single position (proven in
// test_chess.c) — but it is deliberately NOT injective over arbitrary Move values:
// queen promotions fold into the queen planes (a7a8=Q shares an index with a hypothetical
// non-promo slide to a8), and the quiet/capture flag is dropped (two moves to the same
// square share an index). Two colliding moves are never simultaneously legal, so within a
// single position every legal move gets a unique index. Consequence for the #16 policy
// head: build the legal mask from the position's legal-move list, and never invert a
// policy index without that list (chess_index_to_move requires it — an index alone does
// not determine a move).
#define CHESS_POLICY_SIZE 4672  // 64 * 73

// Map a move to its policy index. Pure function of (from, to, promotion flag). See the
// non-injectivity note above: distinct hypothetical moves may collide; legal ones do not.
int chess_move_to_index(Move m);

// Inverse, resolved against a position's legal-move list: return the legal move whose
// index == idx, or MOVE_NONE if none. The legal list is REQUIRED (see note above).
// Decoding is only ever applied to masked (legal) indices.
Move chess_index_to_move(int idx, const Move *legal, int n_legal);

// Fill a legal-move mask: mask[i] = 1 iff some legal move maps to policy index i.
void chess_legal_mask(const Move *legal, int n_legal, uint8_t mask[CHESS_POLICY_SIZE]);

#endif  // LILBRO_CHESS_H
