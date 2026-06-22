// replay.h — sliding-window self-play replay buffer (build-step 4, ADR 0005, issue #18).
//
// The learner trains on a uniform sample of the most-recent self-play positions (the
// classic AlphaZero "always-latest" window — ADR 0005 decision 14: no league, no
// best-checkpoint gating). This is the ring that holds them: a fixed-capacity sliding
// window (oldest evicted first) + uniform sampling with replacement, deterministic from
// a seed. Pure C, zero deps beyond the engine codec (the project law, CLAUDE.md).
//
// One sample = one trained position: the network input (token sequence) + the two
// AlphaZero targets (ADR 0005 decision 13):
//   - policy <- the temperature-shaped MCTS visit/improved distribution, stored SPARSELY
//     as (policy-index, prob) over the position's legal moves (a dense 4672-vector per
//     sample would be ~80x larger; the learner re-expands it). chess_move_to_index.
//   - value  <- the game outcome z in [-1,1] from THIS position's side-to-move (win=+1).
// The legal mask is NOT stored: the learner recomputes it from the decoded position
// (chess_decode -> chess_legal_mask), so the mask can never drift from the tokens.
#ifndef LILBRO_CHESS_REPLAY_H
#define LILBRO_CHESS_REPLAY_H

#include "chess.h"
#include <stdint.h>

#define REPLAY_MAX_POLICY MAX_MOVES   // <= one sparse policy entry per legal move

typedef struct {
    uint16_t tokens[CHESS_PAD_TOKENS];        // padded position tokens (SEQ); the net input
    int      n_policy;                        // number of valid (idx,p) policy entries
    int      policy_idx[REPLAY_MAX_POLICY];   // dense policy index per entry (chess_move_to_index)
    float    policy_p[REPLAY_MAX_POLICY];     // target prob per entry (sums to 1 over entries)
    float    z;                               // game outcome in [-1,1], stm perspective
} ReplaySample;

// A fixed-capacity sliding window: the buffer always retains the most-recent `capacity`
// samples (older ones are overwritten in ring order). count = min(total_added, capacity).
typedef struct {
    ReplaySample *buf;
    int       capacity;     // window size (max retained samples); >= 1
    int       count;        // current valid samples (<= capacity)
    int       head;         // ring write cursor = total_added % capacity
    long      total_added;  // lifetime samples added (monotone; diagnostics)
    uint64_t  rng;          // splitmix64 state for uniform sampling (deterministic)
} ReplayBuffer;

// Allocate a window of `capacity` samples; `seed` fixes the sampling RNG (reproducible).
void replay_init(ReplayBuffer *rb, int capacity, uint64_t seed);
void replay_free(ReplayBuffer *rb);

// Append one sample. When full, overwrites the OLDEST (sliding window). O(1).
void replay_add(ReplayBuffer *rb, const ReplaySample *s);

static inline int replay_count(const ReplayBuffer *rb) { return rb->count; }
static inline int replay_full(const ReplayBuffer *rb)  { return rb->count >= rb->capacity; }

// One uniform sample (with replacement) from the current window, or NULL if empty.
// The pointer is into the buffer and is valid until the slot is overwritten.
const ReplaySample *replay_sample(ReplayBuffer *rb);

// Fill out[0..k) with k uniform samples (with replacement). Returns the number written
// (0 if the buffer is empty; otherwise k). Deterministic given the buffer's RNG state.
int replay_sample_batch(ReplayBuffer *rb, const ReplaySample **out, int k);

#endif  // LILBRO_CHESS_REPLAY_H
