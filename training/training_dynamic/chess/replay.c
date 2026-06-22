// replay.c — sliding-window self-play replay buffer. See replay.h for the contract.
// Pure C, zero deps (stdlib/string). Deterministic uniform sampling via splitmix64
// (the same generator family as mcts.c, so the whole loop is reproducible from seeds).

#include "replay.h"
#include <stdlib.h>
#include <string.h>

static inline uint64_t splitmix64(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// Uniform integer in [0, n) without modulo bias (rejection on the top remainder).
static int rng_below(uint64_t *s, int n) {
    if (n <= 1) return 0;
    uint64_t un = (uint64_t)n;
    uint64_t lim = UINT64_MAX - (UINT64_MAX % un);   // largest multiple of n (exclusive bound)
    uint64_t r;
    do { r = splitmix64(s); } while (r >= lim);
    return (int)(r % un);
}

void replay_init(ReplayBuffer *rb, int capacity, uint64_t seed) {
    if (capacity < 1) capacity = 1;
    rb->buf = (ReplaySample*)malloc((size_t)capacity * sizeof(ReplaySample));
    rb->capacity = capacity;
    rb->count = 0;
    rb->head = 0;
    rb->total_added = 0;
    rb->rng = seed ? seed : 0x123456789ABCDEFull;   // avoid an all-zero splitmix state
}

void replay_free(ReplayBuffer *rb) {
    free(rb->buf);
    rb->buf = NULL;
    rb->capacity = rb->count = rb->head = 0;
    rb->total_added = 0;
}

void replay_add(ReplayBuffer *rb, const ReplaySample *s) {
    rb->buf[rb->head] = *s;                          // overwrite oldest when full
    rb->head = (rb->head + 1) % rb->capacity;
    if (rb->count < rb->capacity) rb->count++;
    rb->total_added++;
}

const ReplaySample *replay_sample(ReplayBuffer *rb) {
    if (rb->count == 0) return NULL;
    return &rb->buf[rng_below(&rb->rng, rb->count)];
}

int replay_sample_batch(ReplayBuffer *rb, const ReplaySample **out, int k) {
    if (rb->count == 0) return 0;
    for (int i = 0; i < k; i++)
        out[i] = &rb->buf[rng_below(&rb->rng, rb->count)];
    return k;
}
