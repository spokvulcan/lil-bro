// test_replay.c — invariants for the self-play replay buffer (build-step 4, #18).
//
// Pure C, no ANE — the fast TDD gate for the sliding-window + uniform-sampling
// contract the learner depends on. A replay bug (wrong eviction, biased/duplicated
// sampling, off-by-one window) would silently corrupt the learning signal long before
// any win-rate curve could reveal it — exactly the class of error the project's
// "evidence before assertion" discipline exists to catch early.
//
// Build: see Makefile target `test_replay`.

#include "replay.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int g_fail = 0;
static void check(int cond, const char *name) {
    printf("  %-56s %s\n", name, cond ? "OK" : "*** FAIL ***");
    if (!cond) g_fail++;
}

// A sample tagged with an integer id (carried in tokens[0] and z) so eviction and
// sampling can be verified by identity.
static ReplaySample mk(int id) {
    ReplaySample s;
    memset(&s, 0, sizeof s);
    s.tokens[0] = (uint16_t)id;
    s.z = (float)id;
    s.n_policy = 1;
    s.policy_idx[0] = id;
    s.policy_p[0] = 1.0f;
    return s;
}
static int sid(const ReplaySample *s) { return (int)s->tokens[0]; }

// ---- empty-buffer invariants ----
static void test_empty(void) {
    printf("[empty buffer]\n");
    ReplayBuffer rb; replay_init(&rb, 8, 42);
    check(replay_count(&rb) == 0, "fresh buffer count == 0");
    check(!replay_full(&rb), "fresh buffer not full");
    check(replay_sample(&rb) == NULL, "sample of empty buffer -> NULL");
    const ReplaySample *out[4];
    check(replay_sample_batch(&rb, out, 4) == 0, "sample_batch of empty -> 0");
    replay_free(&rb);
}

// ---- partial fill: count tracks adds; the exact samples are retained ----
static void test_partial_fill(void) {
    printf("[partial fill (count < capacity)]\n");
    ReplayBuffer rb; replay_init(&rb, 100, 42);
    for (int i = 0; i < 30; i++) { ReplaySample s = mk(i); replay_add(&rb, &s); }
    check(replay_count(&rb) == 30, "count == number added (30)");
    check(!replay_full(&rb), "30/100 -> not full");
    check(rb.total_added == 30, "total_added == 30");
    // every id 0..29 must be present exactly once
    int seen[30]; memset(seen, 0, sizeof seen);
    int ok = 1;
    for (int i = 0; i < rb.count; i++) { int id = sid(&rb.buf[i]); if (id < 0 || id >= 30 || seen[id]) ok = 0; else seen[id] = 1; }
    for (int i = 0; i < 30; i++) if (!seen[i]) ok = 0;
    check(ok, "ids 0..29 each present exactly once");
    replay_free(&rb);
}

// ---- overflow: sliding window keeps exactly the most-recent `capacity` samples ----
static void test_sliding_window(void) {
    printf("[sliding window (overflow evicts oldest)]\n");
    const int cap = 64, extra = 50;
    ReplayBuffer rb; replay_init(&rb, cap, 7);
    for (int i = 0; i < cap + extra; i++) { ReplaySample s = mk(i); replay_add(&rb, &s); }
    check(replay_count(&rb) == cap, "count saturates at capacity");
    check(replay_full(&rb), "buffer reports full");
    check(rb.total_added == cap + extra, "total_added counts all adds (lifetime)");
    // the retained id set must be exactly { extra .. cap+extra-1 } (most-recent cap).
    int lo = extra, hi = cap + extra;   // [lo, hi)
    int ok = 1, present[200]; memset(present, 0, sizeof present);
    for (int i = 0; i < rb.count; i++) {
        int id = sid(&rb.buf[i]);
        if (id < lo || id >= hi) ok = 0;        // an evicted (too-old) id leaked
        else present[id] = 1;
    }
    for (int id = lo; id < hi; id++) if (!present[id]) ok = 0;   // a recent id is missing
    check(ok, "retained set == the most-recent `capacity` ids (oldest evicted)");
    replay_free(&rb);
}

// ---- uniform sampling: in-window, deterministic, covers the whole window, ~uniform ----
static void test_uniform_sampling(void) {
    printf("[uniform sampling]\n");
    const int cap = 64;
    ReplayBuffer rb; replay_init(&rb, cap, 12345);
    for (int i = 0; i < cap; i++) { ReplaySample s = mk(i); replay_add(&rb, &s); }   // ids 0..63

    const int draws = cap * 1000;       // ~1000 expected hits per id
    int *hits = (int*)calloc(cap, sizeof(int));
    int in_window = 1;
    for (int d = 0; d < draws; d++) {
        const ReplaySample *s = replay_sample(&rb);
        int id = sid(s);
        if (id < 0 || id >= cap) in_window = 0; else hits[id]++;
    }
    check(in_window, "every drawn sample is within the window");
    int covered = 1; for (int i = 0; i < cap; i++) if (hits[i] == 0) covered = 0;
    check(covered, "every in-window sample is drawn at least once (coverage)");
    // loose uniformity: every bucket within 0.5x..2x of the expected count.
    int expected = draws / cap, lo = expected/2, hi = expected*2, uniform = 1;
    for (int i = 0; i < cap; i++) if (hits[i] < lo || hits[i] > hi) uniform = 0;
    check(uniform, "per-sample hit counts within 0.5x..2x of uniform expectation");
    free(hits);
    replay_free(&rb);
}

// ---- determinism: same seed -> same sample sequence; different seed -> different ----
static void test_determinism(void) {
    printf("[sampling determinism]\n");
    const int cap = 32, draws = 200;
    ReplayBuffer a, b, c;
    replay_init(&a, cap, 999); replay_init(&b, cap, 999); replay_init(&c, cap, 1000);
    for (int i = 0; i < cap; i++) { ReplaySample s = mk(i); replay_add(&a, &s); replay_add(&b, &s); replay_add(&c, &s); }
    int same_ab = 1, any_diff_ac = 0;
    for (int d = 0; d < draws; d++) {
        int ia = sid(replay_sample(&a)), ib = sid(replay_sample(&b)), ic = sid(replay_sample(&c));
        if (ia != ib) same_ab = 0;
        if (ia != ic) any_diff_ac = 1;
    }
    check(same_ab, "identical seed -> identical sample sequence");
    check(any_diff_ac, "different seed -> different sample sequence");
    replay_free(&a); replay_free(&b); replay_free(&c);
}

// ---- batch sampling returns k in-window samples ----
static void test_sample_batch(void) {
    printf("[batch sampling]\n");
    const int cap = 50;
    ReplayBuffer rb; replay_init(&rb, cap, 2024);
    for (int i = 0; i < cap; i++) { ReplaySample s = mk(i); replay_add(&rb, &s); }
    const ReplaySample *out[128];
    int k = replay_sample_batch(&rb, out, 128);
    check(k == 128, "sample_batch returns the requested k");
    int ok = 1; for (int i = 0; i < k; i++) { int id = sid(out[i]); if (id < 0 || id >= cap) ok = 0; }
    check(ok, "every batch sample is within the window");
    // sample content is intact (z mirrors the id we stored)
    int intact = 1; for (int i = 0; i < k; i++) if ((int)out[i]->z != sid(out[i])) intact = 0;
    check(intact, "sample payload (z, tokens) intact through the buffer");
    replay_free(&rb);
}

int main(void) {
    test_empty();
    test_partial_fill();
    test_sliding_window();
    test_uniform_sampling();
    test_determinism();
    test_sample_batch();
    printf("\n%s (%d failure%s)\n", g_fail ? "TESTS FAILED" : "ALL TESTS PASSED",
           g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
