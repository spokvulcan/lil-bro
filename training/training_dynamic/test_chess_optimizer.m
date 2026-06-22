// test_chess_optimizer.m — optimizer split contract for the chess learner.
//
// Slice 5 requires the V4 split: canonical 2D trunk matrices use Muon, while
// embeddings, RMSNorm gains, and policy/value heads stay on AdamW.

#include "mil_dynamic.h"
#include "cpu_ops.h"
#include "chess/chess.h"
#include "chess/chess_heads.h"
#include "chess/chess_net.h"
#include <stdio.h>
#include <math.h>

static int g_fail = 0;

static float adamw_after_one(float w, float g, float lr, float wd) {
    (void)g;  // first AdamW step bias-corrects mh=g and vh=g*g.
    float step = lr * ((g >= 0.0f ? 1.0f : -1.0f) + wd * w);
    return w - step;
}

static void check_close(const char *name, float got, float want) {
    float err = fabsf(got - want);
    if (err > 2e-6f) {
        printf("FAIL %-10s got %.9f want %.9f err %.3e\n", name, got, want, err);
        g_fail = 1;
    } else {
        printf("ok   %-10s AdamW update preserved\n", name);
    }
}

int main(void) {
    ChessNet W, G;
    chess_net_alloc(&W, 0);
    chess_net_alloc(&G, 1);

    ParamRef wp[256], gp[256];
    int k = chess_net_params(&W, wp);
    chess_net_params(&G, gp);
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < wp[i].n; j++) wp[i].p[j] = 0.125f;
        for (int j = 0; j < gp[i].n; j++) gp[i].p[j] = 0.25f;
    }

    chess_net_register(&W, &G);
    chess_optimizer_set_muon(1);

    float trunk0 = W.W[0].Wq[0];
    float rms0 = W.W[0].rms_att[0];
    float tok0 = W.tok_emb[0];
    float pol0 = W.W_pol[0];
    float val0 = W.W_val[0];

    optimizer_step(1.0f, 0.0f, 1, 0.01f, 0.1f);

    float adam_trunk = adamw_after_one(trunk0, 0.25f, 0.01f, 0.1f);
    if (!isfinite(W.W[0].Wq[0])) {
        printf("FAIL trunk Wq produced non-finite Muon update\n");
        g_fail = 1;
    } else if (fabsf(W.W[0].Wq[0] - adam_trunk) < 1e-6f) {
        printf("FAIL trunk Wq matched AdamW; expected Muon dispatch\n");
        g_fail = 1;
    } else {
        printf("ok   trunk Wq  Muon update selected (AdamW would be %.9f, got %.9f)\n",
               adam_trunk, W.W[0].Wq[0]);
    }

    check_close("rms_att", W.W[0].rms_att[0], adamw_after_one(rms0, 0.25f, 0.01f, 0.1f));
    check_close("tok_emb", W.tok_emb[0],      adamw_after_one(tok0, 0.25f, 0.01f, 0.1f));
    check_close("W_pol",   W.W_pol[0],        adamw_after_one(pol0, 0.25f, 0.01f, 0.1f));
    check_close("W_val",   W.W_val[0],        adamw_after_one(val0, 0.25f, 0.01f, 0.1f));

    if (g_fail) {
        printf("*** test_chess_optimizer: FAILURES ***\n");
        return 1;
    }
    printf("# test_chess_optimizer: ALL TESTS PASSED\n");
    return 0;
}
