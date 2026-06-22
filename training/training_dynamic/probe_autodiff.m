// probe_autodiff.m — Slice 0 autodiff spike (issue #23 / ADR 0006 build-step 0),
// now also the Slice 2 gate (issue #25): the autodiff trunk lives in chess_net.h
// (mg_ad_*), so this probe exercises the PRODUCTION learner-path functions.
//
// The repo had ZERO prior MPSGraph autodiff usage — every graph was hand-written
// forward. This probe answers ONE question and is now the gate for both slices:
//
//   Does MPSGraph gradientForPrimaryTensor:withTensors: produce a trunk backward
//   that matches the CPU chess_trunk_backward reference to fp32 cosine >= 0.999
//   (slice 2 criterion #2, now that #36 fixed the CPU reference), match finite-
//   difference to >= 0.998 (the fp32-FD ceiling), and does it train (G0 overfit)?
//
// Three modes (the gates):
//
//   ./probe_autodiff --grad-diff [--batch N]   (default; slice 2 criterion #2)
//     Grad-diff mg_ad_backward (the production GPU learner backward) vs the CPU
//     chess_trunk_backward (FD-verified via test_attn + this probe's --fd). PASS =
//     worst fp32 cosine >= 0.999 across all canonical weights + x_in + rms_final.
//     --batch N verifies at the learner's batch width (default 1).
//
//   ./probe_autodiff --fd [--batch N]
//     Finite-difference ground truth (central-difference via the CPU forward, which
//     perturbs the forward only so cannot share a backward bug with either path).
//     Reports cos(FD, autodiff) and cos(FD, cpu). PASS = worst cos(FD,ad) >= 0.998.
//
//   ./probe_autodiff --g0
//     Behavioral safety net. Overfits one chess position via mg_ad_forward/backward
//     (the production GPU trunk) + the existing CPU heads/loss/posenc/embed/optimizer.
//     PASS = both cross-entropies collapse under thresh (loss -> ~0).
//
// VERDICT (all green) => hybrid autodiff viable; slice 2's GPU learner backward is
// verified. Results in results/autodiff_spike.md.
//
// Build: make probe_autodiff   (see Makefile target; mirrors train_chess flags).

#include "mil_dynamic.h"          // pulls io.h + config.h (DIM/SEQ/... from chess_g0.h)
#include "cpu_ops.h"              // rmsnorm(_bwd), attn_cpu_*, adam_update, embed_*
#include "chess/chess.h"          // engine + codec (#15): encode, legal moves/mask
#include "chess/chess_heads.h"    // heads: posenc, policy, value, L2 (FD-gated)
#include "chess/chess_net.h"      // trunk fwd/bwd, ChessNet, mg_ad_* (the GPU learner path)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mach/mach_time.h>

// ============================================================================
// Helpers: fp32 cosine, deterministic RNG (mirrors probe_mps_graph's fill_randn).
// ============================================================================
static double ad_cosine(const float *a, const float *b, size_t n) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; i++) { dot += (double)a[i]*b[i]; na += (double)a[i]*a[i]; nb += (double)b[i]*b[i]; }
    return dot / (sqrt(na) * sqrt(nb) + 1e-30);
}
static void ad_fill_randn(float *p, size_t n, unsigned *seed) {
    for (size_t i = 0; i < n; i++) {
        float s = 0;
        for (int k = 0; k < 12; k++) { *seed = *seed * 1103515245u + 12345u; s += (float)(*seed >> 8) / 16777216.0f; }
        p[i] = (s - 6.0f) * 0.1f;
    }
}

// Report one tensor's cosine; track the running worst. Returns 1 to keep scanning.
static int ad_report(const char *name, const float *a, const float *b, size_t n,
                     double *worst, char *worst_name, int show_fail_only) {
    double c = ad_cosine(a, b, n);
    if (c < *worst) { *worst = c; strncpy(worst_name, name, 63); worst_name[63] = '\0'; }
    if (!show_fail_only || c < 0.999) printf("   %-18s n=%-9zu cos=%.6f\n", name, n, c);
    return 1;
}

// ============================================================================
// MODE 1 — grad-diff: mg_ad_backward (production GPU learner) vs CPU chess_trunk_backward.
// ============================================================================
static int run_grad_diff(int B) {
    int S = B*SEQ;
    printf("# probe_autodiff --grad-diff  (B=%d)\n", B);
    printf("# mg_ad_backward (GPU learner) vs CPU chess_trunk_backward (fp32 cosine)\n");
    printf("# DIM=%d HIDDEN=%d HEADS=%d HD=%d SEQ=%d NLAYERS=%d\n\n",
           DIM, HIDDEN, HEADS, HD, SEQ, NLAYERS);

    ChessNet Wn, G_cpu, G_ad;
    chess_net_alloc(&Wn, 0); chess_net_alloc(&G_cpu, 1); chess_net_alloc(&G_ad, 1);
    chess_net_init(&Wn, 42);

    float *x_in    = fmalloc((size_t)DIM*S);
    float *dx_final= fmalloc((size_t)DIM*S);
    float *x_pre   = fmalloc((size_t)DIM*S);
    float *x_final = fmalloc((size_t)DIM*S);
    float *dy_cpu  = fmalloc((size_t)DIM*S);
    float *dy_ad   = fmalloc((size_t)DIM*S);
    float *grms_cpu= fcalloc(DIM);
    float *grms_ad = fcalloc(DIM);
    unsigned seed = 7;
    ad_fill_randn(x_in, (size_t)DIM*S, &seed);
    ad_fill_randn(dx_final, (size_t)DIM*S, &seed);

    // ---- CPU reference: chess_trunk_forward(save_acts=1) + chess_trunk_backward ----
    g_cpu_mm = 1;
    CLayer Wl[NLAYERS], Gl[NLAYERS]; CActs acts[NLAYERS];
    for (int L = 0; L < NLAYERS; L++) {
        Wl[L] = Wn.W[L]; Gl[L] = G_cpu.W[L]; cacts_alloc(&acts[L], S);
    }
    for (int L = 0; L < NLAYERS; L++) chess_layer_build_fused(&Wl[L]);
    chess_net_init_rmstmp(S);
    float res_alpha = 1.0f/sqrtf(2.0f*NLAYERS);
    chess_trunk_forward(Wl, acts, x_in, B, x_pre, x_final, Wn.rms_final, res_alpha, 1);
    // Sanity: does the autodiff forward agree with the CPU forward? (isolates fwd vs bwd)
    float *x_final_ad = fmalloc((size_t)DIM*S);
    mg_ad_forward(B, x_in, Wn.W, Wn.rms_final, x_final_ad);
    double fwd_cos = ad_cosine(x_final, x_final_ad, (size_t)DIM*S);
    double fwd_max = 0; for (int i = 0; i < DIM*S; i++) { double e = fabs((double)x_final[i]-x_final_ad[i]); if (e>fwd_max) fwd_max=e; }
    printf("## forward agreement: cos(CPU,autodiff)=%.6f  max_abs_err=%.3e\n", fwd_cos, fwd_max);
    free(x_final_ad);
    grads_zero();
    chess_trunk_backward(Wl, Gl, acts, dx_final, B, x_pre, Wn.rms_final, grms_cpu, dy_cpu, res_alpha);

    // ---- production GPU autodiff backward ----
    // mg_ad_backward writes ABSOLUTE grads; G_ad/grms_ad/dy_ad were calloc'd (zero).
    chess_net_init_rmstmp(S);
    mg_ad_backward(B, x_in, Wn.W, Wn.rms_final, dx_final, G_ad.W, grms_ad, dy_ad);

    // ---- per-tensor grad-diff ----
    double worst = 2.0; char worst_name[64] = "(none)";
    printf("## per-tensor fp32 cosine (autodiff grad vs CPU grad)\n");
    #define CMP(NAME, A, Bp, N) ad_report(NAME, A, Bp, (size_t)(N), &worst, worst_name, 0)
    CMP("x_in (dy)",      dy_ad, dy_cpu, DIM*S);
    CMP("rms_final",      grms_ad, grms_cpu, DIM);
    for (int L = 0; L < NLAYERS; L++) {
        char nm[64];
        snprintf(nm, sizeof nm, "L%d.Wq", L);      CMP(nm, G_ad.W[L].Wq,      G_cpu.W[L].Wq,      DIM*Q_DIM);
        snprintf(nm, sizeof nm, "L%d.Wk", L);      CMP(nm, G_ad.W[L].Wk,      G_cpu.W[L].Wk,      DIM*KV_DIM);
        snprintf(nm, sizeof nm, "L%d.Wv", L);      CMP(nm, G_ad.W[L].Wv,      G_cpu.W[L].Wv,      DIM*KV_DIM);
        snprintf(nm, sizeof nm, "L%d.Wo", L);      CMP(nm, G_ad.W[L].Wo,      G_cpu.W[L].Wo,      Q_DIM*DIM);
        snprintf(nm, sizeof nm, "L%d.W1", L);      CMP(nm, G_ad.W[L].W1,      G_cpu.W[L].W1,      DIM*HIDDEN);
        snprintf(nm, sizeof nm, "L%d.W2", L);      CMP(nm, G_ad.W[L].W2,      G_cpu.W[L].W2,      HIDDEN*DIM);
        snprintf(nm, sizeof nm, "L%d.W3", L);      CMP(nm, G_ad.W[L].W3,      G_cpu.W[L].W3,      DIM*HIDDEN);
        snprintf(nm, sizeof nm, "L%d.rms_att", L); CMP(nm, G_ad.W[L].rms_att, G_cpu.W[L].rms_att, DIM);
        snprintf(nm, sizeof nm, "L%d.rms_ffn", L); CMP(nm, G_ad.W[L].rms_ffn, G_cpu.W[L].rms_ffn, DIM);
    }
    #undef CMP
    const double GATE = 0.999;
    printf("\n## worst cosine vs CPU = %.6f @ %s  (gate >= %.3f)  =>  %s\n",
           worst, worst_name, GATE, worst >= GATE ? "PASS" : "FAIL");
    return worst >= GATE ? 0 : 1;
}

// ============================================================================
// MODE 2 — G0 overfit: mg_ad_* (production GPU trunk) + CPU heads/loss/embed/optimizer.
// Mirrors train_chess.m's --overfit loop with ONLY the trunk fwd/bwd swapped to the GPU.
// ============================================================================
static int run_g0(int steps, float lr, float thresh, float loss_scale, float grad_clip) {
    int B = 1, S = B*SEQ;
    printf("# probe_autodiff --g0\n");
    printf("# overfit one chess position via mg_ad_forward/backward + CPU heads\n");
    printf("# DIM=%d HIDDEN=%d HEADS=%d HD=%d SEQ=%d NLAYERS=%d  (steps=%d lr=%g)\n\n",
           DIM, HIDDEN, HEADS, HD, SEQ, NLAYERS, steps, lr);

    ChessNet Wn, Gn;
    chess_net_alloc(&Wn, 0); chess_net_alloc(&Gn, 1);
    chess_net_init(&Wn, 42);
    chess_net_register(&Wn, &Gn);
    float res_alpha = 1.0f/sqrtf(2.0f*NLAYERS);

    chess_init();
    Position pos; chess_startpos(&pos);
    Move legal[MAX_MOVES]; int nlegal = chess_legal_moves(&pos, legal);
    int16_t toks16[CHESS_NUM_TOKENS]; chess_encode(&pos, toks16);
    uint16_t tokens[SEQ];
    for (int i = 0; i < CHESS_NUM_TOKENS; i++) tokens[i] = (uint16_t)toks16[i];
    for (int i = CHESS_NUM_TOKENS; i < SEQ; i++) tokens[i] = TOK_EMPTY;
    static uint8_t legal_mask[POL]; chess_legal_mask(legal, nlegal, legal_mask);
    static float tgt_pol[POL]; memset(tgt_pol, 0, sizeof tgt_pol);
    Move tgt_move = legal[0];
    int tgt_idx = chess_move_to_index(tgt_move);
    tgt_pol[tgt_idx] = 1.0f;
    float tgt_val[NWDL] = {1.0f, 0.0f, 0.0f};
    { char u[6]; chess_move_to_uci(tgt_move, u);
      printf("# G0 batch: startpos, %d legal; target=%s -> idx %d; value=Win\n\n", nlegal, u, tgt_idx); }

    float *x_in     = fmalloc((size_t)DIM*S);
    float *x_final  = fmalloc((size_t)DIM*S);
    float *dx_final = fmalloc((size_t)DIM*S);
    float *dxv      = fmalloc((size_t)DIM*S);
    float *dy       = fmalloc((size_t)DIM*S);
    chess_net_init_rmstmp(S);

    float *g_pol = fcalloc((size_t)DIM*PLANES), *g_val = fcalloc((size_t)DIM*NWDL);
    reg(Wn.W_pol, g_pol, DIM*PLANES); reg(Wn.W_val, g_val, DIM*NWDL);

    printf("## [G0] overfit (steps=%d lr=%g clip=%g loss_scale=%g thresh=%g)\n",
           steps, lr, grad_clip, loss_scale, thresh);
    float lp = 0, lv = 0; int adam_t = 0;
    for (int step = 0; step < steps; step++) {
        grads_zero();
        embed_lookup(x_in, Wn.tok_emb, tokens, DIM, SEQ);
        chess_posenc_forward(x_in, Wn.rank_emb, Wn.file_emb, Wn.misc_emb, DIM, SEQ, NBOARD);
        mg_ad_forward(B, x_in, Wn.W, Wn.rms_final, x_final);
        memset(dx_final, 0, (size_t)DIM*S*4);
        lp = chess_policy_loss(x_final, Wn.W_pol, DIM, SEQ, NBOARD, PLANES, legal_mask, tgt_pol, dx_final, g_pol);
        memset(dxv, 0, (size_t)DIM*S*4);
        lv = chess_value_loss(x_final, Wn.W_val, DIM, SEQ, NREAL, NWDL, tgt_val, dxv, g_val);
        for (int i = 0; i < DIM*S; i++) dx_final[i] += dxv[i];
        vDSP_vsmul(dx_final, 1, &loss_scale, dx_final, 1, (vDSP_Length)(DIM*S));
        vDSP_vsmul(g_pol, 1, &loss_scale, g_pol, 1, (vDSP_Length)(DIM*PLANES));
        vDSP_vsmul(g_val, 1, &loss_scale, g_val, 1, (vDSP_Length)(DIM*NWDL));
        mg_ad_backward(B, x_in, Wn.W, Wn.rms_final, dx_final, Gn.W, Gn.rms_final, dy);
        chess_posenc_backward(dy, Gn.rank_emb, Gn.file_emb, Gn.misc_emb, DIM, SEQ, NBOARD);
        embed_backward(Gn.tok_emb, dy, tokens, DIM, SEQ);
        adam_t++;
        optimizer_step(1.0f/loss_scale, grad_clip, adam_t, lr, 0.0f);

        if (step % 50 == 0 || step == steps-1)
            printf("   step %-4d  loss_pol=%.5f  loss_val=%.5f\n", step, lp, lv);
    }
    int pass = (lp < thresh) && (lv < thresh);
    printf("\n## [G0] final: loss_pol=%.5f loss_val=%.5f  (thresh %.3f)  =>  %s\n",
           lp, lv, thresh, pass ? "PASS (G0-green on autodiff)" : "FAIL");
    return pass ? 0 : 1;
}

// ============================================================================
// MODE 3 — finite-difference: the decisive ground truth.
// Central-difference d(sum(x_out*dx_final))/d(W_elem) via the CPU forward
// (g_cpu_mm=1, deterministic fp32) on a random sample of weight elements,
// then report cosine(FD, autodiff) and cosine(FD, cpu_backward) per tensor.
// ============================================================================
static float fd_loss(CLayer *Wl, CActs *acts, const float *x_in, int B,
                     float *x_pre, float *x_final, const float *rms_final,
                     float res_alpha, const float *dx_final) {
    chess_trunk_forward(Wl, acts, x_in, B, x_pre, x_final, rms_final, res_alpha, 1);
    double s = 0;
    for (int i = 0; i < DIM*B*SEQ; i++) s += (double)x_final[i] * dx_final[i];
    return (float)s;
}
static float fd_one(float *w, size_t i, float eps, CLayer *Wl, int L,
                    CActs *acts, const float *x_in, int B, float *x_pre, float *x_final,
                    const float *rms_final, float res_alpha, const float *dx_final) {
    float orig = w[i];
    w[i] = orig + eps; chess_layer_build_fused(&Wl[L]);
    float lp = fd_loss(Wl, acts, x_in, B, x_pre, x_final, rms_final, res_alpha, dx_final);
    w[i] = orig - eps; chess_layer_build_fused(&Wl[L]);
    float lm = fd_loss(Wl, acts, x_in, B, x_pre, x_final, rms_final, res_alpha, dx_final);
    w[i] = orig; chess_layer_build_fused(&Wl[L]);
    return (lp - lm) / (2*eps);
}
static void fd_sample(float *w, size_t tn, int n, unsigned *seed, float eps,
                      CLayer *Wl, int L, CActs *acts, const float *x_in, int B,
                      float *x_pre, float *x_final, const float *rms_final, float res_alpha,
                      const float *dx_final, const float *g_ad, const float *g_cpu,
                      double *dot_fa, double *na_f, double *na_a, double *dot_fc,
                      double *nc_f, double *nc_c, int *used) {
    double fmax = 0;
    float *fds = (float*)malloc(n*sizeof(float));
    size_t *idxs = (size_t*)malloc(n*sizeof(size_t));
    for (int s = 0; s < n; s++) {
        *seed = *seed * 1103515245u + 12345u;
        size_t i = ((size_t)(*seed >> 8) * tn) >> 24;
        if (i >= tn) i %= tn;
        idxs[s] = i;
        fds[s] = fd_one(w, i, eps, Wl, L, acts, x_in, B, x_pre, x_final, rms_final, res_alpha, dx_final);
        if (fabsf(fds[s]) > fmax) fmax = fabsf(fds[s]);
    }
    double clip = 0.05 * fmax;
    int u = 0;
    for (int s = 0; s < n; s++) {
        if (fabsf(fds[s]) < clip) continue;
        size_t i = idxs[s];
        float fd = fds[s], a = g_ad[i], c = g_cpu[i];
        *dot_fa += (double)fd*a; *na_f += (double)fd*fd; *na_a += (double)a*a;
        *dot_fc += (double)fd*c; *nc_f += (double)fd*fd; *nc_c += (double)c*c;
        u++;
    }
    *used = u;
    free(fds); free(idxs);
}
static int run_fd(int B) {
    int S = B*SEQ;
    int NSAMP = 64; float eps = 1e-3f;
    printf("# probe_autodiff --fd  (B=%d; finite-difference ground truth; sample=%d eps=%g)\n", B, NSAMP, eps);
    printf("# cos(FD, autodiff) is the spike's real gate; cos(FD, cpu) cross-checks the reference.\n\n");
    ChessNet Wn, G_cpu, G_ad;
    chess_net_alloc(&Wn, 0); chess_net_alloc(&G_cpu, 1); chess_net_alloc(&G_ad, 1);
    chess_net_init(&Wn, 42);
    float *x_in=fmalloc((size_t)DIM*S), *dx_final=fmalloc((size_t)DIM*S);
    float *x_pre=fmalloc((size_t)DIM*S), *x_final=fmalloc((size_t)DIM*S);
    float *dy_cpu=fmalloc((size_t)DIM*S), *dy_ad=fmalloc((size_t)DIM*S);
    float *grms_cpu=fcalloc(DIM), *grms_ad=fcalloc(DIM);
    unsigned seed = 7;
    ad_fill_randn(x_in, (size_t)DIM*S, &seed);
    ad_fill_randn(dx_final, (size_t)DIM*S, &seed);

    float *x_final_ad = fmalloc((size_t)DIM*S);
    mg_ad_forward(B, x_in, Wn.W, Wn.rms_final, x_final_ad);   // prime the autodiff graph
    free(x_final_ad);

    g_cpu_mm = 1;
    CLayer Wl[NLAYERS], Gl[NLAYERS]; CActs acts[NLAYERS];
    for (int L = 0; L < NLAYERS; L++) { Wl[L] = Wn.W[L]; Gl[L] = G_cpu.W[L]; cacts_alloc(&acts[L], S); }
    for (int L = 0; L < NLAYERS; L++) chess_layer_build_fused(&Wl[L]);
    chess_net_init_rmstmp(S);
    float res_alpha = 1.0f/sqrtf(2.0f*NLAYERS);
    chess_trunk_forward(Wl, acts, x_in, B, x_pre, x_final, Wn.rms_final, res_alpha, 1);
    chess_trunk_backward(Wl, Gl, acts, dx_final, B, x_pre, Wn.rms_final, grms_cpu, dy_cpu, res_alpha);
    mg_ad_backward(B, x_in, Wn.W, Wn.rms_final, dx_final, G_ad.W, grms_ad, dy_ad);

    printf("   %-14s %-9s %-9s %-5s\n", "tensor", "cos(FD,ad)", "cos(FD,cpu)", "n");
    double worst_fa = 2.0;
    for (int L = 0; L < NLAYERS; L++) {
        { double dfa=0,nf=0,na=0,dfc=0,nf2=0,nc=0; int used=0;
          fd_sample(Wn.W[L].Wq, (size_t)DIM*Q_DIM, NSAMP, &seed, eps, Wl, L, acts, x_in, B, x_pre, x_final, Wn.rms_final, res_alpha, dx_final, G_ad.W[L].Wq, G_cpu.W[L].Wq, &dfa,&nf,&na,&dfc,&nf2,&nc,&used);
          double ca=dfa/(sqrt(nf)*sqrt(na)+1e-30), cc=dfc/(sqrt(nf2)*sqrt(nc)+1e-30); if(ca<worst_fa)worst_fa=ca;
          char nm[16]; snprintf(nm,sizeof nm,"L%d.Wq",L); printf("   %-14s %-9.6f %-9.6f %-5d\n", nm, ca, cc, used); }
        { double dfa=0,nf=0,na=0,dfc=0,nf2=0,nc=0; int used=0;
          fd_sample(Wn.W[L].Wk, (size_t)DIM*KV_DIM, NSAMP, &seed, eps, Wl, L, acts, x_in, B, x_pre, x_final, Wn.rms_final, res_alpha, dx_final, G_ad.W[L].Wk, G_cpu.W[L].Wk, &dfa,&nf,&na,&dfc,&nf2,&nc,&used);
          double ca=dfa/(sqrt(nf)*sqrt(na)+1e-30), cc=dfc/(sqrt(nf2)*sqrt(nc)+1e-30); if(ca<worst_fa)worst_fa=ca;
          char nm[16]; snprintf(nm,sizeof nm,"L%d.Wk",L); printf("   %-14s %-9.6f %-9.6f %-5d\n", nm, ca, cc, used); }
        { double dfa=0,nf=0,na=0,dfc=0,nf2=0,nc=0; int used=0;
          fd_sample(Wn.W[L].Wo, (size_t)Q_DIM*DIM, NSAMP, &seed, eps, Wl, L, acts, x_in, B, x_pre, x_final, Wn.rms_final, res_alpha, dx_final, G_ad.W[L].Wo, G_cpu.W[L].Wo, &dfa,&nf,&na,&dfc,&nf2,&nc,&used);
          double ca=dfa/(sqrt(nf)*sqrt(na)+1e-30), cc=dfc/(sqrt(nf2)*sqrt(nc)+1e-30); if(ca<worst_fa)worst_fa=ca;
          char nm[16]; snprintf(nm,sizeof nm,"L%d.Wo",L); printf("   %-14s %-9.6f %-9.6f %-5d\n", nm, ca, cc, used); }
        { double dfa=0,nf=0,na=0,dfc=0,nf2=0,nc=0; int used=0;
          fd_sample(Wn.W[L].W2, (size_t)HIDDEN*DIM, NSAMP, &seed, eps, Wl, L, acts, x_in, B, x_pre, x_final, Wn.rms_final, res_alpha, dx_final, G_ad.W[L].W2, G_cpu.W[L].W2, &dfa,&nf,&na,&dfc,&nf2,&nc,&used);
          double ca=dfa/(sqrt(nf)*sqrt(na)+1e-30), cc=dfc/(sqrt(nf2)*sqrt(nc)+1e-30); if(ca<worst_fa)worst_fa=ca;
          char nm[16]; snprintf(nm,sizeof nm,"L%d.W2",L); printf("   %-14s %-9.6f %-9.6f %-5d\n", nm, ca, cc, used); }
    }
    const double FD_GATE = 0.998;
    printf("\n## worst cos(FD, autodiff) = %.6f  (fp32-FD gate >= %.3f)  =>  %s\n",
           worst_fa, FD_GATE, worst_fa>=FD_GATE ? "PASS" : "FAIL");
    return worst_fa >= FD_GATE ? 0 : 1;
}

// ============================================================================
int main(int argc, char **argv) {
    @autoreleasepool {
        ane_init();
        mach_timebase_info(&g_tb);
        mps_init();
        if (!g_mtl_dev) { fprintf(stderr, "[probe_autodiff] no Metal device — spike cannot run\n"); return 1; }

        int do_g0 = 0, do_grad_diff = 0, do_fd = 0, steps = 300, batch = 1;
        float lr = 1e-3f, thresh = 0.05f, loss_scale = 256.0f, grad_clip = 1.0f;
        for (int i = 1; i < argc; i++) {
            if      (!strcmp(argv[i], "--grad-diff")) do_grad_diff = 1;
            else if (!strcmp(argv[i], "--g0"))        do_g0 = 1;
            else if (!strcmp(argv[i], "--fd"))        do_fd = 1;
            else if (!strcmp(argv[i], "--batch") && i+1 < argc) batch = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--steps") && i+1 < argc) steps = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--lr")     && i+1 < argc) lr = atof(argv[++i]);
            else if (!strcmp(argv[i], "--thresh") && i+1 < argc) thresh = atof(argv[++i]);
        }
        if (do_g0) return run_g0(steps, lr, thresh, loss_scale, grad_clip);
        if (do_fd) return run_fd(batch);
        (void)do_grad_diff;
        return run_grad_diff(batch);
    }
}
