// train_chess.m — Build-step 2 of chess-RL-on-ANE (ADR 0005, issue #16):
// policy (8x8x73, legal-masked) + WDL value heads + 2D rank+file posenc (RoPE off)
// + the AlphaZero loss, trained ON THE ANE, with the G0 overfit gate.
//
// As of build-step 4 (#18) the trunk (ANE matmul + forward/backward + param registry)
// lives in chess/chess_net.h, shared with the self-play learner — "the thing you train
// is the thing you proved." This file is unchanged in BEHAVIOR (it drives the shared
// trunk at B=1, byte-identical to the pre-#18 inline version); `make g0` re-verifies it.
//
// WHAT RUNS WHERE (the CPU/ANE split, ADR 0004 / [[ane-resident-training-cpu-floor]]):
//   ANE (fp16): every trunk matmul — QKV/Wo/W1/W3/W2 forward AND the dx backward
//     matmuls — via gen_dyn_matmul_mil(ic,oc,seq), the proven substrate, un-fused.
//   CPU (fp32): RMSNorm, attention softmax (attn_cpu_*, RoPE OFF), SiLU, dW (cblas),
//     the embedding + 2D posenc, the policy/value heads, the AZ loss, optimizer — the
//     irreducible CPU floor. Slice 5 uses Muon for canonical 2D trunk matrices and
//     AdamW for embeddings, RMSNorm gains, and policy/value heads.
//
// G0 GATE (the discipline — a MEASURED gate): `./train_chess --overfit` pins ONE
// chess position -> (one-hot target policy, one-hot target value) and trains until
// BOTH cross-entropies collapse to ~0. Exit 0 on pass, 1 on fail.
//
// Build: make train_chess   ·   Gate: make g0

#include "mil_dynamic.h"          // pulls io.h + config.h (DIM/SEQ/... from chess_g0.h)
#include "cpu_ops.h"              // rmsnorm(_bwd), attn_cpu_*, adam_update, embed_*
#include "chess/chess.h"          // engine + codec (#15): encode, legal moves/mask, move index
#include "chess/chess_heads.h"    // NEW heads: posenc, policy, value, L2 (FD-gated)
#include "chess/chess_net.h"      // shared trunk: ANE matmul, fwd/bwd, shapes, param registry
#include <stdio.h>
#include <string.h>

static float frand(void) { return (float)(2*drand48()-1); }

int main(int argc, char *argv[]) {
    @autoreleasepool {
        ane_init();
        mach_timebase_info(&g_tb);

        int do_overfit = 0, do_selfcheck = 0, steps = 600;
        float lr = 1e-3f, thresh = 0.05f, loss_scale = 256.0f, grad_clip = 1.0f, wd = 0.0f;
        float vw = 1.0f, l2 = 0.0f;   // value-loss weight; L2 (0 for the gate so CEs -> 0)
        int opt_is_muon = OPTIMIZER_IS_MUON;
        for (int i = 1; i < argc; i++) {
            if      (!strcmp(argv[i], "--overfit"))   do_overfit = 1;
            else if (!strcmp(argv[i], "--selfcheck")) do_selfcheck = 1;
            else if (!strcmp(argv[i], "--steps") && i+1<argc) steps = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--lr")    && i+1<argc) lr = atof(argv[++i]);
            else if (!strcmp(argv[i], "--thresh")&& i+1<argc) thresh = atof(argv[++i]);
            else if (!strcmp(argv[i], "--clip")  && i+1<argc) grad_clip = atof(argv[++i]);
            else if (!strcmp(argv[i], "--l2")    && i+1<argc) l2 = atof(argv[++i]);
            else if (!strcmp(argv[i], "--opt") && i+1<argc) {
                const char *o = argv[++i];
                if (!strcmp(o, "muon")) opt_is_muon = 1;
                else if (!strcmp(o, "adamw")) opt_is_muon = 0;
                else { printf("unknown --opt %s (use adamw|muon)\n", o); return 1; }
            }
        }
        chess_optimizer_set_muon(opt_is_muon);
        if (do_overfit) do_selfcheck = 1;  // the gate always proves the substrate is live first
        if (!do_overfit && !do_selfcheck) do_selfcheck = 1;

        printf("# chess G0 trainer — DIM=%d HEADS=%d HD=%d HIDDEN=%d L=%d SEQ=%d VOCAB=%d\n",
               DIM, HEADS, HD, HIDDEN, NLAYERS, SEQ, VOCAB);
        printf("# trunk matmuls on ANE (fp16) via gen_dyn_matmul_mil; heads/loss/norm/attn on CPU (fp32); optimizer=%s\n\n",
               chess_optimizer_name());

        float res_alpha = 1.0f/sqrtf(2.0f*NLAYERS);

        // ---- allocate net + grads + acts (B=1 trunk: acts sized for SEQ) ----
        CLayer W[NLAYERS], G[NLAYERS]; CActs acts[NLAYERS];
        for (int L = 0; L < NLAYERS; L++) { clayer_alloc(&W[L]); clayer_calloc(&G[L]); cacts_alloc(&acts[L], SEQ); }
        float *rms_final=fmalloc(DIM), *grms_final=fcalloc(DIM);
        float *tok_emb=fmalloc((size_t)VOCAB*DIM), *g_tok=fcalloc((size_t)VOCAB*DIM);
        float *rank_emb=fmalloc(8*DIM), *g_rank=fcalloc(8*DIM);
        float *file_emb=fmalloc(8*DIM), *g_file=fcalloc(8*DIM);
        float *misc_emb=fmalloc((size_t)NMISC*DIM), *g_misc=fcalloc((size_t)NMISC*DIM);
        float *W_pol=fmalloc((size_t)DIM*PLANES), *g_pol=fcalloc((size_t)DIM*PLANES);
        float *W_val=fmalloc((size_t)DIM*NWDL), *g_val=fcalloc((size_t)DIM*NWDL);

        // ---- init (mirror train.m scales; heads/embeds small like the LM head) ----
        srand48(42);
        float sd=1.0f/sqrtf(DIM), sq=1.0f/sqrtf(Q_DIM), sh=1.0f/sqrtf(HIDDEN), rs=res_alpha, e=0.02f;
        for (int L=0;L<NLAYERS;L++) {
            for (int i=0;i<DIM*Q_DIM;i++) W[L].Wq[i]=sd*frand();
            for (int i=0;i<DIM*KV_DIM;i++){ W[L].Wk[i]=sd*frand(); W[L].Wv[i]=sd*frand(); }
            for (int i=0;i<Q_DIM*DIM;i++) W[L].Wo[i]=sq*rs*frand();
            for (int i=0;i<DIM*HIDDEN;i++){ W[L].W1[i]=sh*frand(); W[L].W3[i]=sh*frand(); }
            for (int i=0;i<HIDDEN*DIM;i++) W[L].W2[i]=sd*rs*frand();
            for (int i=0;i<DIM;i++){ W[L].rms_att[i]=1.0f; W[L].rms_ffn[i]=1.0f; }
        }
        for (int i=0;i<DIM;i++) rms_final[i]=1.0f;
        for (int i=0;i<(int)(VOCAB*DIM);i++) tok_emb[i]=e*frand();
        for (int i=0;i<8*DIM;i++){ rank_emb[i]=e*frand(); file_emb[i]=e*frand(); }
        for (int i=0;i<NMISC*DIM;i++) misc_emb[i]=e*frand();
        for (int i=0;i<DIM*PLANES;i++) W_pol[i]=e*frand();
        for (int i=0;i<DIM*NWDL;i++)  W_val[i]=e*frand();

        // ---- register params for the optimizer (order is irrelevant) ----
        for (int L=0;L<NLAYERS;L++) {
            reg_muon2d(W[L].Wq,G[L].Wq,DIM,Q_DIM); reg_muon2d(W[L].Wk,G[L].Wk,DIM,KV_DIM); reg_muon2d(W[L].Wv,G[L].Wv,DIM,KV_DIM);
            reg_muon2d(W[L].Wo,G[L].Wo,Q_DIM,DIM); reg_muon2d(W[L].W1,G[L].W1,DIM,HIDDEN); reg_muon2d(W[L].W2,G[L].W2,HIDDEN,DIM);
            reg_muon2d(W[L].W3,G[L].W3,DIM,HIDDEN); reg(W[L].rms_att,G[L].rms_att,DIM); reg(W[L].rms_ffn,G[L].rms_ffn,DIM);
        }
        reg(rms_final,grms_final,DIM); reg(tok_emb,g_tok,VOCAB*DIM);
        reg(rank_emb,g_rank,8*DIM); reg(file_emb,g_file,8*DIM); reg(misc_emb,g_misc,NMISC*DIM);
        reg(W_pol,g_pol,DIM*PLANES); reg(W_val,g_val,DIM*NWDL);

        // Build the fused forward-only weights (QKV, W1/W3) before any forward; rebuilt after
        // every optimizer step below. [iter 6]
        for (int L=0;L<NLAYERS;L++) chess_layer_build_fused(&W[L]);

        // ============================================================
        // Build the fixed G0 batch from the engine (ONE position).
        // ============================================================
        chess_init();
        Position pos; chess_startpos(&pos);
        Move legal[MAX_MOVES]; int nlegal = chess_legal_moves(&pos, legal);
        int16_t toks16[CHESS_NUM_TOKENS]; chess_encode(&pos, toks16);
        uint16_t tokens[SEQ];
        for (int i=0;i<CHESS_NUM_TOKENS;i++) tokens[i]=(uint16_t)toks16[i];
        for (int i=CHESS_NUM_TOKENS;i<SEQ;i++) tokens[i]=TOK_EMPTY;
        // target policy: one-hot on a fixed legal move (=> CE floor 0). target value: one-hot Win.
        static uint8_t legal_mask[POL]; chess_legal_mask(legal, nlegal, legal_mask);
        static float tgt_pol[POL]; memset(tgt_pol,0,sizeof tgt_pol);
        Move tgt_move = legal[0];
        int tgt_idx = chess_move_to_index(tgt_move);
        tgt_pol[tgt_idx] = 1.0f;
        float tgt_val[NWDL] = {1.0f, 0.0f, 0.0f};  // Win (one-hot)
        { char u[6]; chess_move_to_uci(tgt_move,u);
          printf("# G0 batch: startpos, %d legal moves; target move=%s -> policy idx %d; target value=Win\n\n",
                 nlegal, u, tgt_idx); }

        // input embedding + 2D posenc (recomputed each step from the live tables)
        float *x_in=fmalloc(DIM*SEQ), *x_pre=fmalloc(DIM*SEQ), *x_final=fmalloc(DIM*SEQ);
        float *dx_final=fmalloc(DIM*SEQ), *dy_in=fmalloc(DIM*SEQ);

        // ------------------------------------------------------------
        // SELF-CHECK: ANE matmul vs cblas, and ANE-trunk vs CPU-trunk.
        // ------------------------------------------------------------
        if (do_selfcheck) {
            const double COS_MIN = 0.99;
            double worst_cos = 1.0, cosv;
            printf("## [selfcheck] ANE matmul vs cblas (cos; fp16 expected ~0.999+)\n");
            int shapes[][2] = {{DIM,DIM},{DIM,HIDDEN},{HIDDEN,DIM}};
            for (int s=0;s<3;s++) {
                int ic=shapes[s][0], oc=shapes[s][1];
                float *xx=fmalloc((size_t)ic*SEQ), *ww=fmalloc((size_t)ic*oc);
                float *ya=fmalloc((size_t)oc*SEQ), *yc=fmalloc((size_t)oc*SEQ);
                for (int i=0;i<ic*SEQ;i++) xx[i]=0.1f*frand();
                for (int i=0;i<ic*oc;i++)  ww[i]=0.1f*frand();
                g_cpu_mm=0; ane_matmul(ic,oc,SEQ,xx,ww,ya);
                g_cpu_mm=1; ane_matmul(ic,oc,SEQ,xx,ww,yc); g_cpu_mm=0;
                double dot=0,na=0,nc=0;
                for (int i=0;i<oc*SEQ;i++){ dot+=(double)ya[i]*yc[i]; na+=(double)ya[i]*ya[i]; nc+=(double)yc[i]*yc[i]; }
                cosv = dot/(sqrt(na)*sqrt(nc)+1e-30); if (cosv<worst_cos) worst_cos=cosv;
                printf("   matmul ic=%-4d oc=%-4d : cos=%.6f\n", ic, oc, cosv);
                free(xx);free(ww);free(ya);free(yc);
            }
            // full trunk: ANE vs CPU forward (cos of x_final)
            embed_lookup(x_in, tok_emb, tokens, DIM, SEQ);
            chess_posenc_forward(x_in, rank_emb, file_emb, misc_emb, DIM, SEQ, NBOARD);
            float *xf_cpu=fmalloc(DIM*SEQ);
            g_cpu_mm=1; chess_trunk_forward(W, acts, x_in, 1, x_pre, xf_cpu, rms_final, res_alpha, 1);
            g_cpu_mm=0; chess_trunk_forward(W, acts, x_in, 1, x_pre, x_final, rms_final, res_alpha, 1);
            double dot=0,na=0,nc=0;
            for (int i=0;i<DIM*SEQ;i++){ dot+=(double)x_final[i]*xf_cpu[i]; na+=(double)x_final[i]*x_final[i]; nc+=(double)xf_cpu[i]*xf_cpu[i]; }
            cosv = dot/(sqrt(na)*sqrt(nc)+1e-30); if (cosv<worst_cos) worst_cos=cosv;
            printf("   full %d-layer trunk fwd: cos(ANE, CPU)=%.6f\n", NLAYERS, cosv);
            free(xf_cpu);
            // BACKWARD substrate check: the dx-matmuls must match CPU too.
            float *dxr=fmalloc(DIM*SEQ), *dyA=fmalloc(DIM*SEQ), *dyC=fmalloc(DIM*SEQ);
            for (int i=0;i<DIM*SEQ;i++) dxr[i]=0.1f*frand();
            grads_zero(); g_cpu_mm=1; chess_trunk_backward(W,G,acts,dxr,1,x_pre,rms_final,grms_final,dyC,res_alpha);
            grads_zero(); g_cpu_mm=0; chess_trunk_backward(W,G,acts,dxr,1,x_pre,rms_final,grms_final,dyA,res_alpha);
            grads_zero();   // leave grads clean for training
            dot=0;na=0;nc=0;
            for (int i=0;i<DIM*SEQ;i++){ dot+=(double)dyA[i]*dyC[i]; na+=(double)dyA[i]*dyA[i]; nc+=(double)dyC[i]*dyC[i]; }
            cosv = dot/(sqrt(na)*sqrt(nc)+1e-30); if (cosv<worst_cos) worst_cos=cosv;
            printf("   full %d-layer trunk bwd: cos(ANE, CPU)=%.6f\n", NLAYERS, cosv);
            free(dxr);free(dyA);free(dyC);
            if (worst_cos < COS_MIN) {
                printf("   [selfcheck] FAIL: worst cos %.6f < %.2f — the ANE path is broken/stale\n\n", worst_cos, COS_MIN);
                return 1;
            }
            printf("   [selfcheck] substrate OK (min cos %.6f)\n\n", worst_cos);
            if (!do_overfit) return 0;
        }

        // ------------------------------------------------------------
        // G0 OVERFIT GATE: drive BOTH cross-entropies to ~0 on the ANE.
        // ------------------------------------------------------------
        printf("## [G0] overfit one batch (steps=%d lr=%g clip=%g loss_scale=%g thresh=%g)\n",
               steps, lr, grad_clip, loss_scale, thresh);
        float lp=0, lv=0; int adam_t=0;
        for (int step=0; step<steps; step++) {
            grads_zero();
            // forward
            embed_lookup(x_in, tok_emb, tokens, DIM, SEQ);
            chess_posenc_forward(x_in, rank_emb, file_emb, misc_emb, DIM, SEQ, NBOARD);
            chess_trunk_forward(W, acts, x_in, 1, x_pre, x_final, rms_final, res_alpha, 1);
            // heads + AZ loss (CPU); dx_final accumulates policy + value gradients
            memset(dx_final, 0, DIM*SEQ*4);
            lp = chess_policy_loss(x_final, W_pol, DIM, SEQ, NBOARD, PLANES, legal_mask, tgt_pol, dx_final, g_pol);
            float *dxv=fcalloc(DIM*SEQ);
            lv = chess_value_loss(x_final, W_val, DIM, SEQ, NREAL, NWDL, tgt_val, dxv, g_val);
            for (int i=0;i<DIM*SEQ;i++) dx_final[i] += vw*dxv[i];
            for (int i=0;i<DIM*NWDL;i++) g_val[i] *= vw;
            free(dxv);
            float l2pen = 0, l2s = l2*loss_scale;
            if (l2 > 0) {
                for (int L=0;L<NLAYERS;L++){
                    l2pen += chess_l2_penalty(W[L].Wq,DIM*Q_DIM,l2s,G[L].Wq);
                    l2pen += chess_l2_penalty(W[L].Wk,DIM*KV_DIM,l2s,G[L].Wk);
                    l2pen += chess_l2_penalty(W[L].Wv,DIM*KV_DIM,l2s,G[L].Wv);
                    l2pen += chess_l2_penalty(W[L].Wo,Q_DIM*DIM,l2s,G[L].Wo);
                    l2pen += chess_l2_penalty(W[L].W1,DIM*HIDDEN,l2s,G[L].W1);
                    l2pen += chess_l2_penalty(W[L].W2,HIDDEN*DIM,l2s,G[L].W2);
                    l2pen += chess_l2_penalty(W[L].W3,DIM*HIDDEN,l2s,G[L].W3);
                }
                l2pen /= loss_scale;
            }
            // loss-scaling: scale the grad entering the trunk + the head weight grads
            vDSP_vsmul(dx_final,1,&loss_scale,dx_final,1,(vDSP_Length)(DIM*SEQ));
            vDSP_vsmul(g_pol,1,&loss_scale,g_pol,1,(vDSP_Length)(DIM*PLANES));
            vDSP_vsmul(g_val,1,&loss_scale,g_val,1,(vDSP_Length)(DIM*NWDL));
            // trunk backward (ANE dx matmuls + CPU dW), then posenc + embed backward
            chess_trunk_backward(W, G, acts, dx_final, 1, x_pre, rms_final, grms_final, dy_in, res_alpha);
            chess_posenc_backward(dy_in, g_rank, g_file, g_misc, DIM, SEQ, NBOARD);
            embed_backward(g_tok, dy_in, tokens, DIM, SEQ);
            // optimizer: unscale (1/loss_scale), global-clip, then the selected V4 split.
            adam_t++;
            optimizer_step(1.0f/loss_scale, grad_clip, adam_t, lr, wd);
            for (int L=0;L<NLAYERS;L++) chess_layer_build_fused(&W[L]);   // keep fused fwd weights in sync [iter 6]

            if (step % 50 == 0 || step == steps-1) {
                if (l2 > 0) printf("   step %-4d  loss_pol=%.5f  loss_val=%.5f  l2=%.5f\n", step, lp, lv, l2pen);
                else        printf("   step %-4d  loss_pol=%.5f  loss_val=%.5f\n", step, lp, lv);
            }
        }

        int pass = (lp < thresh) && (lv < thresh);
        printf("\n## [G0] final: loss_pol=%.5f loss_val=%.5f  (thresh %.3f)  =>  %s\n",
               lp, lv, thresh, pass ? "PASS (G0-green)" : "FAIL");
        return pass ? 0 : 1;
    }
}
