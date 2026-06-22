// chess_g0.h — hand-written tiny chess policy/value net config (NOT generated).
//
// Build-step 2 of the chess-RL-on-ANE build order (ADR 0005, issue #16). This is
// the current chess smoke net: a 6-layer MHA transformer
// trunk at chess shapes (seq 77 -> 96 padded, ADR 0005 decision 5). Scale here is
// for *correctness* (overfit one position), not strength (ADR 0005 decision 12:
// the optional ~2-5M proof-of-life, not r2_small).
//
// Selected via the Makefile `-include models/chess_g0.h` like any model header, so
// config.h's #error guard is satisfied and io.h/mil_dynamic.h derive sizes from it.
// Unlike models/gen_*.h this is hand-written and committed (it is not emitted by
// lilbro.configs.emit_c).
#pragma once

#define MODEL_NAME "chess_g0"

// --- trunk shape (MHA: HEADS==KV_HEADS, HD=32 => Q_DIM==KV_DIM==DIM) ----------
#define DIM 256
#define HIDDEN 512
#define HEADS 8
#define KV_HEADS 8
#define HD 32                 // explicit head_dim (HEADS*HD == DIM here)
#define GQA_RATIO (HEADS / KV_HEADS)
#define Q_DIM (HEADS * HD)
#define KV_DIM (KV_HEADS * HD)
#define SEQ 96                // CHESS_PAD_TOKENS (77 real chess tokens, mult-of-32 pad)
#define NLAYERS 6
#define VOCAB 38              // CHESS_VOCAB (chess.h)

#define CKPT_PATH "ane_chess_g0_ckpt.bin"   // unused by train_chess (G0 is in-memory)
#define DEFAULT_DATA_PATH ""                 // chess builds its batch from the engine

// --- shared-config extensions (match the emit_c.py / gen_r0_overfit.h defaults) ---
#ifndef NORM_EPS
#define NORM_EPS 1e-05f
#endif
#ifndef ROPE_THETA
#define ROPE_THETA 10000.0f
#endif
#ifndef MTP_DEPTH
#define MTP_DEPTH 0
#endif
#ifndef OPTIMIZER_IS_MUON
#define OPTIMIZER_IS_MUON 1   // Slice 5 default: V4 split (2D trunk Muon, rest AdamW)
#endif

// --- DeepSeek-V4 ablation knobs: ALL OFF for chess v1 (ADR 0005 decision 11:
//     plain transformer + Muon; mHC/MTP/qk-norm/attn-sink/swiglu-clamp default-off). ---
#ifndef QK_NORM
#define QK_NORM 0
#endif
#ifndef ATTN_SINK
#define ATTN_SINK 0
#endif
#ifndef SWIGLU_CLAMP
#define SWIGLU_CLAMP 0
#endif
// 2D rank+file posenc replaces 1D RoPE (ADR 0005 decision 10). The chess trunk
// never calls the RoPE path; ROPE_ROTARY_DIMS=0 makes ROPE_ROTARY_EFF=0 so any
// shared RoPE helper that is reached is an exact identity (belt-and-suspenders).
#ifndef ROPE_ROTARY_DIMS
#define ROPE_ROTARY_DIMS 0
#endif
#ifndef N_HC
#define N_HC 1                // mHC residual-stream width; 1 = off
#endif
