"""Emit a ``models/*.h``-compatible C header from a shared ``Config``.

This is the **ANE consumer** of the shared config: it produces exactly the
``#define`` set the upstream Objective-C trainer expects
(``training/training_dynamic/models/*.h``), so the same config that drives the
MLX twin can drive the ANE trainer without editing source per ablation cell.

It additionally emits the architecture constants the trainer currently hardcodes
(``NORM_EPS``, ``ROPE_THETA``, ``MTP_DEPTH``, ``OPTIMIZER``) plus the DeepSeek-V4
ablation knobs (``QK_NORM``, ``ATTN_SINK``, ``SWIGLU_CLAMP``, ``ROPE_ROTARY_DIMS``,
``N_HC``; PRD #2), all under ``#ifndef`` guards so the hand-written model headers
(``qwen3_06b.h``, ``stories110m.h``) — which predate the shared config — still
compile against the trainer's ``config.h`` fallbacks. Every knob defaults
off / identity, so a freshly-emitted header drives the *exact* plain transformer
until a later slice wires a knob to behavior.
"""

from __future__ import annotations

from pathlib import Path

from .schema import Config

_TEMPLATE = """\
// {name}.h — GENERATED from lilbro/configs by emit_c.py. Do not edit by hand.
// Regenerate: python -m lilbro.configs.emit_c {name}
#pragma once

#define MODEL_NAME "{name}"

#define DIM {dim}
#define HIDDEN {hidden}
#define HEADS {n_heads}
#define KV_HEADS {kv_heads}
#define HD {head_dim}               // explicit head_dim (NOT DIM/HEADS under GQA)
#define GQA_RATIO (HEADS / KV_HEADS)
#define Q_DIM (HEADS * HD)
#define KV_DIM (KV_HEADS * HD)
#define SEQ {seq}
#define NLAYERS {n_layers}
#define VOCAB {vocab}

#define CKPT_PATH "{ckpt_path}"
#define DEFAULT_DATA_PATH "{data_path}"

// --- shared-config extensions (trainer currently hardcodes these; see emit_c.py) ---
#ifndef NORM_EPS
#define NORM_EPS {norm_eps_c}f
#endif
#ifndef ROPE_THETA
#define ROPE_THETA {rope_theta_c}f
#endif
#ifndef MTP_DEPTH
#define MTP_DEPTH {mtp_depth}
#endif
#ifndef OPTIMIZER_IS_MUON
#define OPTIMIZER_IS_MUON {is_muon}   // 0 = adamw, 1 = muon
#endif

// --- DeepSeek-V4 ablation knobs (PRD #2; default off/identity — see lilbro/configs/schema.py) ---
#ifndef QK_NORM
#define QK_NORM {qk_norm}              // 0 = off; 1 = Q/KV RMSNorm before scores (#7)
#endif
#ifndef ATTN_SINK
#define ATTN_SINK {attn_sink}          // 0 = off; 1 = learnable per-head sink logit (#8)
#endif
#ifndef SWIGLU_CLAMP
#define SWIGLU_CLAMP {swiglu_clamp}    // 0 = off; 1 = clamp SwiGLU linear/gate (#9)
#endif
#ifndef ROPE_ROTARY_DIMS
#define ROPE_ROTARY_DIMS {rope_rotary_dims}   // partial RoPE: rotate min(HD, this) (#10)
#endif
#ifndef N_HC
#define N_HC {n_hc}                    // mHC residual-stream width; 1 = off (#11)
#endif
"""


def emit_c_header(cfg: Config) -> str:
    """Render the C header text for ``cfg`` (deterministic; no I/O)."""
    return _TEMPLATE.format(
        name=cfg.name,
        dim=cfg.dim,
        hidden=cfg.hidden,
        n_heads=cfg.n_heads,
        kv_heads=cfg.kv_heads,
        head_dim=cfg.head_dim,
        seq=cfg.seq,
        n_layers=cfg.n_layers,
        vocab=cfg.vocab,
        ckpt_path=cfg.ckpt_path,
        data_path=cfg.data_path,
        norm_eps_c=repr(float(cfg.norm_eps)),
        rope_theta_c=repr(float(cfg.rope_theta)),
        mtp_depth=cfg.mtp_depth,
        is_muon=1 if cfg.optimizer == "muon" else 0,
        qk_norm=1 if cfg.qk_norm else 0,
        attn_sink=1 if cfg.attn_sink else 0,
        swiglu_clamp=1 if cfg.swiglu_clamp else 0,
        rope_rotary_dims=cfg.rope_rotary_dims,
        n_hc=cfg.n_hc,
    )


def write_c_header(cfg: Config, out_dir: str | Path) -> Path:
    """Write ``<out_dir>/gen_<name>.h`` and return its path.

    The ``gen_`` prefix keeps generated headers out of git (see .gitignore) and
    distinct from the hand-written upstream model headers.
    """
    out = Path(out_dir) / f"gen_{cfg.name}.h"
    out.write_text(emit_c_header(cfg))
    return out


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    import sys

    from .schema import LADDER, load_config

    arg = sys.argv[1] if len(sys.argv) > 1 else "r0_overfit"
    cfg = LADDER[arg] if arg in LADDER else load_config(arg)
    sys.stdout.write(emit_c_header(cfg))
