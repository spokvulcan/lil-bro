"""The shared model/run config schema.

This replaces upstream's compile-time ``#define`` model dimensions
(``training/training_dynamic/models/*.h``) with a single runtime config that is
consumed *identically* by the ANE trainer (via a generated C header) and the MLX
twin (by importing ``Config`` directly).

The architecture mirrors the upstream ANE model exactly so the MLX twin is a
faithful correctness oracle (see ``lilbro/mlx_ref``):

  - pre-norm Llama/Qwen-style decoder, **no biases**
  - RMSNorm with ``norm_eps`` inside the sqrt
  - interleaved (GPT-J) RoPE, base ``rope_theta``, pairs ``(2i, 2i+1)``
  - GQA via *tile* (query head ``h`` attends to kv head ``h % kv_heads``)
  - SwiGLU FFN ``W2(silu(W1 x) * (W3 x))``
  - residual scaling ``res_alpha = 1/sqrt(2 * n_layers)`` on both sublayers
  - tied token-embedding / LM head
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from math import sqrt
from pathlib import Path
from typing import Any

# Core fields named in the PRD: dim, n_layers, n_heads, seq, vocab,
# optimizer (adamw|muon), mtp_depth, lr, batch_tokens, seed.
OPTIMIZERS = ("adamw", "muon")


@dataclass(frozen=True)
class Config:
    """One config == one ladder rung / ablation cell.

    Required architecture + run fields are explicit; head_dim and hidden default
    to the conventional derivations when omitted (``-1`` sentinel).
    """

    name: str

    # --- model dimensions ---
    dim: int
    n_layers: int
    n_heads: int
    seq: int
    vocab: int
    kv_heads: int = -1      # default -> n_heads (MHA, no GQA)
    head_dim: int = -1      # default -> dim // n_heads
    hidden: int = -1        # FFN inner dim; default -> ~ 8/3 * dim, rounded to 64

    # --- architecture constants (match the ANE model) ---
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    # --- ablation knobs ---
    optimizer: str = "adamw"   # adamw | muon
    mtp_depth: int = 0         # 0 = off; k = predict k extra future tokens

    # --- optimization / run ---
    lr: float = 3e-4
    batch_tokens: int = 4096   # tokens per optimizer step; batch_size = batch_tokens // seq
    weight_decay: float = 0.0
    seed: int = 0

    # --- bookkeeping ---
    ckpt_path: str = ""        # default -> ane_<name>_ckpt.bin
    data_path: str = "../tinystories_data00.bin"      # train shard (data00)
    val_data_path: str = "../tinystories_data01.bin"  # held-out val shard (data01)

    def __post_init__(self) -> None:
        # Resolve sentinels (frozen dataclass -> object.__setattr__).
        if self.kv_heads == -1:
            object.__setattr__(self, "kv_heads", self.n_heads)
        if self.head_dim == -1:
            object.__setattr__(self, "head_dim", self.dim // self.n_heads)
        if self.hidden == -1:
            # SwiGLU convention ~ 8/3 * dim, rounded up to a multiple of 64.
            h = int(8 * self.dim / 3)
            object.__setattr__(self, "hidden", ((h + 63) // 64) * 64)
        if not self.ckpt_path:
            object.__setattr__(self, "ckpt_path", f"ane_{self.name}_ckpt.bin")
        self._validate()

    def _validate(self) -> None:
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {self.optimizer!r}")
        if self.n_heads % self.kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be a multiple of kv_heads ({self.kv_heads})"
            )
        if self.mtp_depth < 0:
            raise ValueError(f"mtp_depth must be >= 0, got {self.mtp_depth}")
        for nm in ("dim", "n_layers", "n_heads", "seq", "vocab", "head_dim", "hidden"):
            if getattr(self, nm) <= 0:
                raise ValueError(f"{nm} must be > 0, got {getattr(self, nm)}")
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {self.head_dim}")

    # --- derived (never stored; always consistent) ---
    @property
    def gqa_ratio(self) -> int:
        return self.n_heads // self.kv_heads

    @property
    def q_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def batch_size(self) -> int:
        return max(1, self.batch_tokens // self.seq)

    @property
    def res_alpha(self) -> float:
        """Residual scaling applied to both sublayer outputs (matches ANE)."""
        return 1.0 / sqrt(2.0 * self.n_layers)


def config_to_dict(cfg: Config) -> dict[str, Any]:
    """Stored fields only (derived properties are recomputed on load)."""
    return asdict(cfg)


def config_from_dict(d: dict[str, Any]) -> Config:
    known = {f.name for f in fields(Config)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"unknown config fields: {sorted(unknown)}")
    return Config(**d)


def save_config(cfg: Config, path: str | Path) -> None:
    Path(path).write_text(json.dumps(config_to_dict(cfg), indent=2, sort_keys=True) + "\n")


def load_config(path: str | Path) -> Config:
    return config_from_dict(json.loads(Path(path).read_text()))


# --- The scaling ladder (ROADMAP.md). Reference rungs; ablation cells override. ---
LADDER: dict[str, Config] = {
    # R0 / R1 correctness rungs: tiny, byte-level vocab, MHA.
    "r0_overfit": Config(
        name="r0_overfit", dim=64, n_layers=1, n_heads=4, head_dim=16,
        seq=64, vocab=256, hidden=128, batch_tokens=64 * 4, lr=1e-3,
    ),
    # R2 headline ablation rung: small dense, 32K vocab.
    "r2_small": Config(
        name="r2_small", dim=256, n_layers=6, n_heads=8, head_dim=32,
        seq=256, vocab=32000, hidden=768, batch_tokens=256 * 16, lr=3e-4,
    ),
    # R3 confirmation + energy rung: ~stories110m.
    "r3_110m": Config(
        name="r3_110m", dim=768, n_layers=12, n_heads=12, head_dim=64,
        seq=256, vocab=32000, hidden=2048, batch_tokens=256 * 16, lr=3e-4,
    ),
}
