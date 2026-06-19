"""Read ANE training checkpoints into shared-config params (for generation).

``train.m`` writes a v4 checkpoint (``save_checkpoint`` / ``CkptHdr`` in
``config.h``): a 96-byte header, then for each layer the 9 weight tensors
followed by their Adam ``m``/``v`` state, then ``rms_final`` (+adam), then the
tied ``embed`` (+adam). For *generation* we only need the weights, so this reader
walks that layout and skips the optimizer state, returning a ``{name: ndarray}``
dict keyed by ``param_spec`` — exactly what the MLX twin's sampler consumes.

The header is self-describing (dims are stored), but we still pass a ``Config``:
it carries the architecture constants the header omits (``rope_theta``,
``norm_eps``) and lets us *assert* the checkpoint matches the config we think we
trained — a silent dim mismatch would otherwise read garbage.

Layout/round-trip is pinned by ``tests/test_ane_checkpoint.py`` against
``write_ckpt`` (a faithful mirror of ``save_checkpoint``); the C-writer ↔
Python-reader path is exercised by the real ANE smoke run.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from lilbro.configs import Config
from lilbro.mlx_ref import param_shapes

CKPT_MAGIC = 0x424C5A54
CKPT_VERSION = 4

# CkptHdr (config.h): 10 ints, 2 floats, 3 doubles, 6 ints — natural alignment
# inserts no padding (the doubles land at offset 48, already 8-aligned), so the
# packed little-endian layout matches the C struct byte-for-byte at 96 bytes.
_HDR_FMT = "<10i2f3d6i"
HEADER_BYTES = struct.calcsize(_HDR_FMT)  # 96
_HDR_FIELDS = (
    "magic", "version", "step", "total_steps",
    "n_layers", "vocab_size", "dim", "hidden_dim", "n_heads", "seq_len",
    "lr", "loss",
    "cum_compile", "cum_train", "cum_wall",
    "cum_steps", "cum_batches", "adam_t",
    "kv_heads", "head_dim", "q_dim",
)

# Order of the 9 weight tensors within a layer block (== param_spec block order,
# == train.m save order).
_BLOCK = ["wq", "wk", "wv", "wo", "w1", "w2", "w3", "rms_att", "rms_ffn"]


def read_header(path: str | Path) -> dict:
    """Parse the 96-byte checkpoint header into a field dict."""
    with open(path, "rb") as f:
        raw = f.read(HEADER_BYTES)
    if len(raw) < HEADER_BYTES:
        raise ValueError(f"{path}: truncated header ({len(raw)} < {HEADER_BYTES})")
    h = dict(zip(_HDR_FIELDS, struct.unpack(_HDR_FMT, raw)))
    if h["magic"] != CKPT_MAGIC:
        raise ValueError(f"{path}: bad magic 0x{h['magic']:08X} (not a lil-bro ckpt)")
    if h["version"] != CKPT_VERSION:
        raise ValueError(f"{path}: unsupported ckpt version {h['version']}")
    return h


def _check_against_config(h: dict, cfg: Config) -> None:
    want = {
        "n_layers": cfg.n_layers, "vocab_size": cfg.vocab, "dim": cfg.dim,
        "hidden_dim": cfg.hidden, "n_heads": cfg.n_heads, "seq_len": cfg.seq,
        "kv_heads": cfg.kv_heads, "head_dim": cfg.head_dim, "q_dim": cfg.q_dim,
    }
    bad = {k: (h[k], v) for k, v in want.items() if h[k] != v}
    if bad:
        raise ValueError(
            f"checkpoint dims disagree with config {cfg.name!r}: "
            + ", ".join(f"{k}: ckpt={a} cfg={b}" for k, (a, b) in bad.items()))


def load_ckpt_params(path: str | Path, cfg: Config) -> dict[str, np.ndarray]:
    """Load model **weights** (not Adam state) into a ``{name: float32}`` dict.

    Keys/shapes follow ``param_spec(cfg)``: ``embed``, per-layer
    ``layer.<l>.{wq,wk,wv,wo,w1,w2,w3,rms_att,rms_ffn}``, then ``rms_final``.
    """
    h = read_header(path)
    _check_against_config(h, cfg)
    shapes = param_shapes(cfg)

    with open(path, "rb") as f:
        f.seek(HEADER_BYTES)
        flat = np.fromfile(f, dtype=np.float32)

    out: dict[str, np.ndarray] = {}
    i = 0

    def take(name: str) -> None:
        nonlocal i
        n = int(np.prod(shapes[name]))
        out[name] = flat[i:i + n].reshape(shapes[name]).copy()
        i += n

    def skip(name: str, copies: int = 2) -> None:  # Adam m+v
        nonlocal i
        i += copies * int(np.prod(shapes[name]))

    for L in range(cfg.n_layers):
        for kind in _BLOCK:
            take(f"layer.{L}.{kind}")
        for kind in _BLOCK:
            skip(f"layer.{L}.{kind}")
    take("rms_final"); skip("rms_final")
    take("embed"); skip("embed")

    if i != flat.size:
        raise ValueError(
            f"{path}: consumed {i} floats but file has {flat.size} after header "
            f"(layout mismatch for config {cfg.name!r})")
    return out


def write_ckpt(path: str | Path, params: dict[str, np.ndarray], cfg: Config,
               *, step: int = 0, loss: float = 0.0, adam_t: int = 0) -> Path:
    """Write a v4 checkpoint — a faithful mirror of ``train.m:save_checkpoint``.

    Adam state is written zeroed (generation/seeding never needs it). Used by the
    round-trip test and as a tool to seed a run from shared-config params.
    """
    shapes = param_shapes(cfg)

    def w(arr: np.ndarray, name: str) -> np.ndarray:
        a = np.ascontiguousarray(arr, dtype=np.float32)
        if a.shape != shapes[name]:
            raise ValueError(f"{name}: shape {a.shape} != {shapes[name]}")
        return a

    hdr = struct.pack(
        _HDR_FMT, CKPT_MAGIC, CKPT_VERSION, step, 0,
        cfg.n_layers, cfg.vocab, cfg.dim, cfg.hidden, cfg.n_heads, cfg.seq,
        float(cfg.lr), float(loss), 0.0, 0.0, 0.0, 0, 0, adam_t,
        cfg.kv_heads, cfg.head_dim, cfg.q_dim)

    with open(path, "wb") as f:
        f.write(hdr)
        for L in range(cfg.n_layers):
            for kind in _BLOCK:
                name = f"layer.{L}.{kind}"
                f.write(w(params[name], name).tobytes())
            for kind in _BLOCK:               # zeroed Adam m, v
                name = f"layer.{L}.{kind}"
                z = np.zeros(shapes[name], dtype=np.float32)
                f.write(z.tobytes()); f.write(z.tobytes())
        f.write(w(params["rms_final"], "rms_final").tobytes())
        z = np.zeros(shapes["rms_final"], dtype=np.float32)
        f.write(z.tobytes()); f.write(z.tobytes())
        f.write(w(params["embed"], "embed").tobytes())
        z = np.zeros(shapes["embed"], dtype=np.float32)
        f.write(z.tobytes()); f.write(z.tobytes())
    return Path(path)


def generate_from_ckpt(path: str | Path, cfg: Config, prompt, n_new: int,
                       temperature: float = 0.0, seed: int = 0) -> np.ndarray:
    """Load an ANE checkpoint and autoregressively sample ``n_new`` tokens.

    Generation runs on the MLX twin (a faithful oracle of the same model), so we
    reuse the tested sampler rather than reimplement decode on the ANE side.
    """
    from lilbro.eval.sample import generate  # lazy: pulls mlx only when sampling
    params = load_ckpt_params(path, cfg)
    return generate(params, prompt, cfg, n_new, temperature=temperature, seed=seed)
