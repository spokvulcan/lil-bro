"""Flat float32 (de)serialization of dense model state for the ANE bridge.

One layout, two consumers (mirrors the config contract): this module is the
Python end; ``train.m`` is the C end. Keep them in lockstep — the order here is
``param_spec`` order and the C side reads/writes the identical sequence.
"""

from __future__ import annotations

import numpy as np

from lilbro.configs import Config
from lilbro.mlx_ref import param_shapes


def flat_order(cfg: Config) -> list[str]:
    """Parameter names in canonical on-disk order (== dense ``param_spec``).

    Rejects MTP configs: the ANE trainer has no MTP module, so there is no C-side
    layout to match and a silent encode would corrupt the R1 diff."""
    if cfg.mtp_depth > 0:
        raise ValueError(
            f"ane_bridge does not support MTP (mtp_depth={cfg.mtp_depth}); the "
            "ANE trainer has no MTP path. R1 covers the dense base + GQA only.")
    return list(param_shapes(cfg))


def pack(params: dict[str, np.ndarray], cfg: Config) -> bytes:
    """Concatenate ``params`` into the flat float32 blob (row-major, in order)."""
    order = flat_order(cfg)
    shapes = param_shapes(cfg)
    missing = set(order) - set(params)
    if missing:
        raise ValueError(f"params missing {sorted(missing)}")
    chunks = []
    for name in order:
        arr = np.ascontiguousarray(params[name], dtype=np.float32)
        if arr.shape != shapes[name]:
            raise ValueError(
                f"{name}: shape {arr.shape} != expected {shapes[name]}")
        chunks.append(arr.ravel(order="C"))
    return np.concatenate(chunks).tobytes()


def unpack(blob: bytes, cfg: Config) -> dict[str, np.ndarray]:
    """Inverse of :func:`pack`: split the flat blob back into named tensors."""
    shapes = param_shapes(cfg)
    expected = 4 * sum(int(np.prod(s)) for s in shapes.values())
    if len(blob) != expected:
        raise ValueError(
            f"blob size {len(blob)} != expected {expected} bytes for {cfg.name}")
    flat = np.frombuffer(blob, dtype=np.float32)
    out: dict[str, np.ndarray] = {}
    offset = 0
    for name in flat_order(cfg):
        n = int(np.prod(shapes[name]))
        out[name] = flat[offset:offset + n].reshape(shapes[name]).copy()
        offset += n
    return out


# read_flat and read_grads are the same operation (an init file and a grad file
# share the layout); two names keep call sites self-documenting.
def read_flat(path: str, cfg: Config) -> dict[str, np.ndarray]:
    """Read a flat float32 file written by :func:`write_init` / the C dumper."""
    with open(path, "rb") as f:
        return unpack(f.read(), cfg)


def read_grads(path: str, cfg: Config) -> dict[str, np.ndarray]:
    """Read the ANE's dumped per-parameter gradients (same layout as init)."""
    return read_flat(path, cfg)


def write_init(path: str, params: dict[str, np.ndarray], cfg: Config) -> None:
    """Write the shared init to ``path`` for the ANE trainer's ``--init`` flag."""
    with open(path, "wb") as f:
        f.write(pack(params, cfg))
