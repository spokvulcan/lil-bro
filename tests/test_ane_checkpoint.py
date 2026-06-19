"""ANE checkpoint reader — pins the v4 binary layout (train.m:save_checkpoint).

Round-trips weights through ``write_ckpt`` (a faithful mirror of the C writer)
and back, and asserts the reader rejects a checkpoint whose dims disagree with
the config. The C-writer ↔ Python-reader path itself is exercised by the real
ANE smoke run; here we pin the format and the weight/Adam stride.
"""

import struct

import numpy as np
import pytest

from lilbro.ane_bridge.checkpoint import (
    HEADER_BYTES,
    load_ckpt_params,
    read_header,
    write_ckpt,
)
from lilbro.configs import Config
from lilbro.mlx_ref import init_params, param_shapes


def _cfg(**kw):
    base = dict(name="ck", dim=32, n_layers=2, n_heads=4, head_dim=8, seq=16,
                vocab=64, hidden=64, seed=3)
    base.update(kw)
    return Config(**base)


def test_header_is_96_bytes():
    assert HEADER_BYTES == 96


def test_write_read_round_trip(tmp_path):
    cfg = _cfg()
    params = {k: v.astype(np.float32) for k, v in init_params(cfg).items()}
    p = write_ckpt(tmp_path / "ck.bin", params, cfg, step=7, loss=1.23, adam_t=5)

    h = read_header(p)
    assert h["step"] == 7 and h["adam_t"] == 5
    assert h["n_layers"] == cfg.n_layers and h["vocab_size"] == cfg.vocab
    assert h["dim"] == cfg.dim and h["q_dim"] == cfg.q_dim

    got = load_ckpt_params(p, cfg)
    # Only weights come back (no MTP, no adam), keyed by param_spec.
    assert set(got) == set(param_shapes(cfg))
    for name, arr in params.items():
        assert got[name].shape == param_shapes(cfg)[name]
        assert np.allclose(got[name], arr, atol=0), name


def test_gqa_round_trip(tmp_path):
    cfg = _cfg(name="ckg", n_heads=4, kv_heads=2)  # GQA: kv weights are smaller
    params = {k: v.astype(np.float32) for k, v in init_params(cfg).items()}
    p = write_ckpt(tmp_path / "ckg.bin", params, cfg)
    got = load_ckpt_params(p, cfg)
    for name in ("layer.0.wk", "layer.1.wv", "layer.0.wq", "embed", "rms_final"):
        assert np.allclose(got[name], params[name], atol=0), name


def test_dim_mismatch_rejected(tmp_path):
    cfg = _cfg()
    params = {k: v.astype(np.float32) for k, v in init_params(cfg).items()}
    p = write_ckpt(tmp_path / "ck.bin", params, cfg)
    wrong = _cfg(dim=64, head_dim=16)  # different shape -> must refuse
    with pytest.raises(ValueError, match="disagree with config"):
        load_ckpt_params(p, wrong)


def test_bad_magic_rejected(tmp_path):
    p = tmp_path / "bad.bin"
    p.write_bytes(struct.pack("<I", 0xDEADBEEF) + b"\x00" * (HEADER_BYTES))
    with pytest.raises(ValueError, match="bad magic"):
        read_header(p)
