"""Tests for the eval harness: tokenizer, memmap loader, fixed val batches,
val loss, and the generation sampler."""

import numpy as np
import pytest

from lilbro.configs import Config
from lilbro.eval import (
    TokenStream,
    decode_bytes,
    encode_bytes,
    val_loss,
    write_token_stream,
)
from lilbro.mlx_ref import init_params
from lilbro.mlx_ref import model as mlx_twin


def test_byte_tokenizer_round_trip():
    s = "Once upon a time, a little fox ran. 🦊"
    ids = encode_bytes(s)
    assert ids.dtype == np.uint16
    assert ids.max() < 256
    assert decode_bytes(ids) == s


def test_token_stream_shift(tmp_path):
    toks = np.arange(100, dtype=np.uint16)
    p = write_token_stream(tmp_path / "data00.bin", toks)
    ts = TokenStream(p)
    assert len(ts) == 100
    x, y = ts.batch_at([0, 10], seq=8)
    assert x.shape == (2, 8) and y.shape == (2, 8)
    # y is x shifted by one token (next-token target).
    assert np.array_equal(x[0], np.arange(0, 8))
    assert np.array_equal(y[0], np.arange(1, 9))
    assert np.array_equal(y[1], np.arange(11, 19))


def test_fixed_val_batches_deterministic(tmp_path):
    toks = (np.arange(5000) % 256).astype(np.uint16)
    ts = TokenStream(write_token_stream(tmp_path / "data01.bin", toks))
    a = ts.fixed_val_batches(n_batches=3, batch_size=2, seq=16, seed=7)
    b = ts.fixed_val_batches(n_batches=3, batch_size=2, seq=16, seed=7)
    assert len(a) == 3
    for (xa, ya), (xb, yb) in zip(a, b):
        assert np.array_equal(xa, xb) and np.array_equal(ya, yb)
    # Different seed -> different positions.
    c = ts.fixed_val_batches(n_batches=3, batch_size=2, seq=16, seed=8)
    assert not np.array_equal(a[0][0], c[0][0])


def test_val_loss_matches_manual(tmp_path):
    cfg = Config(name="ev", dim=32, n_layers=1, n_heads=2, head_dim=16,
                 seq=16, vocab=256, hidden=64, seed=0)
    toks = (np.arange(3000) % 256).astype(np.uint16)
    ts = TokenStream(write_token_stream(tmp_path / "data01.bin", toks))
    batches = ts.fixed_val_batches(n_batches=2, batch_size=2, seq=cfg.seq, seed=1)
    params = init_params(cfg)
    vl = val_loss(mlx_twin.loss_only, params, batches, cfg)
    manual = np.mean([mlx_twin.loss_only(params, x, y, cfg) for x, y in batches])
    assert vl == pytest.approx(manual, rel=1e-6)
    # Untrained model on 256-way vocab -> loss near ln(256) ~ 5.55.
    assert 3.0 < vl < 7.0


@pytest.mark.slow
def test_sampler_runs_and_respects_context():
    from lilbro.eval.sample import generate
    cfg = Config(name="g", dim=32, n_layers=1, n_heads=2, head_dim=16,
                 seq=16, vocab=256, hidden=64, seed=0)
    params = init_params(cfg)
    prompt = encode_bytes("hello")
    out = generate(params, prompt, cfg, n_new=20, temperature=0.0)
    assert out.shape[0] == len(prompt) + 20
    assert np.array_equal(out[:len(prompt)], prompt.astype(np.int64))
    assert out.max() < cfg.vocab
    # Decodes without raising.
    assert isinstance(decode_bytes(out), str)
