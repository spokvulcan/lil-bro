"""ANE bridge serialization contract (the R1 plumbing's failure point).

The ANE trainer and the Python twins exchange weights/gradients as a flat
float32 binary. If the byte order, per-tensor shape, or dtype drifts between the
two sides, R1 compares garbage and the gate is meaningless. These tests pin the
contract on the Python side: the layout is exactly ``param_spec`` order, every
tensor is float32 row-major ``[out, in]``, and write→read is a perfect round
trip. (The matching C reader/writer lives in ``train.m``; this guards the schema
both sides agree to.)
"""

import struct

import numpy as np
import pytest

from lilbro.configs import Config
from lilbro.mlx_ref import init_params, param_shapes
from lilbro import ane_bridge

CONFIGS = {
    "base": Config(name="b", dim=64, n_layers=2, n_heads=4, head_dim=16,
                   seq=32, vocab=256, hidden=128, seed=0),
    "gqa": Config(name="g", dim=64, n_layers=3, n_heads=8, kv_heads=2,
                  head_dim=16, seq=24, vocab=200, hidden=192, seed=1),
}


@pytest.mark.parametrize("name", list(CONFIGS))
def test_flat_order_is_param_spec_dense(name):
    """The canonical flat order is exactly the dense ``param_spec`` order."""
    cfg = CONFIGS[name]
    assert ane_bridge.flat_order(cfg) == list(param_shapes(cfg))
    # Embedding leads, rms_final trails, 9 tensors per layer in between.
    order = ane_bridge.flat_order(cfg)
    assert order[0] == "embed"
    assert order[-1] == "rms_final"
    assert len(order) == 1 + 9 * cfg.n_layers + 1


def test_flat_order_rejects_mtp():
    """The ANE trainer has no MTP path, so the bridge must refuse MTP configs
    rather than silently emit a layout the C side can't read."""
    cfg = Config(name="m", dim=64, n_layers=1, n_heads=4, head_dim=16, seq=16,
                 vocab=128, hidden=128, mtp_depth=2)
    with pytest.raises(ValueError, match="MTP"):
        ane_bridge.flat_order(cfg)


@pytest.mark.parametrize("name", list(CONFIGS))
def test_init_round_trip(name):
    """write_init → read_flat reconstructs every parameter bit-for-bit (fp32)."""
    cfg = CONFIGS[name]
    params = init_params(cfg)
    blob = ane_bridge.pack(params, cfg)
    back = ane_bridge.unpack(blob, cfg)

    assert set(back) == set(param_shapes(cfg))
    for pname, shape in param_shapes(cfg).items():
        assert back[pname].shape == shape
        assert back[pname].dtype == np.float32
        # Exact match at fp32 (the C side reads the same float32 bytes).
        np.testing.assert_array_equal(
            back[pname], params[pname].astype(np.float32))


@pytest.mark.parametrize("name", list(CONFIGS))
def test_blob_size_is_exact(name):
    """No padding/headers: blob is exactly sum(prod(shape)) * 4 bytes, in order."""
    cfg = CONFIGS[name]
    params = init_params(cfg)
    blob = ane_bridge.pack(params, cfg)
    expected = 4 * sum(int(np.prod(s)) for s in param_shapes(cfg).values())
    assert len(blob) == expected


def test_pack_is_row_major_in_declared_order():
    """Bytes are concatenated float32, row-major, in flat_order — verify against
    a hand-rolled struct unpack so we're not just testing pack against unpack."""
    cfg = CONFIGS["base"]
    params = init_params(cfg)
    blob = ane_bridge.pack(params, cfg)

    offset = 0
    for pname in ane_bridge.flat_order(cfg):
        arr = params[pname].astype(np.float32).ravel(order="C")
        n = arr.size
        raw = struct.unpack(f"<{n}f", blob[offset * 4:(offset + n) * 4])
        np.testing.assert_array_equal(np.asarray(raw, dtype=np.float32), arr)
        offset += n
    assert offset * 4 == len(blob)


@pytest.mark.parametrize("name", list(CONFIGS))
def test_read_grads_from_file(tmp_path, name):
    """read_grads parses a dumped grad file (same layout as init) by name/shape."""
    cfg = CONFIGS[name]
    # A grad blob has the identical layout to an init blob (every param has a
    # grad of the same shape); fabricate one and confirm it parses back.
    fake = {n: (np.arange(int(np.prod(s)), dtype=np.float32).reshape(s) + 0.5)
            for n, s in param_shapes(cfg).items()}
    path = tmp_path / "grads.bin"
    path.write_bytes(ane_bridge.pack(fake, cfg))

    grads = ane_bridge.read_grads(str(path), cfg)
    assert set(grads) == set(param_shapes(cfg))
    for pname, shape in param_shapes(cfg).items():
        assert grads[pname].shape == shape
        np.testing.assert_array_equal(grads[pname], fake[pname])


def test_read_grads_rejects_wrong_size(tmp_path):
    """A truncated/oversized file is a layout mismatch — fail loudly, never read
    past the end or silently ignore trailing bytes."""
    cfg = CONFIGS["base"]
    path = tmp_path / "bad.bin"
    path.write_bytes(b"\x00" * 16)  # nowhere near the right size
    with pytest.raises(ValueError, match="size"):
        ane_bridge.read_grads(str(path), cfg)
