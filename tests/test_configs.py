"""Tests for the shared config layer — the keystone interface.

Asserts on external behavior: the stored<->loaded round-trip, the derived
dimensions, validation, and that the *two consumers* (Python object and the
emitted C header) see identical model dimensions.
"""

import re

import pytest

from lilbro.configs import (
    Config,
    LADDER,
    config_from_dict,
    config_to_dict,
    emit_c_header,
    load_config,
    save_config,
    write_c_header,
)


def test_defaults_resolve():
    c = Config(name="t", dim=64, n_layers=2, n_heads=4, seq=32, vocab=256)
    assert c.kv_heads == 4              # default -> MHA
    assert c.head_dim == 16             # 64 // 4
    assert c.hidden == ((8 * 64 // 3 + 63) // 64) * 64  # 8/3*dim rounded to 64
    assert c.gqa_ratio == 1
    assert c.q_dim == 64 and c.kv_dim == 64
    assert c.ckpt_path == "ane_t_ckpt.bin"


def test_v4_knobs_default_off():
    """Every DeepSeek-V4 ablation knob (PRD #2) defaults to off / identity, so a
    plain config is bit-for-bit the original transformer."""
    c = Config(name="t", dim=64, n_layers=2, n_heads=4, seq=32, vocab=256)
    assert c.qk_norm is False
    assert c.attn_sink is False
    assert c.swiglu_clamp is False
    assert c.rope_rotary_dims == 64
    assert c.n_hc == 1
    assert c.mtp_depth == 0
    # head_dim (16) <= rope_rotary_dims (64) -> full RoPE (identity).
    assert c.rope_rotary_eff == c.head_dim == 16


def test_rope_rotary_eff_partial():
    """rope_rotary_eff = min(head_dim, rope_rotary_dims): partial only when the
    head is wider than the rotary budget (e.g. a 128-dim head)."""
    c = Config(name="big", dim=2048, n_layers=2, n_heads=16, head_dim=128,
               seq=64, vocab=256, rope_rotary_dims=64)
    assert c.rope_rotary_eff == 64          # partial: only the last 64 of 128 rotate
    full = Config(name="sm", dim=128, n_layers=2, n_heads=4, head_dim=32,
                  seq=64, vocab=256, rope_rotary_dims=64)
    assert full.rope_rotary_eff == 32       # identity: whole head rotates


def test_n_hc_zero_normalizes_to_one():
    """The PRD spells mHC-off as n_hc in {0, 1}; both mean a single stream."""
    c = Config(name="t", dim=64, n_layers=2, n_heads=4, seq=32, vocab=256, n_hc=0)
    assert c.n_hc == 1


def test_v4_knobs_round_trip():
    """A config with every knob flipped on survives dict + JSON round-trips."""
    c = Config(name="abl", dim=256, n_layers=6, n_heads=8, head_dim=32, seq=256,
               vocab=32000, hidden=768, qk_norm=True, attn_sink=True,
               swiglu_clamp=True, rope_rotary_dims=64, n_hc=4, mtp_depth=1)
    assert config_from_dict(config_to_dict(c)) == c


def test_derived_gqa():
    c = Config(name="g", dim=1024, n_layers=4, n_heads=16, kv_heads=8,
               head_dim=128, seq=256, vocab=1000)
    assert c.gqa_ratio == 2
    assert c.q_dim == 2048
    assert c.kv_dim == 1024
    assert c.res_alpha == pytest.approx(1.0 / (2 * 4) ** 0.5)


def test_batch_size_from_tokens():
    c = Config(name="b", dim=64, n_layers=1, n_heads=4, seq=128, vocab=256,
               batch_tokens=4096)
    assert c.batch_size == 32


def test_dict_round_trip():
    c = LADDER["r2_small"]
    assert config_from_dict(config_to_dict(c)) == c


def test_json_round_trip(tmp_path):
    c = LADDER["r3_110m"]
    p = tmp_path / "cfg.json"
    save_config(c, p)
    assert load_config(p) == c


def test_unknown_field_rejected():
    d = config_to_dict(LADDER["r0_overfit"])
    d["bogus"] = 1
    with pytest.raises(ValueError, match="unknown config fields"):
        config_from_dict(d)


@pytest.mark.parametrize("bad", [
    dict(optimizer="sgd"),
    dict(n_heads=6, kv_heads=4),   # 6 % 4 != 0
    dict(head_dim=15),             # odd -> RoPE invalid
    dict(mtp_depth=-1),
    dict(rope_rotary_dims=0),      # must be > 0
    dict(rope_rotary_dims=33),     # must be even (RoPE rotates pairs)
    dict(n_hc=-1),                 # must be >= 0
])
def test_validation_rejects(bad):
    base = dict(name="x", dim=64, n_layers=1, n_heads=4, seq=32, vocab=256)
    base.update(bad)
    with pytest.raises(ValueError):
        Config(**base)


@pytest.mark.parametrize("name", list(LADDER))
def test_ladder_presets_valid(name):
    c = LADDER[name]
    assert c.name == name
    # Every preset must round-trip and emit cleanly.
    assert config_from_dict(config_to_dict(c)) == c
    assert emit_c_header(c)


def _parse_defines(header: str) -> dict[str, str]:
    out = {}
    for m in re.finditer(r"^#define\s+(\w+)\s+(.+?)\s*(?://.*)?$", header, re.M):
        out[m.group(1)] = m.group(2).strip()
    return out


def test_two_consumers_identical_dims():
    """The C header (ANE consumer) and the Config (MLX consumer) agree exactly."""
    c = LADDER["r2_small"]
    d = _parse_defines(emit_c_header(c))
    assert int(d["DIM"]) == c.dim
    assert int(d["HIDDEN"]) == c.hidden
    assert int(d["HEADS"]) == c.n_heads
    assert int(d["KV_HEADS"]) == c.kv_heads
    assert int(d["HD"]) == c.head_dim
    assert int(d["SEQ"]) == c.seq
    assert int(d["NLAYERS"]) == c.n_layers
    assert int(d["VOCAB"]) == c.vocab
    assert d["MODEL_NAME"] == f'"{c.name}"'
    assert int(d["MTP_DEPTH"]) == c.mtp_depth


def test_v4_knobs_emitted_to_header():
    """The ANE consumer (C header) carries every V4 knob, off->0 / on->1, so a
    knob is toggled by emitting a new header — never by hand-editing #defines."""
    on = Config(name="abl", dim=256, n_layers=6, n_heads=8, head_dim=32, seq=256,
                vocab=32000, hidden=768, qk_norm=True, attn_sink=False,
                swiglu_clamp=True, rope_rotary_dims=64, n_hc=4)
    d = _parse_defines(emit_c_header(on))
    assert int(d["QK_NORM"]) == 1
    assert int(d["ATTN_SINK"]) == 0
    assert int(d["SWIGLU_CLAMP"]) == 1
    assert int(d["ROPE_ROTARY_DIMS"]) == 64
    assert int(d["N_HC"]) == 4
    # default config -> all off / identity
    off = _parse_defines(emit_c_header(LADDER["r0_overfit"]))
    assert int(off["QK_NORM"]) == 0 and int(off["ATTN_SINK"]) == 0
    assert int(off["SWIGLU_CLAMP"]) == 0 and int(off["N_HC"]) == 1


def test_emit_is_deterministic():
    c = LADDER["r0_overfit"]
    assert emit_c_header(c) == emit_c_header(c)


def test_write_c_header(tmp_path):
    c = LADDER["r0_overfit"]
    p = write_c_header(c, tmp_path)
    assert p.name == "gen_r0_overfit.h"
    assert "#define DIM 64" in p.read_text()


def test_gqa_header_matches_qwen_shape():
    """A qwen-like GQA config emits the same derived Q_DIM/KV_DIM relationship."""
    c = Config(name="qwen_like", dim=1024, n_layers=28, n_heads=16, kv_heads=8,
               head_dim=128, seq=256, vocab=151936, hidden=3072)
    d = _parse_defines(emit_c_header(c))
    # Q_DIM/KV_DIM are macro expressions; check the inputs that define them.
    assert int(d["HEADS"]) == 16 and int(d["KV_HEADS"]) == 8 and int(d["HD"]) == 128
    assert c.q_dim == 2048 and c.kv_dim == 1024
