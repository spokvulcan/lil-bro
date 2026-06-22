"""Unit tests for lilbro.chess: the run-config + the config -> train_selfplay argv
contract. Pure (no ANE, no subprocess) — the fast seam. Mirrors tests/test_ane_run.py."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from lilbro.chess import (
    ChessConfig,
    LADDER,
    config_from_dict,
    config_to_dict,
    load_config,
    save_config,
    selfplay_argv,
)
from lilbro.chess.config import MAX_BATCH

REPO = Path(__file__).resolve().parents[1]
SELFPLAY_C = REPO / "training" / "training_dynamic" / "chess" / "selfplay.c"


def _argv_pairs(argv: list[str]) -> dict[str, str]:
    """Collapse a flag list into {flag: value} (value '' for bare flags)."""
    out: dict[str, str] = {}
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--") and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            out[tok] = argv[i + 1]
            i += 2
        else:
            out[tok] = ""
            i += 1
    return out


def test_to_argv_pulls_from_config():
    cfg = ChessConfig(name="t", batch=48, sims=24, considered=12, iters=7, lr=1e-3,
                      eval_sims=24, bench_games=96)
    p = _argv_pairs(cfg.to_argv())
    assert p["--B"] == "48"
    assert p["--sims"] == "24"
    assert p["--considered"] == "12"
    assert p["--iters"] == "7"
    assert p["--bench-games"] == "96"
    assert float(p["--lr"]) == 1e-3
    assert p["--ckpt"] == "ane_chess_t.ckpt"   # __post_init__ default


def test_boolean_flags():
    # improved_policy default True -> no --visit-policy; curriculum default off
    base = ChessConfig(name="b").to_argv()
    assert "--visit-policy" not in base
    assert "--curriculum" not in base
    # flip them
    a = ChessConfig(name="b", improved_policy=False, curriculum=True, curriculum_plies=6).to_argv()
    assert "--visit-policy" in a
    assert "--curriculum" in a
    assert _argv_pairs(a)["--curriculum-plies"] == "6"


def test_selfplay_argv_modes():
    cfg = LADDER["g2_smoke"]
    g2 = selfplay_argv(cfg, mode="g2")
    assert g2[0].endswith("train_selfplay") and g2[1] == "--g2"
    assert selfplay_argv(cfg, mode="selfcheck")[1] == "--selfcheck"
    assert selfplay_argv(cfg, mode="bench")[1] == "--bench"
    smoke = selfplay_argv(cfg, mode="smoke")
    assert "--g2" not in smoke and "--selfcheck" not in smoke and "--bench" not in smoke
    assert "--resume" in selfplay_argv(cfg, mode="g2", resume=True)
    with pytest.raises(ValueError):
        selfplay_argv(cfg, mode="bogus")


def test_json_roundtrip(tmp_path):
    cfg = LADDER["g2"]
    path = tmp_path / "c.json"
    save_config(cfg, path)
    assert load_config(path) == cfg
    assert config_from_dict(config_to_dict(cfg)) == cfg


def test_unknown_field_rejected():
    d = config_to_dict(LADDER["g2"])
    d["bogus"] = 1
    with pytest.raises(ValueError):
        config_from_dict(d)


def test_validation():
    with pytest.raises(ValueError):
        ChessConfig(name="x", batch=MAX_BATCH + 1)        # past the 16384 ANE wall
    with pytest.raises(ValueError):
        ChessConfig(name="x", sims=4, considered=8)        # SH needs sims >= considered
    with pytest.raises(ValueError):
        ChessConfig(name="x", eval_considered=16, eval_sims=8)  # eval search degenerate
    with pytest.raises(ValueError):
        ChessConfig(name="x", iters=0)                     # must be positive
    with pytest.raises(ValueError):
        ChessConfig(name="x", bench_games=0)
    with pytest.raises(ValueError):
        ChessConfig(name="x", value_weight=0.0)        # blend weight must be > 0 (H3 sign check)


def test_td_lambda_knob():
    base = ChessConfig(name="b")
    assert base.td_lambda == 1.0
    p = _argv_pairs(base.to_argv())
    assert float(p["--td-lambda"]) == 1.0
    cfg = ChessConfig(name="t", td_lambda=0.5)
    assert float(_argv_pairs(cfg.to_argv())["--td-lambda"]) == 0.5
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            ChessConfig(name="x", td_lambda=bad)


def test_ladder_presets_valid():
    for key, cfg in LADDER.items():
        assert cfg.name == key
        assert 1 <= cfg.batch <= MAX_BATCH
        assert cfg.considered <= cfg.sims
        assert cfg.eval_considered <= cfg.eval_sims
        # argv must be parseable: every numeric flag value round-trips
        for flag, val in _argv_pairs(cfg.to_argv()).items():
            if val and flag not in ("--ckpt",):
                float(val)  # raises if not numeric


def test_argv_flags_recognized_by_c_parser():
    """Every flag to_argv()/selfplay_argv() emits must be accepted by the C sp_parse —
    guards against the Python<->C flag contract silently drifting."""
    src = SELFPLAY_C.read_text()
    c_flags = set(re.findall(r'"(--[A-Za-z0-9-]+)"', src))
    assert "--B" in c_flags, "sanity: failed to parse flags from selfplay.c"
    emitted = {f for f in _argv_pairs(selfplay_argv(
        ChessConfig(name="z", improved_policy=False, curriculum=True), mode="g2", resume=True))
        if f.startswith("--")}
    unknown = emitted - c_flags
    assert not unknown, f"flags emitted by Python but not parsed by selfplay.c: {sorted(unknown)}"
