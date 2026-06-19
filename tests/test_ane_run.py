"""Runtime-config CLI — the pure config->argv mapping (User Story 1 / issue #47).

The build/run halves touch hardware (a Makefile + the ANE), so they are covered
by the R2 launch and the existing R1 gate, not here. This pins the deterministic
seam: one shared ``Config`` plus run options -> the exact ``./train`` argv, so an
ablation cell is fully described by the config (no hand-edited #defines, no
recompile to change optimizer/lr/accum/data/val-cadence at a fixed shape).
"""

import pytest

from lilbro.ane_bridge.run import train_argv, model_name
from lilbro.configs import LADDER, Config

BOOL_FLAGS = {"--scratch", "--resume", "--overfit"}


def _argv_to_dict(argv: list[str]) -> dict:
    """Parse the trainer argv into {flag: value or True}; argv[0] is the binary."""
    assert argv, "empty argv"
    out: dict = {"_bin": argv[0]}
    i = 1
    while i < len(argv):
        tok = argv[i]
        assert tok.startswith("--"), f"expected a flag at {i}, got {tok!r}"
        if tok in BOOL_FLAGS:
            out[tok] = True
            i += 1
        else:
            out[tok] = argv[i + 1]
            i += 2
    return out


def test_model_name_maps_to_generated_header():
    assert model_name(LADDER["r2_small"]) == "gen_r2_small"


def test_defaults_from_config():
    cfg = LADDER["r2_small"]
    d = _argv_to_dict(train_argv(cfg, steps=1000))
    assert d["_bin"] == "./train"
    assert d.get("--scratch") is True
    assert "--resume" not in d
    assert d["--data"] == cfg.data_path
    assert d["--steps"] == "1000"
    assert float(d["--lr"]) == pytest.approx(cfg.lr)
    # accum defaults to batch_size = batch_tokens // seq (one seq per micro-step).
    assert int(d["--accum"]) == cfg.batch_size
    assert d["--opt"] == "adamw"
    # No val flags unless explicitly requested.
    assert "--val-data" not in d


def test_muon_optimizer_flag():
    cfg = Config(name="m", dim=64, n_layers=2, n_heads=4, head_dim=16, seq=32,
                 vocab=256, hidden=128, optimizer="muon")
    d = _argv_to_dict(train_argv(cfg))
    assert d["--opt"] == "muon"


def test_optimizer_override_beats_config():
    """An ablation varies the optimizer at a fixed shape without recompiling."""
    cfg = LADDER["r2_small"]  # optimizer="adamw"
    d = _argv_to_dict(train_argv(cfg, opt="muon"))
    assert d["--opt"] == "muon"


def test_resume_excludes_scratch():
    d = _argv_to_dict(train_argv(LADDER["r2_small"], scratch=False, resume=True))
    assert d.get("--resume") is True
    assert "--scratch" not in d


def test_validation_flags_present_when_requested():
    cfg = LADDER["r2_small"]
    d = _argv_to_dict(train_argv(cfg, steps=2000, val_every=200, val_batches=20))
    assert d["--val-data"] == cfg.val_data_path
    assert d["--val-every"] == "200"
    assert d["--val-batches"] == "20"


def test_accum_override_respected():
    d = _argv_to_dict(train_argv(LADDER["r2_small"], accum=4))
    assert int(d["--accum"]) == 4


def test_passthrough_init_and_dumps():
    cfg = LADDER["r0_overfit"]
    d = _argv_to_dict(train_argv(cfg, overfit=True, init="i.bin",
                                 dump_grads="g.bin", dump_weights="w.bin",
                                 clip=0.0, warmup=1, steps=1))
    assert d.get("--overfit") is True
    assert d["--init"] == "i.bin"
    assert d["--dump-grads"] == "g.bin"
    assert d["--dump-weights"] == "w.bin"
    assert float(d["--clip"]) == 0.0


def test_wd_and_warmup_from_config_and_args():
    cfg = Config(name="w", dim=64, n_layers=2, n_heads=4, head_dim=16, seq=32,
                 vocab=256, hidden=128, weight_decay=0.123)
    d = _argv_to_dict(train_argv(cfg, warmup=50))
    assert float(d["--wd"]) == pytest.approx(0.123)
    assert d["--warmup"] == "50"
