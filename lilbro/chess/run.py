"""Runtime driver for the ANE chess self-play trainer (build-step 4, ADR 0005, #18).

One ``ChessConfig`` (a ladder name or a JSON file) -> build ``train_selfplay`` (cached by
make's mtime) -> run it with flags derived from the config. The eval-side twin of
``lilbro/ane_bridge/run.py``, but simpler: the chess net's SHAPE is the hand-written
``models/chess_g0.h`` (chess does not emit a C header — see ``lilbro/configs``), so there
is no header step; every self-play knob is a runtime ``train_selfplay`` flag.

``selfplay_argv`` is the pure, unit-testable seam (config -> argv). ``build`` / ``run``
shell out. No python-chess; the hot loop is the C binary.

CLI:  ``python -m lilbro.chess.run g2 --mode g2``
      ``python -m lilbro.chess.run g2 --mode bench``
      ``python -m lilbro.chess.run g2 --dry-run``        # print argv, don't run
      ``python -m lilbro.chess.run my_cfg.json --resume``
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from lilbro.chess.config import ChessConfig, LADDER, load_config, save_config

REPO = Path(__file__).resolve().parents[2]
TRAIN_DIR = REPO / "training" / "training_dynamic"
TRAIN_BIN = TRAIN_DIR / "train_selfplay"

_MODE_FLAG = {"g2": "--g2", "selfcheck": "--selfcheck", "bench": "--bench", "elo": "--elo", "smoke": None}


def selfplay_argv(cfg: ChessConfig, *, binary: str = "./train_selfplay",
                  mode: str = "g2", resume: bool = False) -> list[str]:
    """Map a ``ChessConfig`` + run mode to the exact ``train_selfplay`` argv. Pure and
    deterministic — the unit-testable seam. ``mode`` is one of ``g2`` (the gate),
    ``selfcheck`` (batched-forward verification), ``bench`` (generation throughput),
    ``elo`` (self-anchored Elo round-robin over <ckpt>.eloNNN snapshots), or
    ``smoke`` (a short default run)."""
    if mode not in _MODE_FLAG:
        raise ValueError(f"mode must be one of {sorted(_MODE_FLAG)}, got {mode!r}")
    argv = [binary]
    flag = _MODE_FLAG[mode]
    if flag:
        argv.append(flag)
    if resume:
        argv.append("--resume")
    argv += cfg.to_argv()
    return argv


def build(*, quiet: bool = True) -> Path:
    """Build ``train_selfplay`` (make caches by mtime; an unchanged tree is a no-op)."""
    proc = subprocess.run(["make", "train_selfplay"], cwd=TRAIN_DIR,
                          capture_output=quiet, text=True)
    if proc.returncode != 0:
        out = (proc.stdout or "") + (proc.stderr or "") if quiet else ""
        raise RuntimeError(f"build failed:\n{out}")
    return TRAIN_BIN


def run(cfg: ChessConfig, *, mode: str = "g2", do_build: bool = True, resume: bool = False,
        stream: bool = True, check: bool = True) -> subprocess.CompletedProcess:
    """Build (optional) then run the self-play trainer for ``cfg``. With ``stream`` the
    child's stdout goes to the terminal live (the win-rate curve); otherwise it is
    captured on the returned ``CompletedProcess``."""
    if do_build:
        build()
    argv = selfplay_argv(cfg, mode=mode, resume=resume)
    if stream:
        return subprocess.run(argv, cwd=TRAIN_DIR, check=check)
    return subprocess.run(argv, cwd=TRAIN_DIR, check=check, capture_output=True, text=True)


def _resolve(name_or_path: str) -> ChessConfig:
    if name_or_path in LADDER:
        return LADDER[name_or_path]
    return load_config(name_or_path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Drive the ANE chess self-play trainer (G2).")
    ap.add_argument("config", help="ladder name (g2_smoke / g2 / g2_full) or path to a config JSON")
    ap.add_argument("--mode", choices=sorted(_MODE_FLAG), default="g2",
                    help="g2 = measured gate; selfcheck = batched-forward check; bench = throughput; smoke = short run")
    ap.add_argument("--resume", action="store_true", help="resume weights from the config's checkpoint")
    ap.add_argument("--no-build", action="store_true", help="skip the make step (use the existing binary)")
    ap.add_argument("--dry-run", action="store_true", help="print the resolved argv and exit (no build/run)")
    ap.add_argument("--save", metavar="PATH", help="write the resolved config JSON to PATH and exit")
    args = ap.parse_args(argv)

    cfg = _resolve(args.config)
    if args.save:
        save_config(cfg, args.save)
        print(f"wrote {args.save}")
        return 0
    if args.dry_run:
        print(" ".join(selfplay_argv(cfg, mode=args.mode, resume=args.resume)))
        return 0
    proc = run(cfg, mode=args.mode, do_build=not args.no_build, resume=args.resume, check=False)
    return proc.returncode


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
