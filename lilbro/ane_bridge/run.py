"""Runtime-config CLI: drive the ANE trainer from one shared ``Config``.

One config (a ladder name or a JSON file) -> emit the C header -> build (cached
per shape) -> run ``./train`` with flags derived from the config. This is the
ANE-side realization of User Story 1 / issue #47: a ladder rung or ablation cell
is fully described by the config, with **no hand-edited #defines**.

What is and isn't "runtime" here is dictated by the hardware, not a shortcut.
The ANE compiles a **fixed-shape** MIL graph (``reshape``/``concat``/``transpose``
fail at runtime — see issue #47 and ``config.h``), so the model *dimensions*
(dim, layers, heads, seq, vocab) are baked at compile time: changing a dimension
re-emits the header and recompiles (``make`` caches by mtime, so an unchanged
shape is a no-op). Everything that does **not** change a tensor shape —
optimizer, lr, weight decay, accumulation, warmup, clip, data shard, validation
cadence, resume/scratch — is a runtime ``./train`` flag. So an ablation that
varies *optimizer* or *lr* at a fixed shape never recompiles: exactly the sweep
the PRD asks for.

CLI:  ``python -m lilbro.ane_bridge.run r2_small --steps 20000 --val``
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from lilbro.configs import Config, LADDER, load_config, write_c_header

REPO = Path(__file__).resolve().parents[2]
TRAIN_DIR = REPO / "training" / "training_dynamic"
MODELS_DIR = TRAIN_DIR / "models"
TRAIN_BIN = TRAIN_DIR / "train"


def model_name(cfg: Config) -> str:
    """The ``make MODEL=`` target / generated-header stem for ``cfg``."""
    return f"gen_{cfg.name}"


def _fmt(x: float) -> str:
    """Format a float for ``atof`` on the C side (exact, no locale surprises)."""
    return repr(float(x))


def train_argv(
    cfg: Config,
    *,
    binary: str = "./train",
    scratch: bool = True,
    resume: bool = False,
    steps: int | None = None,
    accum: int | None = None,
    warmup: int = 100,
    clip: float = 1.0,
    lr: float | None = None,
    wd: float | None = None,
    opt: str | None = None,
    data: str | None = None,
    val_data: str | None = None,
    val_every: int = 0,
    val_batches: int = 0,
    ckpt: str | None = None,
    overfit: bool = False,
    init: str | None = None,
    dump_grads: str | None = None,
    dump_weights: str | None = None,
) -> list[str]:
    """Map a shared ``Config`` (+ run options) to the exact ``./train`` argv.

    Pure and deterministic — the unit-testable seam of the CLI. Defaults pull
    from the config (lr, weight_decay, optimizer, data shard, accum=batch_size);
    explicit kwargs override so an ablation can vary one knob at a fixed shape.
    Validation flags are emitted only when ``val_every > 0`` and a val shard is
    available.
    """
    argv = [binary]
    if resume:
        argv.append("--resume")
    if scratch and not resume:
        argv.append("--scratch")
    if overfit:
        argv.append("--overfit")

    argv += ["--data", data if data is not None else cfg.data_path]
    if steps is not None:
        argv += ["--steps", str(steps)]
    argv += ["--lr", _fmt(cfg.lr if lr is None else lr)]
    argv += ["--accum", str(accum if accum is not None else cfg.batch_size)]
    argv += ["--wd", _fmt(cfg.weight_decay if wd is None else wd)]
    argv += ["--warmup", str(warmup)]
    argv += ["--clip", _fmt(clip)]
    argv += ["--opt", opt if opt is not None else cfg.optimizer]

    if ckpt is not None:
        argv += ["--ckpt", ckpt]

    # Validation: a fixed deterministic batch set on the held-out shard.
    vdata = val_data if val_data is not None else cfg.val_data_path
    if val_every > 0 and vdata:
        argv += ["--val-data", vdata,
                 "--val-every", str(val_every),
                 "--val-batches", str(val_batches if val_batches > 0 else 20)]

    # R1/step-diff bridge hooks (shared init in, grads/weights out).
    if init is not None:
        argv += ["--init", init]
    if dump_grads is not None:
        argv += ["--dump-grads", dump_grads]
    if dump_weights is not None:
        argv += ["--dump-weights", dump_weights]
    return argv


def build(cfg: Config, *, quiet: bool = True) -> Path:
    """Emit ``cfg``'s C header and build the trainer for it. Returns the binary.

    Re-emitting the header bumps its mtime, so ``make`` rebuilds whenever the
    shape changed and is a near-instant no-op when it didn't.
    """
    write_c_header(cfg, MODELS_DIR)
    proc = subprocess.run(
        ["make", f"MODEL={model_name(cfg)}"], cwd=TRAIN_DIR,
        capture_output=quiet, text=True)
    if proc.returncode != 0:
        out = (proc.stdout or "") + (proc.stderr or "") if quiet else ""
        raise RuntimeError(f"build failed for {cfg.name}:\n{out}")
    return TRAIN_BIN


def run(cfg: Config, *, stream: bool = True, check: bool = True,
        **argv_kwargs) -> subprocess.CompletedProcess:
    """Build for ``cfg`` then run the trainer. ``argv_kwargs`` -> ``train_argv``."""
    build(cfg)
    argv = train_argv(cfg, **argv_kwargs)
    if stream:
        return subprocess.run(argv, cwd=TRAIN_DIR, check=check)
    return subprocess.run(argv, cwd=TRAIN_DIR, check=check,
                          capture_output=True, text=True)


def _resolve(name_or_path: str) -> Config:
    if name_or_path in LADDER:
        return LADDER[name_or_path]
    return load_config(name_or_path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m lilbro.ane_bridge.run",
        description="Build + run the ANE trainer from a shared config.")
    ap.add_argument("config", help="ladder name (e.g. r2_small) or path to a config JSON")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--accum", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=None)
    ap.add_argument("--opt", choices=("adamw", "muon"), default=None,
                    help="override cfg.optimizer at a fixed shape (no recompile)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--val", action="store_true", help="enable periodic val loss on data01")
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--val-batches", type=int, default=20)
    ap.add_argument("--build-only", action="store_true")
    args = ap.parse_args(argv)

    cfg = _resolve(args.config)
    print(f"[run] {cfg.name}: dim={cfg.dim} layers={cfg.n_layers} "
          f"heads={cfg.n_heads}/{cfg.kv_heads} seq={cfg.seq} vocab={cfg.vocab} "
          f"opt={args.opt or cfg.optimizer}")
    build(cfg)
    if args.build_only:
        return 0
    proc = run(
        cfg, stream=True, check=False,
        scratch=not args.resume, resume=args.resume, overfit=args.overfit,
        steps=args.steps, accum=args.accum, warmup=args.warmup, clip=args.clip,
        lr=args.lr, wd=args.wd, opt=args.opt,
        val_every=args.val_every if args.val else 0, val_batches=args.val_batches)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
