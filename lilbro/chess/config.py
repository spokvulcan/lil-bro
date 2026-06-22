"""ChessConfig — the self-play RUN config for the ANE chess trainer (build-step 4,
ADR 0005, issue #18).

This is the eval-side twin of the C ``SPConfig`` in ``training/training_dynamic/
chess/selfplay.h``: one frozen dataclass that names every self-play knob and maps to
the exact ``train_selfplay`` argv. It is a *run* config, NOT a model config — the chess
net's SHAPE lives in the hand-written ``models/chess_g0.h`` (chess does not go through
``lilbro.configs.emit_c``; see ``lilbro/configs``), so there is no C-header emission
here, only argv generation. Mirrors ``lilbro/configs/schema.py``: frozen dataclass,
``-1``/sentinel-free explicit defaults, JSON save/load, a ``LADDER`` of named presets.

Zero deps; no python-chess (the hot loop is the C binary). Pure config + argv mapping,
unit-tested in ``tests/test_chess_config.py`` (no ANE needed)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

# train_selfplay packs B up to a multiple of 32, so round32(B)*96 must stay <= 16384.
MAX_BATCH = 160


@dataclass(frozen=True)
class ChessConfig:
    """One self-play run == one ChessConfig. Fields mirror the C SPConfig 1:1; the
    defaults match ``sp_defaults()`` so a bare ``ChessConfig(name=...)`` reproduces the
    binary's built-in defaults."""

    name: str

    # --- generation (self-play) ---
    batch: int = 64              # B parallel games per generation step (-> --B)
    sims: int = 32               # MCTS simulations per move (-> --sims)
    considered: int = 16         # Gumbel considered actions at the root (-> --considered)
    dir_alpha: float = 0.3       # Purist-Zero root Dirichlet alpha (-> --dir-alpha)
    dir_frac: float = 0.25       # root-noise mixing fraction (-> --dir-frac)
    temp: float = 1.0            # move-selection temperature (-> --temp)
    temp_moves: int = 15         # plies sampled at temperature before going greedy (-> --temp-moves)
    max_plies: int = 100         # per-game ply cap (-> --max-plies)
    improved_policy: bool = True  # Gumbel improved-policy target (False -> --visit-policy)
    curriculum: bool = False     # random-opening fallback, default OFF (-> --curriculum)
    curriculum_plies: int = 8    # random plies if curriculum on (-> --curriculum-plies)
    adjudicate: bool = False     # material-adjudicate capped TRAINING games (-> --adjudicate); a
                                 # cold-start mitigation (the value head needs a non-draw signal).
                                 # TRAINING-ONLY: the eval ladder always scores real game outcomes.
    td_lambda: float = 1.0       # TD(lambda) value-target knob (-> --td-lambda): 1.0 = terminal z
                                 # (legacy behavior), 0.0 = 1-step TD; densifies the value signal.

    # --- replay + learner ---
    replay_cap: int = 30000      # sliding-window size (-> --replay)
    learner_batch: int = 64      # minibatch K (-> --lbatch)
    learner_steps: int = 16      # optimizer steps per iteration (-> --lsteps)
    iters: int = 60              # self-play iterations (-> --iters)
    lr: float = 2e-3             # AdamW learning rate (-> --lr)
    loss_scale: float = 256.0    # fp16 loss-scaling (-> --loss-scale)
    grad_clip: float = 1.0       # global grad-norm clip (-> --clip)
    weight_decay: float = 0.0    # AdamW decoupled weight decay (-> --wd)
    value_weight: float = 1.0    # value-loss weight in the AZ loss (-> --vw)

    # --- eval ladder (vs random-mover, then 1-ply greedy) ---
    eval_games: int = 40         # games per opponent per eval (-> --eval-games)
    eval_every: int = 5          # eval cadence in iterations (-> --eval-every)
    eval_sims: int = 32          # MCTS sims during eval (-> --eval-sims)
    eval_considered: int = 16    # Gumbel considered at eval (-> --eval-considered); a TRAINED
                                 # policy guides a narrow/cheap eval search (decoupled from gen)
    eval_max_plies: int = 120    # per-eval-game ply cap (-> --eval-max-plies)

    # --- benchmark harness ---
    bench_games: int = 256        # fixed generated games for --bench (-> --bench-games)

    # --- bookkeeping ---
    seed: int = 42               # deterministic RNG seed (-> --seed)
    ckpt: str = ""               # checkpoint path (-> --ckpt; default -> ane_<name>.ckpt)

    # --- backend (GPU/MPS rewrite, Phase 2) ---
    use_mps_graph: bool = False  # --mps-graph: route the whole trunk forward through one MPSGraph
                                  # (5.3-6.0x vs ANE+CPU on M3 Max; eval-only — the learner stays on ANE)

    def __post_init__(self) -> None:
        if not self.ckpt:
            object.__setattr__(self, "ckpt", f"ane_chess_{self.name}.ckpt")
        self._validate()

    def _validate(self) -> None:
        if not 1 <= self.batch <= MAX_BATCH:
            raise ValueError(f"batch must be in [1,{MAX_BATCH}] (the 16384 ANE wall), got {self.batch}")
        for k in ("sims", "considered", "max_plies", "iters", "learner_batch", "learner_steps",
                  "replay_cap", "eval_games", "eval_sims", "eval_considered", "eval_max_plies",
                  "bench_games"):
            if getattr(self, k) <= 0:
                raise ValueError(f"{k} must be positive, got {getattr(self, k)}")
        # Sequential Halving needs sims >= considered for a non-degenerate schedule. The C
        # search clamps safely either way (mcts_seq_halving floors visits at 1), but a config
        # that asks for more considered actions than it can visit is a footgun — reject it.
        # Generation and eval have independent (considered, sims) budgets.
        if self.considered > self.sims:
            raise ValueError(f"considered ({self.considered}) > sims ({self.sims}): SH needs sims >= considered")
        if self.eval_considered > self.eval_sims:
            raise ValueError(f"eval_considered ({self.eval_considered}) > eval_sims ({self.eval_sims}): "
                             f"the eval search would be degenerate")
        if not 0.0 <= self.dir_frac <= 1.0:
            raise ValueError(f"dir_frac must be in [0,1], got {self.dir_frac}")
        if self.value_weight <= 0.0:
            raise ValueError(f"value_weight must be positive (0 silences the value head; "
                             f"the AZ policy/value blend), got {self.value_weight}")
        if not 0.0 <= self.td_lambda <= 1.0:
            raise ValueError(f"td_lambda must be in [0,1] (1.0=terminal z, 0.0=1-step TD), "
                             f"got {self.td_lambda}")

    def to_argv(self) -> list[str]:
        """Map this config to the ``train_selfplay`` flag list (NOT including the run mode
        ``--g2`` / ``--selfcheck`` or ``--resume`` — ``run.py`` adds those). Deterministic
        and pure: the unit-testable seam."""
        argv = [
            "--B", str(self.batch),
            "--sims", str(self.sims),
            "--considered", str(self.considered),
            "--dir-alpha", _fmt(self.dir_alpha),
            "--dir-frac", _fmt(self.dir_frac),
            "--temp", _fmt(self.temp),
            "--temp-moves", str(self.temp_moves),
            "--max-plies", str(self.max_plies),
            "--replay", str(self.replay_cap),
            "--lbatch", str(self.learner_batch),
            "--lsteps", str(self.learner_steps),
            "--iters", str(self.iters),
            "--lr", _fmt(self.lr),
            "--loss-scale", _fmt(self.loss_scale),
            "--clip", _fmt(self.grad_clip),
            "--wd", _fmt(self.weight_decay),
            "--vw", _fmt(self.value_weight),
            "--td-lambda", _fmt(self.td_lambda),
            "--eval-games", str(self.eval_games),
            "--eval-every", str(self.eval_every),
            "--eval-sims", str(self.eval_sims),
            "--eval-considered", str(self.eval_considered),
            "--eval-max-plies", str(self.eval_max_plies),
            "--bench-games", str(self.bench_games),
            "--seed", str(self.seed),
            "--ckpt", self.ckpt,
        ]
        if not self.improved_policy:
            argv.append("--visit-policy")
        if self.curriculum:
            argv += ["--curriculum", "--curriculum-plies", str(self.curriculum_plies)]
        if self.adjudicate:
            argv.append("--adjudicate")
        if self.use_mps_graph:
            argv.append("--mps-graph")
        return argv


def _fmt(x: float) -> str:
    """Locale-safe float formatting (mirrors ane_bridge/run.py)."""
    return repr(float(x))


# ---- JSON round-trip (mirrors lilbro/configs/schema.py) ---------------------------------
def config_to_dict(cfg: ChessConfig) -> dict[str, Any]:
    return asdict(cfg)


def config_from_dict(d: dict[str, Any]) -> ChessConfig:
    known = {f.name for f in fields(ChessConfig)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"unknown chess-config fields: {sorted(unknown)}")
    return ChessConfig(**d)


def save_config(cfg: ChessConfig, path: str | Path) -> None:
    Path(path).write_text(json.dumps(config_to_dict(cfg), indent=2, sort_keys=True) + "\n")


def load_config(path: str | Path) -> ChessConfig:
    return config_from_dict(json.loads(Path(path).read_text()))


# ---- named presets (the gate ladder) ---------------------------------------------------
# Sizes reflect the MEASURED ANE-dispatch-bound cost (~7 matmul dispatches/forward; the
# self-play loop is dispatch-bound, results/chess_throughput_probe.md + the #18 profile):
# the gate proves LEARNING (the win-rate curve climbs), not strength, so the budgets are
# deliberately low. `g2` is the feasible gate rung; `g2_full` is the stronger (slower) run.
LADDER: dict[str, ChessConfig] = {
    # Fast end-to-end shake-out (seconds-to-a-minute): does the loop run + eval at all.
    "g2_smoke": ChessConfig(
        name="g2_smoke", batch=16, sims=8, considered=8, max_plies=20, iters=4,
        learner_steps=4, eval_games=8, eval_sims=4, eval_considered=4, eval_max_plies=20, eval_every=2,
        bench_games=64,
    ),
    # The G2 gate rung: a feasible run whose vs-random curve must climb. The MEASURED
    # cold-start dynamics (see results/chess_g2.md) dictate the shape:
    #   * cold-start mitigations ON (curriculum + adjudicate) — a tiny net in pure self-play
    #     from startpos only ever draws (no value signal); these give one (ADR 0005's escape);
    #   * the wall budget is shifted onto the LEARNER: a measured diagnostic showed the loop
    #     learns (loss_val drops off the ln3 uniform floor) but ~100 updates barely moves it —
    #     the value/policy need THOUSANDS of updates. So iters*learner_steps ~= 2400, not ~400.
    #   * generation is cheap (moderate sims): B is throughput-NEUTRAL (the batched forward is
    #     bandwidth-bound past seq~2048, MEASURED B=64 vs B=128 = ~1:2 wall), so a small B just
    #     buys faster iters + more eval points on the curve, at no throughput cost;
    #   * eval search is cheap (a trained policy guides it); eval_max_plies is generous so a net
    #     that learned to win material can convert it to mate vs a random-mover.
    # The eval ladder stays pure (real game outcomes; a capped eval game is a draw).
    "g2": ChessConfig(
        name="g2", batch=64, sims=16, considered=16, max_plies=20, iters=30,
        learner_steps=60, learner_batch=96, replay_cap=80000, lr=5e-3, temp_moves=8,
        curriculum=True, curriculum_plies=8, adjudicate=True, value_weight=1.5,
        bench_games=320,
        # Eval is the loop's dominant cost (full games vs a non-resigning random-mover run to
        # the ply cap), so it is kept CHEAP and infrequent — the per-iter loss + gen-W/D/L row
        # is the primary learning signal; eval only confirms the win-rate effect at 6 points.
        eval_games=8, eval_sims=8, eval_considered=8, eval_max_plies=50, eval_every=6,
    ),
    # The G2 diagnosis rung (issue #24): the g2 shape with eval_games=200 to read the win-rate
    # curve BELOW the noise floor (±0.035 at 200 games vs ±0.08 at 40 — kills H4). Run with
    # --mps-graph (the GPU learner path); stderr carries the per-step grad-norm/NaN-count
    # (grads_diagnose) and stdout carries the split loss_pol/loss_val. The H1–H4 read is off
    # the instrumented curve (see results/chess_g2_diagnosis.md).
    "g2_diag": ChessConfig(
        name="g2_diag", batch=64, sims=16, considered=16, max_plies=20, iters=30,
        learner_steps=60, learner_batch=96, replay_cap=80000, lr=5e-3, temp_moves=8,
        curriculum=True, curriculum_plies=8, adjudicate=True, value_weight=1.5,
        bench_games=320,
        eval_games=200, eval_sims=8, eval_considered=8, eval_max_plies=50, eval_every=5,
    ),
    # Raw-throughput bulk-data phase. This still runs the real self-play/evaluator path, but
    # deliberately uses one-sim searches and short capped games with material adjudication.
    # Measured 2026-06-22: ~151k games/hour on M3 Max at --mode bench.
    "speed_bulk": ChessConfig(
        name="speed_bulk", batch=160, sims=1, considered=1, max_plies=4, iters=30,
        learner_steps=60, learner_batch=96, replay_cap=80000, lr=5e-3, temp_moves=4,
        curriculum=True, curriculum_plies=8, adjudicate=True, value_weight=1.5,
        bench_games=160,
        eval_games=8, eval_sims=1, eval_considered=1, eval_max_plies=20, eval_every=6,
    ),
    # Stronger, slower: more search + more iterations once the loop is proven to learn.
    "g2_full": ChessConfig(
        name="g2_full", batch=128, sims=32, considered=32, max_plies=80, iters=60,
        learner_steps=48, learner_batch=128, replay_cap=120000, lr=3e-3,
        curriculum=True, adjudicate=True,
        eval_games=40, eval_sims=16, eval_considered=16, eval_max_plies=120, eval_every=5,
        bench_games=320,
    ),
}
