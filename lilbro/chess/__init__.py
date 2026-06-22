"""lilbro.chess — eval-side orchestration for the ANE chess self-play trainer (#18).

A run-config (``ChessConfig`` + the ``LADDER`` of presets) and a thin driver that builds
and runs the C/Obj-C ``train_selfplay`` binary. EVAL-SIDE ONLY: the hot self-play loop
(generation + replay + learner) is the ANE binary; nothing here is in that loop and there
is no ``python-chess`` dependency. The model SHAPE is the hand-written
``training/training_dynamic/models/chess_g0.h`` — chess does not emit a C header."""

from .config import (
    ChessConfig,
    LADDER,
    config_from_dict,
    config_to_dict,
    load_config,
    save_config,
)
from .run import build, run, selfplay_argv

__all__ = [
    "ChessConfig",
    "LADDER",
    "config_from_dict",
    "config_to_dict",
    "load_config",
    "save_config",
    "selfplay_argv",
    "build",
    "run",
]
