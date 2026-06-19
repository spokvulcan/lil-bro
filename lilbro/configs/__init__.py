"""Shared parametric config — the single most important interface in lil-bro.

One schema (`Config`), two consumers:
  - the ANE trainer (Objective-C) via a generated C header  (`emit_c`)
  - the MLX twin (Python) by importing the `Config` directly

Both consumers read identical dimensions, so the R1 correctness diff and the
energy comparison are guaranteed to compare the *same* model.
"""

from .schema import (
    Config,
    LADDER,
    config_from_dict,
    config_to_dict,
    load_config,
    save_config,
)
from .emit_c import emit_c_header, write_c_header

__all__ = [
    "Config",
    "LADDER",
    "config_from_dict",
    "config_to_dict",
    "load_config",
    "save_config",
    "emit_c_header",
    "write_c_header",
]
