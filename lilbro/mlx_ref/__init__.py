"""The MLX twin + torch oracle, plus the shared optimizer and training loop.

  - ``model``       : MLX twin (fp32, Metal GPU) — correctness oracle + GPU baseline.
  - ``twin_torch``  : torch fp64 reference — ground-truth grads / tie-breaker.
  - ``optim``       : shared CPU optimizers (AdamW, Muon).
  - ``params``      : single shared numpy init both backends load.
  - ``train``       : framework-agnostic training loop.

``model`` is imported lazily so that ``params``/``optim``/``twin_torch`` are
usable without MLX present.
"""

from .params import MTP_LAMBDA, init_params, is_muon_param, param_shapes, param_spec
from .optim import AdamW, Muon, make_optimizer, optimizer_step
from .train import train_steps

__all__ = [
    "MTP_LAMBDA",
    "init_params",
    "is_muon_param",
    "param_shapes",
    "param_spec",
    "AdamW",
    "Muon",
    "make_optimizer",
    "optimizer_step",
    "train_steps",
]
