"""ANE ↔ Python bridge for the R1 correctness gate.

The ANE trainer (``training/training_dynamic/train.m``) and the Python twins
exchange model state as a flat, header-less float32 binary so that R1 can:

  1. seed the ANE trainer with the *same* shared init the twins load
     (``write_init``), and
  2. read back the ANE's per-parameter gradients for one fixed batch
     (``read_grads``) to diff against the torch fp64 oracle.

The on-disk layout is the single contract both sides obey: tensors concatenated
in ``param_spec`` order — ``embed`` first, then the 9 tensors of each transformer
block, then ``rms_final`` — each row-major (``[out, in]``) float32. No MTP: the
ANE trainer has no MTP path, so MTP configs are rejected rather than mis-encoded.

The matching C reader/writer is ``load_init_weights`` / ``dump_grads`` in
``train.m``; ``tests/test_ane_bridge.py`` pins the Python side of the contract.
"""

from .serialize import flat_order, pack, unpack, write_init, read_grads, read_flat
from .compare import tensor_metrics, grad_metrics, summarize, gate
from .checkpoint import (
    load_ckpt_params,
    read_header,
    write_ckpt,
    generate_from_ckpt,
)

__all__ = [
    "flat_order",
    "pack",
    "unpack",
    "write_init",
    "read_flat",
    "read_grads",
    "tensor_metrics",
    "grad_metrics",
    "summarize",
    "gate",
    "load_ckpt_params",
    "read_header",
    "write_ckpt",
    "generate_from_ckpt",
]
