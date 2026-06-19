"""Generate sample stories from an ANE checkpoint, across temperatures.

Wires the compact-vocab sampling mask end to end. The ANE trains with vocab
compaction (only tokens present in the train shard get LM-head gradients — see
``lilbro.eval.sample``), so this driver derives ``allowed_ids`` from that shard
(``TokenStream.active_vocab``) and restricts sampling to it. Prompt and output
use the 32K SentencePiece tokenizer the model trained on, so the stories read
back as text.

    python -m lilbro.eval.generate --config r2_small \\
        --ckpt training/training_dynamic/ane_r2_small_ckpt.bin \\
        --train-data training/tinystories_data00.bin \\
        --prompt "Once upon a time" --n-new 150 --temps 0 0.5 0.8 1.0

``--no-mask`` samples over the full 32K vocab instead — useful for *seeing* the
untrained-tail derail that the mask exists to prevent.
"""

from __future__ import annotations

import argparse

from lilbro.ane_bridge.checkpoint import generate_from_ckpt
from lilbro.configs.schema import LADDER
from lilbro.eval.data import TokenStream
from lilbro.eval.spm_bin import SpmBinTokenizer


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Sample stories from an ANE checkpoint across temperatures.")
    ap.add_argument("--config", default="r2_small", choices=sorted(LADDER),
                    help="ladder config the checkpoint was trained with")
    ap.add_argument("--ckpt", required=True, help="v4 checkpoint from train.m")
    ap.add_argument("--train-data", default="training/tinystories_data00.bin",
                    help="shard whose active vocab masks sampling (the compact vocab)")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--n-new", type=int, default=150)
    ap.add_argument("--temps", type=float, nargs="+", default=[0.0, 0.5, 0.8, 1.0],
                    help="temperatures to sample at (0 = greedy/argmax)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-mask", action="store_true",
                    help="sample over the full 32K vocab (shows the untrained-tail derail)")
    args = ap.parse_args(argv)

    cfg = LADDER[args.config]
    tok = SpmBinTokenizer()
    prompt_ids = tok.encode(args.prompt, bos=True)

    allowed = None
    if not args.no_mask:
        allowed = TokenStream(args.train_data).active_vocab()
        print(f"[mask] {len(allowed)} active ids / {cfg.vocab} vocab "
              f"(compact vocab of {args.train_data})")
    else:
        print(f"[mask] OFF — sampling full {cfg.vocab} vocab")
    print(f"[gen]  config={cfg.name} ckpt={args.ckpt} prompt={args.prompt!r} "
          f"n_new={args.n_new} seed={args.seed}\n")

    for t in args.temps:
        ids = generate_from_ckpt(args.ckpt, cfg, prompt_ids, args.n_new,
                                 temperature=t, seed=args.seed, allowed_ids=allowed)
        label = "greedy" if t <= 0.0 else f"temp={t:g}"
        print(f"───── {label} " + "─" * 44)
        print(tok.decode([int(i) for i in ids]))
        print()


if __name__ == "__main__":
    main()
