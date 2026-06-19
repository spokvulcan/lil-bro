# lilbro/eval

Validation/eval harness — the infrastructure upstream lacks (it measures train
loss only).

- **Split:** `data00.bin` = train, `data01.bin` = val (separate TinyStories shards
  → clean split, no train/val window leakage).
- **Val loss:** periodic eval on a fixed val batch set; this is the signal behind
  the headline **tokens-to-target** metric.
- **Target:** defined at the knee of the dense+AdamW baseline's val curve @ rung 2.
- **Generation sampler:** qualitative check (coherent TinyStories text).
