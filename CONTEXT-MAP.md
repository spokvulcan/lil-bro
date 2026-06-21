# Context Map

lil-bro now spans two bounded contexts sharing one ANE-trainer substrate.

## Contexts

- [LM / V4-ablation](./CONTEXT.md) — the original thesis: a small **dense** LM trained from
  scratch on the ANE, used to ablate DeepSeek-V4 architecture ideas. Headline =
  tokens-to-target validation loss.
- [Chess RL](./docs/chess/CONTEXT.md) — a search-guided (AlphaZero-style) policy/value
  transformer trained from scratch on the ANE via RL self-play. Headline = Elo vs *capped*
  Stockfish. *(Doc lives in `docs/chess/` until the chess code layout is decided; it may move
  next to the code.)*

## Relationships

- **Shared substrate.** Both contexts compile down to the same ANE forward/backward
  transformer kernels, the compile-once dynamic pipeline, Muon/AdamW, fp16+loss-scaling, and
  the same CPU floor (dW / optimizer / loss). Chess *adds*, on top: a value head, a policy
  move-head, and a CPU-side self-play + MCTS environment.
- **LM → Chess.** The V4-ablation findings carry directly into the chess net's architecture
  scope: Muon wins; mHC is redundant-with-Muon and CPU-expensive; CSA/HCA are long-context-only.
- **Chess → North Star.** The chess-competent net is the intended *backbone* for a later
  language / teaching capability (the "chess tutor" — out of scope for v1).
