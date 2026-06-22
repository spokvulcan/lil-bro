# lil-bro Chess — Context

Glossary for the chess RL self-play project: a search-guided (AlphaZero-style) policy/value
transformer trained from scratch on the Apple Neural Engine. Sibling context to the root
[CONTEXT.md](../../CONTEXT.md) (the DeepSeek-V4 → small-dense-LM ablation); both sit on the
same ANE-trainer substrate. Decisions in [docs/adr/](../adr/).

## Language

**method-first** (vs *strength-first*):
The project's governing priority — the deliverable is a self-play RL loop that *provably*
learns chess *on the ANE*, not the strongest engine obtainable on a Mac. When the RL method
and raw strength conflict, the method wins. (Strength-first would point at supervised
Stockfish distillation, which we reject — [ADR 0005](../adr/0005-chess-rl-self-play-on-ane.md).)

**yardstick** (not *bar*):
"Compete with Stockfish" is a *measuring stick*, not a pass/fail bar: success = a *climbing*
Elo curve that beats **capped** Stockfish levels — not parity with full-strength Stockfish
(≈infeasible on a Mac from scratch).
_Avoid_: "beat Stockfish" (unqualified).

**self-anchored Elo**:
The headline strength metric — relative Elo from a round-robin among the net's *own past
checkpoints* (AlphaZero's own-curve): cheap, smooth, continuously available. Absolute strength
is anchored only *periodically* by calibration games vs (capped) Stockfish, which stays an
occasional eval tool, not a continuous dependency. The proof of "genuinely learning" is the
*monotonic climb* of this curve.
_Avoid_: implying the day-to-day metric is games-vs-Stockfish (that's the calibration, not the curve).

**search-guided self-play**:
Self-play where MCTS (Gumbel-AlphaZero, low-simulation) is the *policy-improvement operator*:
the net plays itself, MCTS sharpens the raw policy into a stronger target, and training
regresses the policy toward the search visit-distribution and the value toward the game
outcome. Distinct from search-free policy-gradient self-play (the known-weak path, rejected).
_Avoid_: "self-play" unqualified — it hides whether search is in the loop.

**policy/value net**:
The trained network: a shared transformer trunk with two heads — a **policy head**
(per-square 8×8×73 = 4672 move logits, legal-masked before softmax; reuses the LM
classifier-head pattern) and a **value head** (3-way Win/Draw/Loss softmax from a pooled
representation; genuinely new — no scalar/WDL head exists in the LM trainer).

**CPU/ANE split** (chess):
The LM trainer's division of labor, extended: the **ANE** evaluates the net (policy+value
forward, and training fwd + bwd-dx); the **CPU** runs everything tree-/control-shaped — MCTS,
legal-move generation, the game environment, the replay buffer — plus the existing CPU floor
(dW, optimizer, loss). "In parallel" = batching many games' MCTS leaf positions into one ANE
forward.

**GPU iteration path** (vs *ANE thesis path*):
The MPSGraph fp32 forward+backward learner path used for fast iteration — ~10x faster than the
ANE/CPU learner path, and the path all G2-learning-quality work runs on. The ANE forward path
stays as a **non-default compile option** so the "ANE can train" thesis (ADR 0001/0004/0005)
stays documented and buildable. The GPU port subsumes the backward NaN fix by construction
(fp32 rsqrt vs the CPU `rmsnorm_bwd` `vvrsqrtf` overflow), recasting QK-norm/SwiGLU-clamp as
V4-fidelity ablations on a stable base, not bug-fixes ([ADR 0006](../adr/0006-g2-learning-quality-gpu-path.md)).
_Avoid_: "the GPU path" unqualified — it hides that the ANE thesis path is deliberately retained.

**V4-inspired** (chess scope):
The V4 ideas that transfer to a *small, short-context* net — **Muon** (the proven winner) and
the cheap stabilizers (qk-norm / attn-sink / swiglu-clamp / partial-RoPE), plus a deliberate
fork on mHC and MTP. It explicitly does **not** mean V4's headline **CSA/HCA** hybrid
attention, a million-token-context mechanism irrelevant at chess's ~77-token context (root
[CONTEXT.md](../../CONTEXT.md) Tier C; [ADR 0001](../adr/0001-deepseek-v4-for-dense-ane.md)).

**chess tutor** (North Star):
The out-of-scope-for-v1 end goal: a model that not only plays but *explains* positions, moves,
and reasoning, and *teaches* humans at chosen Elo levels. v1 is play-only; the chess-competent
net is the *backbone* a later language phase builds on. v1 keeps the door open for ~free via
the shared-trunk-multi-head design + a growable vocab — nothing more.

**gate ladder** (perft / G0–G3):
The chess analog of the LM R-ladder — nothing downstream is trusted until its gate is green
("no win from a silently-wrong backward pass"). **perft** = movegen correctness; **G0** =
heads/loss overfit-one-batch → ~0 (new kernels correct on fp16 ANE); **G1** = Gumbel-MCTS
solves known tactics (search correct); **G2** = self-play win-rate climbs vs a fixed baseline
(the loop learns); **G3** = self-anchored Elo climbs (the headline).

**throughput probe**:
The first thing built, before any kernel work: a measurement of batched forward-only ANE eval
at chess shapes + stub movegen + stub MCTS → **games/day**. It gates the whole project and sets
net size / sims-per-move / parallel-game count. The number is *measured*, never guessed.
_Avoid_: committing scale by intuition before the probe.
