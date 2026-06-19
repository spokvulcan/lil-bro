# CLAUDE.md

## What this is — and why

lil-bro is a research fork of [maderix/ANE](https://github.com/maderix/ANE) (MIT)
that trains a small dense transformer **from scratch on the Apple Neural Engine**
and measures whether **DeepSeek-V4** ideas improve the small-dense efficiency
frontier.

But the reason it exists runs deeper. Apple ships the ANE as *inference-only* —
the silicon can train; the software was never allowed to. This project is the
living proof that the barrier was always software, not hardware. The only real
failure here is deciding something is impossible before you've actually tried it.

So the philosophy:

- **Assume it's possible until the hardware says no.** "The ANE can't do that" is
  a hypothesis to attack, not a fact to accept. Every workaround already in this
  codebase exists because someone refused to take the first "no" for an answer.
- **Attack from another angle.** A wall on one path — an op won't compile, a
  request errors, a gradient underflows — is a signal to reframe, not to stop.
  There is almost always another decomposition.
- **Aim past the frontier, don't admire it.** We're at the edge of what's
  possible on-device; aim *beyond* it in every decision. Never write an idea off
  as "too ambitious for an NPU."
- **Never give up quietly.** Try one more angle, then another. If you're truly
  stuck, say exactly what you tried and where it broke — so the next attempt
  starts ahead, not from zero.

## Evidence before assertion

Ambition is worth nothing here if the numbers can't be trusted — the whole point
is telling "genuinely learning" apart from "silently wrong gradients." So **check
before you claim.** Any load-bearing statement must rest on something you read or
ran *this session* — a file:line, a config value, a measured number — not on what
is plausible.

- **Name the evidence, or mark it unverified.**
- **Keep "I verified X" separate from "I suspect Y."**
- **A correction means re-derive from evidence, not re-guess** — return to
  primary sources and rebuild; don't fire off another unchecked theory.

Match the ambition above with this rigor: a frontier result nobody can reproduce
is not a result.

## Working here

```bash
cd training/training_dynamic
make MODEL=stories110m   # or MODEL=qwen3_06b (default)
./train --scratch        # from random init   (--resume to continue)
```

Data: `cd training && bash download_data.sh` (`data00` train / `data01` val).
No external deps on the ANE side — add the op or do it on CPU, don't add a
dependency. `origin` = `spokvulcan/lil-bro`, `upstream` = `maderix/ANE`; pull
fixes via `upstream` and keep MIT attribution (`NOTICE`, `LICENSE`) intact.
Commits follow Conventional Commits.

## Subagents & workflows

- **Model: Opus-tier, always** — spawn subagents and workflow agents on Opus 4.8
  (`opus`) or higher; never downgrade to save cost.
- **Thinking effort: flexible** — low/medium for easy work, high/max for hard
  tasks and implementation.

## Docs (read before touching the area)

Thesis & results → `README.md` · phasing, ladder & gates → `ROADMAP.md` · full
spec → `docs/PRD.md` · ANE trainer internals → `training/README.md` · subpackage
intent → `lilbro/*/README.md`. Issues live in `spokvulcan/lil-bro` (`gh` CLI).
