# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root — the glossary of contested/load-bearing terms (DeepSeek-V4 parity, mHC, residual stream, Tier A/B/C/D).
- **`docs/adr/`** — read ADRs that touch the area you're about to work in (currently `0001-deepseek-v4-for-dense-ane.md`).

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront. The `/domain-modeling` skill (reached via `/grill-with-docs` and `/improve-codebase-architecture`) creates them lazily when terms or decisions actually get resolved.

## File structure

This is a single-context repo — one `CONTEXT.md` + `docs/adr/` at the root:

```
/
├── CONTEXT.md
├── docs/
│   ├── PRD.md
│   └── adr/
│       └── 0001-deepseek-v4-for-dense-ane.md
└── ...
```

There is no `CONTEXT-MAP.md`. If one ever appears at the root, this repo has gone
multi-context: read each per-context `CONTEXT.md` it points at, and also check
`src/<context>/docs/adr/` for context-scoped decisions.

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids — e.g. say "mHC", not "hyper-connections"; say "DeepSeek-V4 parity" only in the scoped sense the glossary pins down, not "V4-equivalent".

If the concept you need isn't in the glossary yet, that's a signal — either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/domain-modeling`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0001 (DeepSeek-V4 for dense ANE) — but worth reopening because…_
