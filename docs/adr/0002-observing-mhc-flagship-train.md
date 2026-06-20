# ADR 0002 — Observing the mHC flagship train: faithful in-trainer sampling + a v5 checkpoint

- **Status:** Accepted — ratified via `/grill-with-docs` on 2026-06-20 (decisions table below)
- **Date:** 2026-06-20
- **Context docs:** [ADR 0001](0001-deepseek-v4-for-dense-ane.md) (mHC design) · [dashboard.py](../../training/dashboard.py) · [training/README](../../training/README.md) · [results/mhc.md](../../results/mhc.md)
- **Supersedes nothing.** Adds the *observability* layer on top of the now-complete PRD #2 mechanisms.

---

## Context

PRD #2 is implementation-complete: all nine V4 child mechanisms are R0-green, mHC
(#11) included. The next thing we want is to **watch the flagship actually train** —
run `gen_r2_small` with mHC (`N_HC=2`) on TinyStories, wired to
[`dashboard.py`](../../training/dashboard.py), and sample text periodically to see
progress qualitatively.

The dashboard already has the machinery: `spawn_training` launches the trainer and
parses its stdout; `generation_thread` reloads the checkpoint every 100 steps and runs
a **numpy CPU forward** (`generate_text`) to show generated text; the trainer saves the
checkpoint every 100 steps on loss improvement (`train.m:1747`); the checkpoint header
is exactly the 96 bytes the loader assumes, with `embed` written before the V4 extras
so the loader stays aligned.

But two facts collide:

1. **The dashboard's numpy forward is a plain GQA transformer.** It implements none of
   the V4 mechanisms (mHC streams, MTP, qk-norm, sink, partial RoPE). Verified: every
   V4 knob defaults **off** in `config.h` (`QK_NORM=0`, `ATTN_SINK=0`, `SWIGLU_CLAMP=0`,
   `ROPE_ROTARY_DIMS=HD`, `N_HC=1`, `MTP_DEPTH=0`), and `gen_r2_small.h` overrides none
   of them — so a *stock* `gen_r2_small` build is the one config the dashboard can
   faithfully sample. Turn on mHC and the generated-text panel samples a **different,
   wrong model**.
2. **mHC params are not in the checkpoint.** `save_checkpoint` writes only the vanilla
   transformer weights (+ `#if ATTN_SINK`/`#if QK_NORM` extras); the `g_mapA[NLAYERS]` /
   `g_mapF[NLAYERS]` maps (`W*/S*/α`) are absent (`resume` re-inits them, per #11's
   commit). So even a *correct* mHC sampler would read maps that aren't there.

Sampling the flagship faithfully therefore requires changes on both sides. The whole
project ethos is **evidence before assertion**: "faithful text" cannot be a
plausible-looking numpy reimplementation we hope is right — it has to be faithful *by
construction*.

## Ratified decisions (grilling — 2026-06-20)

| Decision | Choice | One-line rationale |
|---|---|---|
| **Model under glass** | Stock `gen_r2_small` + `EXTRA=-DN_HC=2` (mHC flagship), from scratch on TinyStories (`data00` train / `data01` val) | The flagship is the point of the exercise; `r2_small` is the smallest "real-text" rung (32K vocab, `SEQ=256`, 6 layers) the 32K tokenizer can read back. |
| **Progress signal** | **Faithful generated text** — not a held-out val-loss surrogate | The ask is qualitative ("see it learn"). `[val] val_loss` (already faithful via `eval_val_loss`) is kept as a cheap secondary, but text is the chosen signal. |
| **mHC in the checkpoint** | Persist `W*/S*/α` **+ Adam `m,v`**, build-tied, bump `CkptHdr.version` 4→5 | Sampling needs the maps; saving Adam makes `--resume` correct (fixes the known re-init limitation), and it is ~0.6 MB against a 152 MB file — free. v5 makes a mismatched build refuse to load rather than read EOF garbage. |
| **The faithful forward** | **Reuse the FD-verified C `forward_hidden`** — no numpy re-implementation | A second mHC implementation (entry broadcast, A/B/C maps, Sinkhorn, recombine, exit-sum) could drift; "faithful" must hold *by construction*, not by an unverified port we'd have to separately cross-check. |
| **Sampler locus** | **In-trainer periodic emission** — print `[gen step=N] <token-ids>`; the dashboard decodes with its existing 32K tokenizer | No second ANE client ⇒ no contention with the live trainer, no checkpoint round-trip, trainer stays tokenizer-free (token-id space). Clones the existing `eval_val_loss` call pattern. |

## Checkpoint v5 layout (build-tied)

The format stays **build-tied, not self-describing** (the existing convention: the
loader reads `#if ATTN_SINK`/`#if QK_NORM` blocks only when those knobs are compiled in;
the 96-byte header carries no mechanism flags). v5 appends one block, written **after**
`embed`/`sink`/`qnorm` so older readers that stop at `embed` stay aligned:

```
... embed, embed_adam(m,v)          [v4 layout, unchanged]
#if ATTN_SINK   attn_sink (+adam)   [unchanged]
#if QK_NORM     qnorm_w, knorm_w (+adam)   [unchanged]
#if N_HC > 1    ── NEW v5 block ──
  for sub in {A (attention), F (ffn)}:
    for L in 0..NLAYERS-1:
      Wpre[M·N_HC], Wres[M·N_HC²], Wpost[M·N_HC],
      Spre[N_HC],   Sres[N_HC²],   Spost[N_HC],
      a_pre, a_res, a_post                      (M = N_HC·DIM, float32)
      + Adam m,v for each of the above
```

`CkptHdr.version` becomes 5; `load_checkpoint` rejects `version != 5`. Because the
flagship is `N_HC=2` and has no v4 ancestor that carries mHC maps, it always trains
**from scratch** — invalidating v4 checkpoints for v5 builds is costless here.

The dashboard does **not** need to read this block (the sampler runs in the trainer);
its only checkpoint use stays the vanilla `embed` load it already does — and for the
flagship that path is disabled (see below).

## In-trainer sampler

A sibling of `eval_val_loss`: same `forward_hidden(..., &g_mhc_val, ...)` call on the
resident weights + `cembed` (compact head) + `VocabMap`, but autoregressive over a
prompt instead of CE over held-out data.

- **Flags:** `--sample-every K` (0 = off), `--sample-tokens N` (default 64),
  `--sample-prompt-ids "1 ..."` (default: BOS only). The dashboard tokenizes a
  human prompt and passes the ids, keeping the C side tokenizer-free.
- **Decode loop:** full-sequence recompute per token (fixed-shape kernels, provably
  matches the trainer's full-seq forward) → compact-head logits via `cblas_sgemm` with
  `cembed` → temperature/top-k sample → map compact→full id → append → repeat.
- **Output:** `[gen step=%d] %d %d %d ...` (full token ids). The dashboard adds a regex
  for this line, decodes with its `Tokenizer`, and renders it in the existing Generated
  Text panel.

## Dashboard wiring

1. New `MODEL_CONFIGS` entry for the flagship (r2_small dims + `n_hc`/`attn_sink`/
   `qk_norm` flags + ckpt path) so `make MODEL=gen_r2_small EXTRA=-DN_HC=2` is launchable
   and the panel header is right.
2. New regex to parse `[gen step=N] <ids>` → decode → Generated Text panel.
3. **Disable the built-in numpy generator** for the flagship (it samples the wrong,
   mHC-less model). The faithful `[gen]` lines replace it.
4. (Cheap secondary) parse `[val] val_loss=…` so the held-out curve is visible too.

## Consequences

- **Fidelity is by construction.** The text on screen comes from the exact, FD-verified
  forward the model trains with — there is no second implementation that could be subtly
  wrong. This is the whole reason for the in-trainer choice over a numpy port.
- **Resume becomes correct for mHC** (Adam persisted), closing a #11 follow-up.
- **The v5 format rejects v4 checkpoints** under an `N_HC>1` build. Acceptable: the
  flagship is scratch-only anyway. A vanilla `N_HC=1` build still writes/reads its own
  v5 (no mHC block) and is byte-compatible in spirit with the old layout up to the
  version int.
- **Early samples are gibberish.** A 6-layer / DIM-256 model from random init needs
  thousands of steps before text is coherent; the panel will show noise first. That is
  expected, not a bug — the loss/val curve is the early signal, text is the late one.
- **Sampling costs a little throughput.** `N` extra forward passes every `K` steps
  (e.g. 64 every 500) — a few percent at most; tune `K` if it bites.

## Next steps (implementation)

1. `train.m`: v5 checkpoint save/load of the mHC block (weights + Adam), `#if N_HC>1`.
2. `train.m`: `--sample-every`/`--sample-tokens`/`--sample-prompt-ids` + the
   `sample_tokens()` routine cloned from `eval_val_loss`; emit `[gen step=N] <ids>`.
3. `dashboard.py`: flagship `MODEL_CONFIGS` entry; `[gen]`/`[val]` regexes; disable the
   numpy generator for the flagship.
4. Verify before the long run: `N_HC=2` R0 still green (regression); v5 round-trips
   (save→load→identical loss, i.e. resume correctness); one sample emits and decodes in
   the dashboard.
5. Launch the flagship on TinyStories under the dashboard and watch loss + `[val]` + text.
