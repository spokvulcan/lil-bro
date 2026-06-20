#!/usr/bin/env python3
"""Turn scale-ladder run logs into val-vs-TRAINING-TOKENS curves + a tokens-to-target
table (the ADR-0003 headline). Reads each rung's per-cell trainer logs under
$LADDER_ROOT/<rung>/logs/r2run_<cell>.log, where every `[val] step=N val_loss=X` line is
one curve point. tokens = step * SEQ (SEQ=256, b=1 per micro-batch — train.m:619), so the
token axis is shared across rungs and val-vs-step IS val-vs-tokens.

No third-party deps (stdlib only). Usage:
    python training/sweep/analyze_ladder.py [--root /tmp/ladder] [--seq 256]
                                            [--targets 4.0,4.5,5.0]
Writes results/scale_ladder_curves.tsv and prints the optimum-LR + tokens-to-target tables.
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path

SEQ_DEFAULT = 256
RUNG_ORDER = ["r2_small", "r2_mid", "r3_110m"]
FAMILY_ORDER = ["adamw", "muon", "mhc2", "mhc4"]
VAL_RE = re.compile(r"\[val\]\s+step=(\d+)\s+val_loss=([0-9.eE+-]+)")

# cell name -> (family, lr_float, lr_label). e.g. "mhc4_lr3e2" -> ("mhc4", 0.03, "3e-2")
LR_MAP = {"3e4": 3e-4, "1e3": 1e-3, "3e3": 3e-3, "1e2": 1e-2, "3e2": 3e-2, "1e2b": 1e-2}


def parse_cell(cell: str):
    m = re.match(r"(.+)_lr([0-9a-z]+)$", cell)
    if not m:
        return cell, None, "?"
    fam, lrcode = m.group(1), m.group(2)
    # lrcode like "3e3" -> 3e-3, "1e2" -> 1e-2 (the harness drops the minus sign)
    mm = re.match(r"(\d)e(\d)$", lrcode)
    lr = float(f"{mm.group(1)}e-{mm.group(2)}") if mm else None
    label = f"{mm.group(1)}e-{mm.group(2)}" if mm else lrcode
    return fam, lr, label


def load_curve(log: Path):
    pts = []
    for ln in log.read_text(errors="ignore").splitlines():
        m = VAL_RE.search(ln)
        if m:
            pts.append((int(m.group(1)), float(m.group(2))))
    pts.sort()
    # de-dup by step (keep last) preserving order
    seen = {}
    for s, v in pts:
        seen[s] = v
    return sorted(seen.items())


def tokens_to_target(curve, target, seq):
    """First step whose val_loss <= target -> tokens. None if never reached."""
    for s, v in curve:
        if v <= target:
            return s * seq
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/tmp/ladder")
    ap.add_argument("--seq", type=int, default=SEQ_DEFAULT)
    ap.add_argument("--targets", default="")
    ap.add_argument("--out", default="results/scale_ladder_curves.tsv")
    args = ap.parse_args()
    root = Path(args.root)
    seq = args.seq

    # rung -> cell -> {curve, family, lr, lr_label, final, best}
    data: dict[str, dict] = {}
    for rung in (RUNG_ORDER + sorted(p.name for p in root.glob("*") if p.is_dir())):
        if rung in data:
            continue
        logdir = root / rung / "logs"
        if not logdir.is_dir():
            continue
        cells = {}
        for log in sorted(logdir.glob("r2run_*.log")):
            cell = log.stem[len("r2run_"):]
            curve = load_curve(log)
            if not curve:
                continue
            fam, lr, label = parse_cell(cell)
            cells[cell] = dict(curve=curve, family=fam, lr=lr, lr_label=label,
                               final=curve[-1][1], best=min(v for _, v in curve))
        if cells:
            data[rung] = cells

    if not data:
        print(f"No ladder logs found under {root}", file=sys.stderr)
        return 1

    # --- tidy long TSV ---
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = ["rung\tfamily\tlr\tcell\tstep\ttokens\tval_loss"]
    for rung, cells in data.items():
        for cell, d in sorted(cells.items()):
            for s, v in d["curve"]:
                rows.append(f"{rung}\t{d['family']}\t{d['lr_label']}\t{cell}\t{s}\t{s*seq}\t{v:.4f}")
    out.write_text("\n".join(rows) + "\n")
    print(f"wrote {out} ({len(rows)-1} curve points across {len(data)} rungs)\n")

    # --- optimum LR per (rung, family): best by final val; flag grid edges ---
    print("=== optimum LR per (rung x family) — by final val@budget ===")
    print(f"{'rung':9s} {'family':6s} {'best_lr':8s} {'final_val':9s} {'best_val':9s}  grid(final_val per lr)")
    best_lr_curve = {}  # (rung,family) -> chosen curve (lowest final)
    best_final = {}     # (rung,family) -> final val at best LR
    for rung in data:
        for fam in FAMILY_ORDER:
            fc = {c: d for c, d in data[rung].items() if d["family"] == fam}
            if not fc:
                continue
            grid = sorted(((d["lr"], d["lr_label"], d["final"], c) for c, d in fc.items()))
            best = min(grid, key=lambda g: g[2])
            lrs = [g[0] for g in grid]
            edge = ""
            if len(lrs) > 1 and best[0] in (min(lrs), max(lrs)):
                edge = "  <-EDGE(extend)"
            gridstr = " ".join(f"{lbl}={fv:.3f}" for _, lbl, fv, _ in grid)
            print(f"{rung:9s} {fam:6s} {best[1]:8s} {best[2]:<9.3f} "
                  f"{min(d['best'] for d in fc.values()):<9.3f}  {gridstr}{edge}")
            best_lr_curve[(rung, fam)] = fc[best[3]]["curve"]
            best_final[(rung, fam)] = best[2]
    print()

    # --- tokens-to-target (headline) at each family's best LR ---
    targets = [float(t) for t in args.targets.split(",") if t.strip()] if args.targets else []
    if not targets:
        # auto: round targets spanning the range all best-LR curves can reach
        finals = [c[-1][1] for c in best_lr_curve.values()]
        hi = max(finals)
        targets = sorted({round(hi + 0.05, 1), round(hi + 0.3, 1), round(hi + 0.8, 1)})
    print(f"=== tokens-to-target (best-LR curve per rung x family); SEQ={seq} ===")
    hdr = "rung      family " + " ".join(f"L<= {t:<6.2f}" for t in targets)
    print(hdr)
    for rung in data:
        for fam in FAMILY_ORDER:
            if (rung, fam) not in best_lr_curve:
                continue
            curve = best_lr_curve[(rung, fam)]
            cells_tt = []
            for t in targets:
                tok = tokens_to_target(curve, t, seq)
                cells_tt.append(f"{tok:>9,}" if tok is not None else f"{'NR':>9s}")
            print(f"{rung:9s} {fam:6s} " + " ".join(f"{c:>11s}" for c in cells_tt))
    print("\n(NR = target not reached within budget; tokens = step*SEQ)")

    # --- cross-size synthesis: the hold/shrink/invert verdict ---
    rungs_present = [r for r in (RUNG_ORDER + [x for x in data if x not in RUNG_ORDER]) if r in data]
    if len(rungs_present) >= 2:
        print("\n=== cross-size synthesis — best-LR val@budget per (family x rung) ===")
        print(f"{'family':6s} " + " ".join(f"{r:>9s}" for r in rungs_present))
        for fam in FAMILY_ORDER:
            cells = [f"{best_final[(r,fam)]:.3f}" if (r, fam) in best_final else "  -" for r in rungs_present]
            print(f"{fam:6s} " + " ".join(f"{c:>9s}" for c in cells))

        def gap(r, a, b):  # a - b at rung r, or None
            if (r, a) in best_final and (r, b) in best_final:
                return best_final[(r, a)] - best_final[(r, b)]
            return None

        def trendline(label, a, b, want):
            vals = [(r, gap(r, a, b)) for r in rungs_present]
            cells = " ".join(f"{r}={g:+.3f}" if g is not None else f"{r}=-" for r, g in vals)
            seq_ = [g for _, g in vals if g is not None]
            verdict = ""
            if len(seq_) >= 2:
                d = seq_[-1] - seq_[0]
                # 'want' is the sign that means "lever helps"; magnitude trend = hold/grow/shrink
                mag0, mag1 = abs(seq_[0]), abs(seq_[-1])
                if (seq_[0] > 0) != (seq_[-1] > 0) and min(mag0, mag1) > 0.02:
                    verdict = "INVERTS"
                elif mag1 > mag0 + 0.03:
                    verdict = "GROWS"
                elif mag1 < mag0 - 0.03:
                    verdict = "SHRINKS"
                else:
                    verdict = "HOLDS"
            print(f"  {label:28s} {cells}    -> {verdict}")

        print("\ngaps vs model size (− = first term better):")
        trendline("optimizer (adamw − muon)", "adamw", "muon", +1)   # + => muon better (dominance)
        trendline("mHC×4 redund (mhc4 − muon)", "mhc4", "muon", 0)    # ~0 => redundant; − => mHC helps
        trendline("mHC×2 redund (mhc2 − muon)", "mhc2", "muon", 0)
        print("  (optimizer: larger + = stronger Muon dominance | mHC: − means mHC beats bare Muon)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
