#!/usr/bin/env python3
"""Per-parameter gradient diff for the V2 placement-toggle equivalence gate.

Compares two raw-gradient dumps (train.m --dump-grads, flat float32 in
dump_grads() order) and reports per-tensor cosine + relative-L2. Used to gate a
per-op CPU->ANE placement migration: the current CPU placement is already
R1-green vs the torch-fp64 oracle (results/r1_grad_diff.md), so an ANE build
whose grads match the CPU build within the fp16 noise floor is itself correct.

Gate (matches results/r1_grad_diff.md): cosine >= 0.99 AND rel_l2 <= 0.10 for
every parameter. fp16 ANE matmuls make direction+magnitude the right yardstick,
not element-wise max-abs.

  python3 grad_diff.py g_cpu.bin g_ane.bin [--dims vocab,dim,hidden,qdim,kvdim,nlayers]

Default dims are stories110m (32000,768,2048,768,768,12).
"""
import sys
import numpy as np

def layout(vocab, dim, hidden, qdim, kvdim, nlayers):
    """Yield (name, nelem) in dump_grads() order."""
    yield ("embed", vocab * dim)
    for L in range(nlayers):
        yield (f"L{L}.Wq", qdim * dim)
        yield (f"L{L}.Wk", kvdim * dim)
        yield (f"L{L}.Wv", kvdim * dim)
        yield (f"L{L}.Wo", dim * qdim)
        yield (f"L{L}.W1", hidden * dim)
        yield (f"L{L}.W2", dim * hidden)
        yield (f"L{L}.W3", hidden * dim)
        yield (f"L{L}.rms_att", dim)
        yield (f"L{L}.rms_ffn", dim)
    yield ("rms_final", dim)

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dims = (32000, 768, 2048, 768, 768, 12)
    for a in sys.argv[1:]:
        if a.startswith("--dims"):
            dims = tuple(int(x) for x in a.split("=", 1)[1].split(","))
    fa, fb = args[0], args[1]
    a = np.fromfile(fa, dtype=np.float32)
    b = np.fromfile(fb, dtype=np.float32)
    if a.shape != b.shape:
        print(f"SIZE MISMATCH {fa}:{a.size} vs {fb}:{b.size}")
        sys.exit(2)

    names = list(layout(*dims))
    total = sum(n for _, n in names)
    if total != a.size:
        print(f"LAYOUT MISMATCH: dims imply {total} elems, file has {a.size}")
        print("  (pass the right --dims=vocab,dim,hidden,qdim,kvdim,nlayers)")
        sys.exit(2)

    worst_cos, worst_cos_t = 1.0, ""
    worst_rl2, worst_rl2_t = 0.0, ""
    off = 0
    for name, n in names:
        x, y = a[off:off+n], b[off:off+n]
        off += n
        nx, ny = np.linalg.norm(x), np.linalg.norm(y)
        if nx == 0 and ny == 0:
            continue
        cos = float(np.dot(x, y) / (nx * ny + 1e-30))
        rl2 = float(np.linalg.norm(x - y) / (nx + 1e-30))
        if cos < worst_cos:
            worst_cos, worst_cos_t = cos, name
        if rl2 > worst_rl2:
            worst_rl2, worst_rl2_t = rl2, name

    ok = worst_cos >= 0.99 and worst_rl2 <= 0.10
    print(f"params={len(names)} elems={a.size}")
    print(f"worst_cosine={worst_cos:.5f} @{worst_cos_t}")
    print(f"worst_rel_l2={worst_rl2:.4f} @{worst_rl2_t}")
    print(f"R1 {'PASS' if ok else 'FAIL'}  (gate: cos>=0.99 rel_l2<=0.10)")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
