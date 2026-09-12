"""Arithmetization error of the factored form against the direct one."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import quant_sweep as qs  

ROOT = SRC.parent
OUT = ROOT / "results" / "quant_form.csv"
FBITS = (4, 8, 12, 16, 20, 24)


def apply_direct(U, W, M, b, f):
    S = 1 << f
    Ut = np.rint(U * S).astype(np.int64)
    Pt = np.rint((W @ M) * S).astype(np.int64)
    bt = np.rint(b * S).astype(np.int64)
    acc = Ut @ Pt
    if np.abs(acc).max() > (1 << 62):
        raise SystemExit(f"f={f} : int64 accumulator overflows")
    return (qs.rescale(acc, f) + bt).astype(np.float64) / S


def main():
    U = np.load(ROOT / "cache" / "embeddings" / "rafdb_clip.npy").astype(np.float64)
    A = qs.LAB["emotion"]
    ho = qs.pick_holdout(A, qs.N_HOLDOUT)
    ktr = (~np.isin(A, ho)) & qs.tr
    W, M, b = qs.fit_tap(U[ktr], A[ktr])

    ref = qs.apply_tap(U, W, M, b, None)          
    nrm = np.linalg.norm(ref)
    rows = []
    for f in FBITS:
        d_fac = qs.apply_tap(U, W, M, b, f) - ref
        d_dir = apply_direct(U, W, M, b, f) - ref
        rows.append(dict(
            f=f,
            fac_abs_max=float(np.abs(d_fac).max()),
            fac_rel_l2=float(np.linalg.norm(d_fac) / nrm),
            dir_abs_max=float(np.abs(d_dir).max()),
            dir_rel_l2=float(np.linalg.norm(d_dir) / nrm),
        ))
    df = pd.DataFrame(rows)
    df["ratio"] = df.dir_rel_l2 / df.fac_rel_l2
    df.to_csv(OUT, index=False)

    print(f"|W|max={np.abs(W).max():.3f}  |M|max={np.abs(M).max():.3f}  "
          f"|P|max={np.abs(W @ M).max():.4f}")
    print(df.to_string(index=False, float_format="%.3e"))
    print(f"\n[wrote] {OUT}")


if __name__ == "__main__":
    main()
