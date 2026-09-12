"""Stage-wise arithmetization error over f."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import quant_sweep as Q  

ROOT = SRC.parent
OUT = ROOT / "results" / "quant_error_ablation.csv"
FBITS = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24)


def staged(U, W, M, b, f, upto):
    S = 1 << f
    Uq = np.rint(U * S).astype(np.int64)
    if upto == 1:                      
        return (Uq / S) @ W @ M + b
    Wq = np.rint(W * S).astype(np.int64)
    Zq = Q.rescale(Uq @ Wq, f)
    if upto == 2:                      
        return (Zq / S) @ M + b
    Mq = np.rint(M * S).astype(np.int64)
    bq = np.rint(b * S).astype(np.int64)
    return (Q.rescale(Zq @ Mq, f) + bq) / S


def main():
    U = np.load(ROOT / "cache" / "embeddings" / "rafdb_clip.npy").astype(np.float64)
    A = Q.LAB["emotion"]
    ho = Q.pick_holdout(A, Q.N_HOLDOUT)
    keep = ~np.isin(A, ho)
    W, M, b = Q.fit_tap(U[keep & Q.tr], A[keep & Q.tr])
    ref = Q.apply_tap(U, W, M, b, None)
    nrm = np.linalg.norm(ref)

    rows = []
    for f in FBITS:
        cum = {}
        for upto, name in ((1, "encode U"), (2, "encode W + rescale"),
                           (3, "encode M,b + rescale")):
            dv = staged(U, W, M, b, f, upto) - ref
            cum[upto] = dict(rel=float(np.linalg.norm(dv) / nrm),
                             mx=float(np.abs(dv).max()))
            rows.append(dict(
                f=f, stage=upto, stage_name=name,
                cum_rel_l2=cum[upto]["rel"], cum_abs_max=cum[upto]["mx"],
                inc_rel_l2=cum[upto]["rel"] - (cum[upto - 1]["rel"] if upto > 1 else 0.0),
                inc_abs_max=cum[upto]["mx"] - (cum[upto - 1]["mx"] if upto > 1 else 0.0)))

        chk = np.abs(staged(U, W, M, b, f, 3) - Q.apply_tap(U, W, M, b, f)).max()
        assert chk == 0, f"f={f}: stage 3 differs from the deployed computation ({chk})"

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    pv = df.pivot(index="f", columns="stage_name", values="cum_rel_l2")
    pv = pv[["encode U", "encode W + rescale", "encode M,b + rescale"]]
    print("cumulative relative error, stages switched on one at a time:")
    print(pv.to_string(float_format=lambda x: f"{x:.3e}"))
    inc = df.pivot(index="f", columns="stage_name", values="inc_rel_l2")
    print("\nper-stage increment:")
    print(inc[pv.columns].to_string(float_format=lambda x: f"{x:+.3e}"))
    share = (inc[pv.columns].div(pv.iloc[:, -1], axis=0) * 100)
    print("\nshare of the total error (%):")
    print(share.to_string(float_format=lambda x: f"{x:+6.1f}"))
    print(f"\n[wrote] {OUT}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
