"""Receiver compatibility under fixed-point arithmetic.

Figure 4 feeds a server model trained on raw U the real-valued Uhat. What is
actually transmitted is Uhat_f, the circuit's output. This repeats the same
measurement on both, so the arithmetization can be read off against a fixed
receiver rather than a retrained probe.
"""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import lib                                                       # noqa: E402

spec = importlib.util.spec_from_file_location("run07", SRC / "07_run.py")
run07 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run07)

ROOT, RES = SRC.parent, SRC.parent / "results"
PARAM = RES / "tap_params"
OUT = RES / "compat_quant.csv"
SERVER_SEEDS = (0, 1, 2)
F = 16


def rescale(acc, f):
    return (acc + (1 << (f - 1))) >> f


def apply_tap(U, W, M, b, f=None):
    if f is None:
        return (U @ W) @ M + b
    S = 1 << f
    Ut, Wt, Mt, bt = (np.rint(x * S).astype(np.int64) for x in (U, W, M, b))
    Zt = rescale(Ut @ Wt, f)
    return (rescale(Zt @ Mt, f) + bt).astype(np.float64) / S


def main():
    cfg = run07.load_rafdb(encoder="clip")
    U, tr, A = cfg["U"], cfg["tr"], cfg["A"]
    holdout = run07.pick_holdout(A, tr, cfg["n_holdout"])
    keep = ~np.isin(A, holdout)

    W, M, b = (np.load(PARAM / f"{n}.npy") for n in ("W", "M", "b"))
    Ute = U[~tr]
    UH_real = apply_tap(Ute, W, M, b, None)
    UH_f = apply_tap(Ute, W, M, b, F)
    eps = float(np.linalg.norm(UH_f - UH_real) / np.linalg.norm(UH_real))
    print(f"f={F}  rel L2 of Uhat_f against Uhat_real = {eps:.3e}", flush=True)

    targets = [("compat_A", A, keep)]
    for bname, bval in cfg["B"].items():
        ex = cfg["B_exclude"].get(bname)
        targets.append((f"compat_B_{bname}", bval,
                        ~ex if ex is not None else np.ones(len(U), bool)))

    rows = []
    for tname, lab, msk in targets:
        mt, me = msk[tr], msk[~tr]
        rm = {c: i for i, c in enumerate(sorted(np.unique(lab[msk])))}
        ytr = np.array([rm[v] for v in lab[tr][mt]])
        yte = np.array([rm.get(v, 0) for v in lab[~tr][me]])
        for s in SERVER_SEEDS:
            t0 = time.time()
            preds = lib.probe_fitted(U[tr][mt], ytr, seed_base=s)
            for fam, cfgname, predict in preds:
                rows.append(dict(
                    dataset="rafdb", target=tname, family=fam, cfg=cfgname,
                    seed=s, f=F, method="lda_shrink", note="compat_quant",
                    on_U=balanced_accuracy_score(yte, predict(U[~tr][me])),
                    on_Uhat_real=balanced_accuracy_score(
                        yte, predict(UH_real[me])),
                    on_Uhat_f=balanced_accuracy_score(yte, predict(UH_f[me])),
                    rel_err=eps))
            print(f"  {tname:18s} seed {s}  {len(preds)} configs  "
                  f"({time.time() - t0:.0f} s)", flush=True)

    df = pd.DataFrame(rows)
    df["delta"] = df.on_Uhat_f - df.on_Uhat_real
    df.to_csv(OUT, index=False)
    print(f"\n[wrote] {OUT}  ({len(df)} rows)")
    print(f"max |delta| over all rows: {df.delta.abs().max():.4f}")


if __name__ == "__main__":
    main()
