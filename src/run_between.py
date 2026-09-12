"""Ablation stage 1 - the projection using S_B only."""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import lib  

spec = importlib.util.spec_from_file_location("run07", SRC / "07_run.py")
run07 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run07)

ROOT, RES = SRC.parent, SRC.parent / "results"
ZDIR = ROOT / "cache" / "Z"
K, METHOD = 4, "between"
SEEDS = (0, 1, 2)

cfg = run07.load_rafdb(encoder="clip")
U, tr, A = cfg["U"], cfg["tr"], cfg["A"]
holdout = run07.pick_holdout(A, tr, cfg["n_holdout"])
keep = ~np.isin(A, holdout)
ktr, kte = keep[tr], keep[~tr]
htr, hte = ~keep[tr], ~keep[~tr]


Utr_k, ytr_k = U[tr][ktr], A[tr][ktr]
n_kept = len(np.unique(ytr_k))
print(f"kept {n_kept} classes  train {ktr.sum():,}   k={K}")

rows, prows = [], []
for seed in SEEDS:
    f = lib.METHODS[METHOD](Utr_k, ytr_k, K, n_kept, seed=seed)
    Ztr, Zte = f(U[tr]), f(U[~tr])
    np.savez_compressed(ZDIR / f"rafdb_{METHOD}_k{K}_s{seed}.npz",
                        Ztr=Ztr, Zte=Zte)
    base = dict(dataset="rafdb", encoder=cfg["encoder"], method=METHOD,
                D=U.shape[1], k=Ztr.shape[1], transmitted_bytes=Ztr.shape[1] * 4,
                seed=seed, note="")

    targets = []
    rm = {c: i for i, c in enumerate(sorted(np.unique(A[tr][ktr])))}
    targets.append(("A_kept", Ztr[ktr], np.array([rm[v] for v in A[tr][ktr]]),
                    Zte[kte], np.array([rm[v] for v in A[~tr][kte]]),
                    run07.FAM_UTILITY))
    rmh = {c: i for i, c in enumerate(sorted(np.unique(A[tr][htr])))}
    targets.append(("Aprime_heldout", Ztr[htr],
                    np.array([rmh[v] for v in A[tr][htr]]), Zte[hte],
                    np.array([rmh.get(v, 0) for v in A[~tr][hte]]),
                    run07.FAM_UTILITY))
    for bname, bval in cfg["B"].items():
        ex = cfg["B_exclude"].get(bname)
        mt = ~ex[tr] if ex is not None else np.ones(tr.sum(), bool)
        me = ~ex[~tr] if ex is not None else np.ones((~tr).sum(), bool)
        targets.append((f"B_{bname}", Ztr[mt], bval[tr][mt], Zte[me],
                        bval[~tr][me], run07.FAM_ATTACK))

    for tname, ztr, ytr_, zte, yte_, fams in targets:
        t0 = time.time()
        best, runs = lib.probe(ztr, ytr_, zte, yte_, seed_base=seed, families=fams)
        rows.append({**base, "target": tname, "target_type": "classification",
                     "n_classes": int(len(np.unique(ytr_))),
                     "n_train": len(ztr), "n_test": len(zte),
                     "chance": pd.Series(yte_).value_counts(normalize=True).iloc[0],
                     "chance_balanced": 1.0 / len(np.unique(yte_)),
                     "best_family": best["family"], "best_cfg": best["cfg"],
                     "train_acc": best["train_acc"], "train_bal": best["train_bal"],
                     "test_acc": best["test_acc"], "test_bal": best["test_bal"],
                     "sec": time.time() - t0})
        for r in runs:
            prows.append({**base, "target": tname, **r})
        print(f"  seed{seed}  {tname:16s} {best['test_bal']:.3f}  ({best['family']})")

MAIN, PROBE = RES / "main_rafdb.csv", RES / "probe_runs_rafdb.csv"
for path, new in ((MAIN, rows), (PROBE, prows)):
    old = pd.read_csv(path)
    old = old[old.method.astype(str) != METHOD]        
    pd.concat([old, pd.DataFrame(new)], ignore_index=True).to_csv(path, index=False)
    print(f"[wrote] {path.name}   +{len(new)} rows")
