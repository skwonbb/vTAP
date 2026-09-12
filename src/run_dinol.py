"""Encoder swap to DINOv2 ViT-L/14."""
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
ZDIR.mkdir(parents=True, exist_ok=True)
SEEDS = (0, 1, 2)
JOBS = [("lda_shrink", None), ("vib_b1", None),
        ("adv2attr_lam03", None), ("leace2attr", None)]


cfg = run07.load_rafdb(encoder="clip")
cfg["U"] = np.load(ROOT / "cache" / "embeddings" / "rafdb_dinol.npy")
cfg["name"], cfg["encoder"] = "rafdb_dinol", "dinov2-l14"
U, tr, A = cfg["U"], cfg["tr"], cfg["A"]
D = U.shape[1]
holdout = run07.pick_holdout(A, tr, cfg["n_holdout"])
keep = ~np.isin(A, holdout)
ktr, kte = keep[tr], keep[~tr]
htr, hte = ~ktr, ~kte
n_kept = len(np.unique(A[tr][ktr]))
K = n_kept - 1
print(f"U {U.shape}   kept {n_kept} classes   k=C-1={K}\n", flush=True)

b_names = list(cfg["B"])
b_cols = np.stack([cfg["B"][n][tr][ktr] for n in b_names], 1)

TARGETS = [("A_kept", A, keep, run07.FAM_ATTACK),
           ("Aprime_heldout", A, ~keep, run07.FAM_ATTACK)]
for bn, bv in cfg["B"].items():
    ex = cfg["B_exclude"].get(bn)
    TARGETS.append((f"B_{bn}", bv,
                    ~ex if ex is not None else np.ones(len(U), bool),
                    run07.FAM_ATTACK))


def measure(meth, Ztr, Zte, seed, mrows, prows):
    kk = Ztr.shape[1]
    base = dict(dataset=cfg["name"], encoder=cfg["encoder"], method=meth,
                D=D, k=kk, transmitted_bytes=kk * 4, seed=seed, note="")
    for tname, lab, msk, fams in TARGETS:
        mt, me = msk[tr], msk[~tr]
        if mt.sum() <= 20 or me.sum() <= 20:
            continue
        rm = {c: i for i, c in enumerate(sorted(np.unique(lab[tr][mt])))}
        ytr = np.array([rm[v] for v in lab[tr][mt]])
        yte = np.array([rm.get(v, 0) for v in lab[~tr][me]])
        t0 = time.time()
        best, runs = lib.probe(Ztr[mt], ytr, Zte[me], yte, seed_base=seed,
                               families=fams)
        mrows.append({**base, "target": tname, "target_type": "classification",
                      "n_classes": len(rm), "n_train": int(mt.sum()),
                      "n_test": int(me.sum()),
                      "chance_balanced": 1.0 / len(np.unique(yte)),
                      "best_family": best["family"], "best_cfg": best["cfg"],
                      "train_bal": best["train_bal"],
                      "test_bal": best["test_bal"], "sec": time.time() - t0})
        prows += [{**base, "target": tname, **r} for r in runs]


mrows, prows = [], []
for seed in SEEDS:

    measure("raw_U", U[tr], U[~tr], seed, mrows, prows)
    print(f"  raw_U seed {seed} done", flush=True)
    for meth, _ in JOBS:
        t0 = time.time()
        kw = {}
        if meth in lib.USES_B_LABELS:
            n = lib.N_ADV_ATTRS.get(meth, 1)
            kw["blabels"] = b_cols[:, :n] if n > 1 else b_cols[:, 0]
        f = lib.METHODS[meth](U[tr][ktr], A[tr][ktr], K, n_kept, seed=seed, **kw)
        Ztr, Zte = f(U[tr]), f(U[~tr])
        np.savez_compressed(
            ZDIR / f"dinol_{meth}_k{Ztr.shape[1]}_s{seed}.npz", Ztr=Ztr, Zte=Zte)
        measure(meth, Ztr, Zte, seed, mrows, prows)
        print(f"  {meth} seed {seed} done ({time.time()-t0:.0f} s)", flush=True)
        pd.DataFrame(mrows).to_csv(RES / "main_rafdb_dinol.csv", index=False)
        pd.DataFrame(prows).to_csv(RES / "probe_runs_rafdb_dinol.csv", index=False)

pd.DataFrame(mrows).to_csv(RES / "main_rafdb_dinol.csv", index=False)
pd.DataFrame(prows).to_csv(RES / "probe_runs_rafdb_dinol.csv", index=False)
print(f"\n[wrote] main_rafdb_dinol.csv {len(mrows)} rows · "
      f"probe_runs_rafdb_dinol.csv {len(prows)} rows")
