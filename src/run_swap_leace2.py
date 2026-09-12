"""Role swap with the two-attribute LEACE baseline."""
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
METH, SEEDS = "leace2attr", (0, 1, 2)

cfg = run07.load_rafdb(swap=True, encoder="clip")
U, tr, A = cfg["U"], cfg["tr"], cfg["A"]
D = U.shape[1]
ex_a = cfg.get("A_exclude")
keep = ~ex_a if ex_a is not None else np.ones(len(U), bool)   
ktr = keep[tr]
n_kept = len(np.unique(A[tr][ktr]))
print(f"A = gender, {n_kept} classes   kept train {ktr.sum():,}", flush=True)

b_names = list(cfg["B"])
b_cols = np.stack([cfg["B"][n][tr][ktr] for n in b_names], 1)
print("B order:", b_names, "-> two attributes =", b_names[:2], flush=True)

TARGETS = [("A_kept", A, keep)]
for bn, bv in cfg["B"].items():
    ex = cfg["B_exclude"].get(bn)
    TARGETS.append((f"B_{bn}", bv,
                    ~ex if ex is not None else np.ones(len(U), bool)))

mrows, prows = [], []
for seed in SEEDS:
    t0 = time.time()
    f = lib.METHODS[METH](U[tr][ktr], A[tr][ktr], D, n_kept, seed=seed,
                          blabels=b_cols[:, :2])
    Ztr, Zte = f(U[tr]), f(U[~tr])
    np.savez_compressed(ZDIR / f"swap_{METH}_k{Ztr.shape[1]}_s{seed}.npz",
                        Ztr=Ztr, Zte=Zte)
    base = dict(dataset="rafdb_swap", encoder=cfg["encoder"], method=METH,
                D=D, k=Ztr.shape[1], transmitted_bytes=Ztr.shape[1] * 4,
                seed=seed, note="")
    for tname, lab, msk in TARGETS:
        mt, me = msk[tr], msk[~tr]
        rm = {c: i for i, c in enumerate(sorted(np.unique(lab[tr][mt])))}
        ytr = np.array([rm[v] for v in lab[tr][mt]])
        yte = np.array([rm.get(v, 0) for v in lab[~tr][me]])
        t1 = time.time()
        best, runs = lib.probe(Ztr[mt], ytr, Zte[me], yte, seed_base=seed,
                               families=run07.FAM_ATTACK)
        mrows.append({**base, "target": tname, "target_type": "classification",
                      "n_classes": len(rm), "n_train": int(mt.sum()),
                      "n_test": int(me.sum()),
                      "chance_balanced": 1.0 / len(np.unique(yte)),
                      "best_family": best["family"], "best_cfg": best["cfg"],
                      "train_bal": best["train_bal"],
                      "test_bal": best["test_bal"], "sec": time.time() - t1})
        prows += [{**base, "target": tname, **r} for r in runs]
    print(f"  seed {seed} done ({time.time()-t0:.0f} s)", flush=True)

    for path, new in ((RES / "main_rafdb_swap.csv", mrows),
                      (RES / "probe_runs_rafdb_swap.csv", prows)):
        old = pd.read_csv(path)
        old = old[old.method.astype(str) != METH]
        pd.concat([old, pd.DataFrame(new)], ignore_index=True).to_csv(
            path, index=False)

print(f"\n[wrote] main_rafdb_swap.csv +{len(mrows)} rows · "
      f"probe_runs_rafdb_swap.csv +{len(prows)} rows")
