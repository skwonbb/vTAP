"""Receiver-model compatibility across five probe families."""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import lib  

spec = importlib.util.spec_from_file_location("run07", SRC / "07_run.py")
run07 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run07)

ROOT, RES = SRC.parent, SRC.parent / "results"
ZDIR = ROOT / "cache" / "Z"
SEED = 0                       
SERVER_SEEDS = (0, 1, 2)       
METHODS = [("lda_shrink", 4), ("vib_b1", 4), ("adv2attr_lam03", 4),
           ("leace2attr", 768)]

cfg = run07.load_rafdb(encoder="clip")
U, tr, A = cfg["U"], cfg["tr"], cfg["A"]
holdout = run07.pick_holdout(A, tr, cfg["n_holdout"])
keep = ~np.isin(A, holdout)
ktr, kte = keep[tr], keep[~tr]


UH = {}
for meth, k in METHODS:
    z = np.load(ZDIR / f"rafdb_{meth}_k{k}_s{SEED}.npz")
    Ztr, Zte = z["Ztr"], z["Zte"]
    if Ztr.shape[1] == U.shape[1]:
        UH[meth] = Zte
        continue
    Aug = np.hstack([Ztr[ktr], np.ones((ktr.sum(), 1))])
    M, *_ = np.linalg.lstsq(Aug, U[tr][ktr], rcond=None)
    UH[meth] = np.hstack([Zte, np.ones((len(Zte), 1))]) @ M
print("Uhat ready:", ", ".join(UH))


TARGETS = [("compat_A", A, keep)]
for bname, bval in cfg["B"].items():
    ex = cfg["B_exclude"].get(bname)
    TARGETS.append((f"compat_B_{bname}", bval,
                    ~ex if ex is not None else np.ones(len(U), bool)))

rows = []
for tname, lab, msk in TARGETS:
    mt, me = msk[tr], msk[~tr]
    rm = {c: i for i, c in enumerate(sorted(np.unique(lab[msk])))}
    ytr = np.array([rm[v] for v in lab[tr][mt]])
    yte = np.array([rm.get(v, 0) for v in lab[~tr][me]])
    Xtr = U[tr][mt]
    for s in SERVER_SEEDS:
        t0 = time.time()

        preds = lib.probe_fitted(Xtr, ytr, seed_base=s)
        for fam, cfgname, predict in preds:
            base = dict(dataset="rafdb", target=tname, target_type="compat",
                        family=fam, cfg=cfgname, seed=s, note="compat_5fam")
            on_u = balanced_accuracy_score(yte, predict(U[~tr][me]))
            for meth, k in METHODS:
                on_uh = balanced_accuracy_score(yte, predict(UH[meth][me]))
                rows.append({**base, "method": meth, "k": k,
                             "compat_on_U": on_u, "compat_on_Uhat": on_uh,
                             "test_bal": on_uh})
        print(f"  {tname:18s} seed {s}  {len(preds)} configs  "
              f"({time.time()-t0:.0f} s)", flush=True)

out = RES / "compat_5fam.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(f"\n[wrote] {out.name}   {len(rows)} rows")
