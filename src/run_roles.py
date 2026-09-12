"""Role assignment experiment."""
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
SEEDS = (0, 1, 2)
FAM = run07.FAM_ATTACK

df = pd.read_csv(ROOT / "data" / "face" / "raf_labels.csv")
U = np.load(ROOT / "cache" / "embeddings" / "rafdb_clip.npy")
tr = (df.split == "train").values
D = U.shape[1]
LAB = {"emotion": df.emotion.values - 1, "gender": df.gender.values,
       "race": df.race.values, "age": df.age.values}
EXCL = {"gender": df.gender.values == 2}      


ROLES = [("race", 0), ("age", 2)]

rows, prows = [], []
for aname, n_ho in ROLES:
    A = LAB[aname]
    ok = ~EXCL[aname] if aname in EXCL else np.ones(len(U), bool)
    ho = run07.pick_holdout(A[ok], tr[ok], n_ho) if n_ho else []
    keep = ok & ~np.isin(A, ho)
    hold = ok & np.isin(A, ho)
    ktr, kte = keep[tr], keep[~tr]
    n_kept = len(np.unique(A[tr][ktr]))
    K = n_kept - 1
    print(f"\n[ ] A = {aname}   kept {n_kept} classes  k={K}  "
          f"held out {ho if n_ho else 'none'}", flush=True)

    tgts = [("A_kept", A, keep)]
    if hold[tr].sum() > 20:
        tgts.append(("Aprime_heldout", A, hold))
    for bn in LAB:
        if bn == aname:
            continue
        m = ~EXCL[bn] if bn in EXCL else np.ones(len(U), bool)
        tgts.append((f"B_{bn}", LAB[bn], m))

    for seed in SEEDS:

        for meth in ("raw_U", "lda_shrink"):
            t0 = time.time()
            if meth == "raw_U":
                Ztr, Zte, kk = U[tr], U[~tr], D
                use = [t for t in tgts if not t[0].startswith("B_")]
            else:
                f = lib.METHODS[meth](U[tr][ktr],
                                      np.array([{c: i for i, c in enumerate(
                                          sorted(np.unique(A[tr][ktr])))}[v]
                                          for v in A[tr][ktr]]),
                                      K, n_kept, seed=seed)
                Ztr, Zte = f(U[tr]), f(U[~tr])
                kk = Ztr.shape[1]
                np.savez_compressed(
                    ZDIR / f"rafdb_role{aname}_{meth}_k{kk}_s{seed}.npz",
                    Ztr=Ztr, Zte=Zte)
                use = tgts
            base = dict(dataset=f"rafdb_role_{aname}", encoder="clip-vit-l14",
                        method=meth, D=D, k=kk, transmitted_bytes=kk * 4,
                        seed=seed, note=f"A={aname}")
            for tname, lab, msk in use:
                mt, me = msk[tr], msk[~tr]
                rm = {c: i for i, c in enumerate(sorted(np.unique(lab[tr][mt])))}
                best, runs = lib.probe(
                    Ztr[mt], np.array([rm[v] for v in lab[tr][mt]]),
                    Zte[me], np.array([rm.get(v, 0) for v in lab[~tr][me]]),
                    seed_base=seed, families=FAM)
                rows.append({**base, "target": tname,
                             "target_type": "classification",
                             "n_classes": len(rm), "n_train": int(mt.sum()),
                             "n_test": int(me.sum()),
                             "chance_balanced": 1.0 / len(rm),
                             "best_family": best["family"],
                             "best_cfg": best["cfg"],
                             "test_bal": best["test_bal"]})
                prows += [{**base, "target": tname, **r} for r in runs]
            print(f"  {meth} seed {seed} ({time.time()-t0:.0f} s)", flush=True)
            pd.DataFrame(rows).to_csv(RES / "main_rafdb_roles.csv", index=False)
            pd.DataFrame(prows).to_csv(RES / "probe_runs_rafdb_roles.csv",
                                       index=False)

print(f"\n[done] {len(rows)} rows, {len(prows)} probes")
