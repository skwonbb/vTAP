"""Per-family probe values for the deployed transform."""
import importlib.util
import shutil
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
PROBE = RES / "probe_runs_rafdb.csv"
BK = RES / "backup_mlpdecoder"
SEED = 0                      
PROBE_SEEDS = (0, 1, 2)       
CONFIGS = [("lda_shrink", 4), ("vib_b1", 4), ("adv2attr_lam03", 4)]

cfg = run07.load_rafdb(encoder="clip")
U, tr, A = cfg["U"], cfg["tr"], cfg["A"]
holdout = run07.pick_holdout(A, tr, cfg["n_holdout"])
keep = ~np.isin(A, holdout)
ktr, kte = keep[tr], keep[~tr]


pr = pd.read_csv(PROBE)
stale = pr.note.astype(str) == "recon"
if stale.any():
    BK.mkdir(exist_ok=True)
    if not (BK / "probe_runs_rafdb.csv").exists():
        shutil.copy2(PROBE, BK / "probe_runs_rafdb.csv")
    print(f"removed {stale.sum()} stale recon rows")
    pr = pr[~stale]


have = set(map(tuple, pr[pr.note.astype(str) == "recon_linear"]
                [["method", "seed"]].drop_duplicates().values))

rows = []
for meth, k in CONFIGS:
    z = np.load(ZDIR / f"rafdb_{meth}_k{k}_s{SEED}.npz")
    Ztr, Zte = z["Ztr"], z["Zte"]
    Aug = np.hstack([Ztr[ktr], np.ones((ktr.sum(), 1))])
    M, *_ = np.linalg.lstsq(Aug, U[tr][ktr], rcond=None)
    Uh_tr = np.hstack([Ztr, np.ones((len(Ztr), 1))]) @ M
    Uh_te = np.hstack([Zte, np.ones((len(Zte), 1))]) @ M

    for s in PROBE_SEEDS:
        if (meth, s) in have:
            print(f"  {meth} k={k} seed {s}  already present, skipped", flush=True)
            continue
        t0 = time.time()
        base = dict(dataset="rafdb", encoder=cfg["encoder"], method=meth,
                    D=U.shape[1], k=k, transmitted_bytes=k * 4, seed=s,
                    note="recon_linear")
        for tname, mt, me in [("recon_A", ktr, kte),
                              ("recon_Aprime", ~ktr, ~kte)]:
            rm = {c: i for i, c in enumerate(sorted(np.unique(A[tr][mt])))}
            _, runs = lib.probe(Uh_tr[mt], np.array([rm[v] for v in A[tr][mt]]),
                                Uh_te[me],
                                np.array([rm.get(v, 0) for v in A[~tr][me]]),
                                seed_base=s)
            rows += [{**base, "target": tname, **r} for r in runs]
        for bname, bval in cfg["B"].items():
            ex = cfg["B_exclude"].get(bname)
            mt = ~ex[tr] if ex is not None else np.ones(tr.sum(), bool)
            me = ~ex[~tr] if ex is not None else np.ones((~tr).sum(), bool)
            _, runs = lib.probe(Uh_tr[mt], bval[tr][mt], Uh_te[me],
                                bval[~tr][me], seed_base=s)
            rows += [{**base, "target": f"recon_B_{bname}", **r} for r in runs]
        print(f"  {meth} k={k} seed {s}  done ({time.time()-t0:.0f} s)",
              flush=True)
        pd.concat([pr, pd.DataFrame(rows)], ignore_index=True).to_csv(
            PROBE, index=False)

pd.concat([pr, pd.DataFrame(rows)], ignore_index=True).to_csv(PROBE, index=False)
print(f"\n[wrote] {PROBE.name}   +{len(rows)} rows (per-family reconstruction)")
