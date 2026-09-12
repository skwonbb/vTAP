"""Role and encoder experiments measured on Uhat."""
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

ROOT = SRC.parent
ZDIR = ROOT / "cache" / "Z"
SEEDS = (0, 1, 2)


OUT = None

df = pd.read_csv(ROOT / "data" / "face" / "raf_labels.csv")
tr = (df.split == "train").values
LAB = {"emotion": df.emotion.values - 1, "gender": df.gender.values,
       "race": df.race.values, "age": df.age.values}
EXCL = {"gender": df.gender.values == 2}


ROLES = [("A=race", "rafdb_clip.npy", "race", 0, "rafdb_rolerace", 2),
         ("A=age", "rafdb_clip.npy", "age", 2, "rafdb_roleage", 2),
         ("A=gender", "rafdb_clip.npy", "gender", 0, "rafdb_swap", 1)]


ENC = [("DINOv2-L", "rafdb_dinol.npy", "emotion", 2, "rafdb_dinol", 4)]


def targets(aname, keep, hold):
    out = [("A_kept", LAB[aname], keep)]
    if hold[tr].sum() > 20:
        out.append(("Aprime_heldout", LAB[aname], hold))
    for bn in LAB:
        if bn == aname:
            continue
        m = ~EXCL[bn] if bn in EXCL else np.ones(len(df), bool)
        out.append((f"B_{bn}", LAB[bn], m))
    return out


def run(jobs):
    global OUT
    have = set()
    if OUT.exists():
        old = pd.read_csv(OUT)
        have = set(map(tuple, old[["setting", "seed"]].drop_duplicates().values))
    rows = []
    for name, emb, aname, n_ho, zpre, K in jobs:
        U = np.load(ROOT / "cache" / "embeddings" / emb)
        A = LAB[aname]
        ok = ~EXCL[aname] if aname in EXCL else np.ones(len(U), bool)
        ho = run07.pick_holdout(A[ok], tr[ok], n_ho) if n_ho else []
        keep, hold = ok & ~np.isin(A, ho), ok & np.isin(A, ho)
        ktr, kte = keep[tr], keep[~tr]
        tg = targets(aname, keep, hold)
        print(f"\n=== {name}  ({emb}, d={U.shape[1]}, k={K}) "
              f"targets={[t[0] for t in tg]}", flush=True)

        for s in SEEDS:
            if (name, s) in have:
                print(f"  seed {s} already present, skipped", flush=True)
                continue
            z = np.load(ZDIR / f"{zpre}_lda_shrink_k{K}_s{s}.npz")
            Ztr, Zte = z["Ztr"], z["Zte"]

            Aug = np.hstack([Ztr[ktr], np.ones((ktr.sum(), 1))])
            M, *_ = np.linalg.lstsq(Aug, U[tr][ktr], rcond=None)
            Uh_tr = np.hstack([Ztr, np.ones((len(Ztr), 1))]) @ M
            Uh_te = np.hstack([Zte, np.ones((len(Zte), 1))]) @ M

            t0 = time.time()
            base = dict(setting=name, encoder=emb.replace("rafdb_", "")
                        .replace(".npy", ""), permitted=aname, D=U.shape[1],
                        k=K, seed=s)
            for tname, lab, msk in tg:
                a, b = msk[tr], msk[~tr]
                vals = sorted(np.unique(lab[tr][a]))
                rm = {v: i for i, v in enumerate(vals)}
                _, runs = lib.probe(Uh_tr[a], np.array([rm[v] for v in lab[tr][a]]),
                                    Uh_te[b],
                                    np.array([rm.get(v, 0) for v in lab[~tr][b]]),
                                    seed_base=s)
                rows += [{**base, "target": tname, **r} for r in runs]
            print(f"  seed {s} done ({time.time() - t0:.0f} s)", flush=True)
            pd.DataFrame(rows).to_csv(
                OUT, mode="a" if OUT.exists() else "w",
                header=not OUT.exists(), index=False)
            rows = []


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    OUT = ROOT / "results" / f"recon_appendix_{which}.csv"
    run({"roles": ROLES, "enc": ENC}.get(which, ROLES + ENC))
