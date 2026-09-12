"""Dense sweep of the learned baselines over their handle."""
import argparse, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import lib  

ROOT = SRC.parent
OUT = ROOT / "results" / "sweep_dense.csv"
HOLDOUT = (2, 4)                       
SEED = 0

KS = [2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
BETAS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
LAMS = [0.3, 0.5, 1.0, 2.0, 3.0, 5.0]

df = pd.read_csv(ROOT / "data/face/raf_labels.csv")
U = np.load(ROOT / "cache/embeddings/rafdb_clip.npy")
tr = (df.split == "train").values
emo = df.emotion.values - 1
keep = ~np.isin(emo, HOLDOUT)
hold = ~keep
cls = sorted(np.unique(emo[tr & keep]))
Utr_k = U[tr & keep]
ytr_k = np.array([cls.index(v) for v in emo[tr & keep]])
b_cols = np.stack([df.gender.values[tr & keep], df.race.values[tr & keep]], 1)

TASKS = [("A_kept", emo, keep, 0.2),
         ("Aprime_heldout", emo, hold, 0.5),
         ("B_gender", df.gender.values, df.gender.values != 2, 0.5),
         ("B_race", df.race.values, np.ones(len(df), bool), 1 / 3),
         ("B_age", df.age.values, np.ones(len(df), bool), 0.2)]


def evaluate(f):
    out = {}
    for name, lab, msk, _ in TASKS:
        a, b = tr & msk, (~tr) & msk
        vals = sorted(np.unique(lab[msk]))
        rm = {v: i for i, v in enumerate(vals)}
        best, _ = lib.probe(f(U[a]), np.array([rm[v] for v in lab[a]]),
                            f(U[b]), np.array([rm[v] for v in lab[b]]),
                            seed_base=SEED)
        out[name] = best["test_bal"]
    return out


def jobs():
    for b in BETAS:
        for k in KS:
            yield ("vib", b, k)
    for l in LAMS:
        for k in KS:
            yield ("adv", l, k)


def build(kind, p, k):
    if kind == "vib":
        return lib.make_vib(Utr_k, ytr_k, k, len(cls), seed=SEED, beta=p)
    return lib.make_adversarial(Utr_k, ytr_k, k, len(cls), seed=SEED,
                                blabels=b_cols, lam=p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", action="store_true", help="time a single configuration")
    args = ap.parse_args()

    J = list(jobs())
    if args.time:
        J = [("vib", 0.01, 8)]
    rows, t0 = [], time.time()
    for i, (kind, p, k) in enumerate(J, 1):
        t = time.time()
        try:
            r = evaluate(build(kind, p, k))
        except Exception as e:
            print(f"  [{i}/{len(J)}] failed {kind} p={p} k={k}: {e!r}")
            continue
        r.update(kind=kind, param=p, k=k, seed=SEED, sec=round(time.time() - t, 1))
        rows.append(r)
        el = time.time() - t0
        print(f"  [{i}/{len(J)}] {kind:3s} p={p:<6g} k={k:<4d} "
              f"{r['sec']:5.1f}s   elapsed {el/60:5.1f} min   "
              f"eta {el/i*(len(J)-i)/60:5.1f} min", flush=True)
        if not args.time and i % 10 == 0:
            pd.DataFrame(rows).to_csv(OUT, index=False)

    if rows and not args.time:
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"\n[wrote] {OUT}   {len(rows)} configurations")
    elif args.time:
        print(f"\n{rows[0]['sec']:.1f} s per configuration; {len(list(jobs()))} configurations would take "
              f"{rows[0]['sec'] * len(list(jobs())) / 60:.0f} min")
