"""Probe accuracy under fixed-point arithmetic over f."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import lib  

ROOT = SRC.parent
OUT = ROOT / "results" / "quant_sweep.csv"
ERR_OUT = ROOT / "results" / "quant_error.csv"
PARAM_DIR = ROOT / "results" / "tap_params"      
SEEDS = (0, 1, 2)


FBITS_ERROR = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24)
FBITS_PROBE = (16,)
K = 4                                            
N_HOLDOUT = 2

df = pd.read_csv(ROOT / "data" / "face" / "raf_labels.csv")
tr = (df.split == "train").values
LAB = {"emotion": df.emotion.values - 1, "gender": df.gender.values,
       "race": df.race.values, "age": df.age.values}
EXCL = {"gender": df.gender.values == 2}


def pick_holdout(A, n):
    order = list(pd.Series(A[tr]).value_counts().index)
    pos = np.linspace(0, len(order) - 1, n + 2)[1:-1]
    return sorted(int(order[int(round(p))]) for p in pos)


def fit_tap(U, y):
    lda = LinearDiscriminantAnalysis(
        n_components=K, solver="eigen", shrinkage="auto").fit(U, y)
    W = lda.scalings_[:, :K]
    Z = U @ W
    aug = np.hstack([Z, np.ones((len(Z), 1))])
    sol, *_ = np.linalg.lstsq(aug, U, rcond=None)
    return W, sol[:-1], sol[-1:]


def rescale(acc, f):
    return (acc + (1 << (f - 1))) >> f


def apply_tap(U, W, M, b, f=None):
    if f is None:
        return (U @ W) @ M + b
    S = 1 << f
    Ut = np.rint(U * S).astype(np.int64)
    Wt = np.rint(W * S).astype(np.int64)
    Mt = np.rint(M * S).astype(np.int64)
    bt = np.rint(b * S).astype(np.int64)
    over = max(np.abs(Ut @ Wt).max(), 1)
    Zt = rescale(Ut @ Wt, f)
    acc = Zt @ Mt
    if max(over, np.abs(acc).max()) > (1 << 62):
        raise SystemExit(f"f={f} : int64 accumulator overflows")
    return (rescale(acc, f) + bt).astype(np.float64) / S


def targets(keep, hold):
    out = [("A_kept", LAB["emotion"], keep),
           ("Aprime_heldout", LAB["emotion"], hold)]
    for bn in ("gender", "race", "age"):
        m = ~EXCL[bn] if bn in EXCL else np.ones(len(df), bool)
        out.append((f"B_{bn}", LAB[bn], m))
    return out


def main(fbits):
    U = np.load(ROOT / "cache" / "embeddings" / "rafdb_clip.npy").astype(np.float64)
    A = LAB["emotion"]
    ho = pick_holdout(A, N_HOLDOUT)
    keep, hold = ~np.isin(A, ho), np.isin(A, ho)
    ktr = keep & tr


    W, M, b = fit_tap(U[ktr], A[ktr])
    PARAM_DIR.mkdir(parents=True, exist_ok=True)
    for nm, arr in (("W", W), ("M", M), ("b", b)):
        np.save(PARAM_DIR / f"{nm}.npy", arr)
    print(f"holdout={ho}  keep(train/test)={int(ktr.sum())}/{int((keep & ~tr).sum())}"
          f"  W{W.shape} M{M.shape} b{b.shape}  -> {PARAM_DIR}", flush=True)

    have = set()
    if OUT.exists():
        old = pd.read_csv(OUT)
        have = set(map(tuple, old[["f", "seed"]].drop_duplicates().values))

    Uh_ref = apply_tap(U, W, M, b, None)
    tg = targets(keep, hold)


    nrm = np.linalg.norm(Uh_ref)
    erows = []
    for f in FBITS_ERROR:
        dv = apply_tap(U, W, M, b, f) - Uh_ref
        q = dv + Uh_ref
        erows.append(dict(
            f=f, ulp=2.0 ** -f, abs_max=float(np.abs(dv).max()),
            rel_l2=float(np.linalg.norm(dv) / nrm),
            one_minus_cos=float(1 - (q * Uh_ref).sum()
                                / (np.linalg.norm(q) * nrm))))
    pd.DataFrame(erows).to_csv(ERR_OUT, index=False)
    print(f"[wrote] {ERR_OUT}  ({len(erows)} rows)")
    print(pd.DataFrame(erows).to_string(index=False, float_format="%.3e"),
          flush=True)


    for f in fbits:
        tag = -1 if f is None else f          
        Uh = apply_tap(U, W, M, b, f)
        err = float(np.abs(Uh - Uh_ref).max())
        rel = float(np.linalg.norm(Uh - Uh_ref) / np.linalg.norm(Uh_ref))
        print(f"\n=== f={tag}  max|Uhat-Uhat_real|={err:.3e}  rel. error={rel:.3e}", flush=True)

        for s in SEEDS:
            if (tag, s) in have:
                print(f"  seed {s} already present, skipped", flush=True)
                continue
            t0 = time.time()
            rows = []
            for tname, lab, msk in tg:
                a, b_ = msk & tr, msk & ~tr
                vals = sorted(np.unique(lab[a]))
                rm = {v: i for i, v in enumerate(vals)}
                _, runs = lib.probe(
                    Uh[a], np.array([rm[v] for v in lab[a]]),
                    Uh[b_], np.array([rm.get(v, 0) for v in lab[b_]]),
                    seed_base=s)
                rows += [dict(f=tag, seed=s, target=tname, n_classes=len(vals),
                              chance_balanced=1 / len(vals),
                              abs_err=err, rel_err=rel, **r) for r in runs]
            pd.DataFrame(rows).to_csv(OUT, mode="a" if OUT.exists() else "w",
                                      header=not OUT.exists(), index=False)
            print(f"  seed {s} done ({time.time() - t0:.0f} s, {len(rows)} rows)",
                  flush=True)


if __name__ == "__main__":
    args = [int(x) for x in sys.argv[1:]]
    main(list(args) if args else list(FBITS_PROBE))
