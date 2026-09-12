"""Appendix runs for the role and encoder experiments."""
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
FAM = run07.FAM_ATTACK


def measure(Ztr, Zte, targets, tr, base, seed, mrows, prows, only=None):
    for tname, lab, msk in targets:
        if only and tname not in only:
            continue
        mt, me = msk[tr], msk[~tr]
        if mt.sum() <= 20 or me.sum() <= 20:
            continue
        rm = {c: i for i, c in enumerate(sorted(np.unique(lab[tr][mt])))}
        ytr = np.array([rm[v] for v in lab[tr][mt]])
        yte = np.array([rm.get(v, 0) for v in lab[~tr][me]])
        t1 = time.time()
        best, runs = lib.probe(Ztr[mt], ytr, Zte[me], yte, seed_base=seed,
                               families=FAM)
        mrows.append({**base, "target": tname, "target_type": "classification",
                      "n_classes": len(rm), "n_train": int(mt.sum()),
                      "n_test": int(me.sum()),
                      "chance_balanced": 1.0 / len(np.unique(yte)),
                      "best_family": best["family"], "best_cfg": best["cfg"],
                      "train_bal": best["train_bal"],
                      "test_bal": best["test_bal"], "sec": time.time() - t1})
        prows += [{**base, "target": tname, **r} for r in runs]


def targets_of(cfg, U, keep):
    o = [("A_kept", cfg["A"], keep)]
    for bn, bv in cfg["B"].items():
        ex = cfg["B_exclude"].get(bn)
        o.append((f"B_{bn}", bv,
                  ~ex if ex is not None else np.ones(len(U), bool)))
    return o


def merge(main_path, probe_path, mrows, prows, drop_method=None,
          drop_target=None):
    for path, new in ((main_path, mrows), (probe_path, prows)):
        old = pd.read_csv(path) if path.exists() else pd.DataFrame()
        if len(old) and drop_method:
            m = old.method.astype(str) == drop_method
            if drop_target:
                m &= old.target.astype(str).isin(drop_target)
            old = old[~m]
        pd.concat([old, pd.DataFrame(new)], ignore_index=True).to_csv(
            path, index=False)


print("[1] swap - LEACE with two attributes", flush=True)
_done1 = (pd.read_csv(RES / "probe_runs_rafdb_swap.csv").method.astype(str)
          == "leace2attr").any()
cfg = run07.load_rafdb(swap=True, encoder="clip")
U, tr, A = cfg["U"], cfg["tr"], cfg["A"]
D = U.shape[1]
ex_a = cfg.get("A_exclude")
keep = ~ex_a if ex_a is not None else np.ones(len(U), bool)
ktr = keep[tr]
n_kept = len(np.unique(A[tr][ktr]))
b_names = list(cfg["B"])
b_cols = np.stack([cfg["B"][n][tr][ktr] for n in b_names], 1)
TG = targets_of(cfg, U, keep)
print(f"  A = gender, {n_kept} classes   B order {b_names} -> two attributes {b_names[:2]}",
      flush=True)

if _done1:
    print("  already present - skipped\n", flush=True)
else:
    mrows, prows = [], []
    for seed in SEEDS:
        t0 = time.time()
        f = lib.METHODS["leace2attr"](U[tr][ktr], A[tr][ktr], D, n_kept,
                                      seed=seed, blabels=b_cols[:, :2])
        Ztr, Zte = f(U[tr]), f(U[~tr])
        np.savez_compressed(ZDIR / f"rafdb_swap_leace2attr_k{D}_s{seed}.npz",
                            Ztr=Ztr, Zte=Zte)
        measure(Ztr, Zte, TG, tr,
                dict(dataset="rafdb_swap", encoder=cfg["encoder"],
                     method="leace2attr", D=D, k=D, transmitted_bytes=D * 4,
                     seed=seed, note=""), seed, mrows, prows)
        print(f"  seed {seed} done ({time.time()-t0:.0f} s)", flush=True)
    merge(RES / "main_rafdb_swap.csv", RES / "probe_runs_rafdb_swap.csv",
          mrows, prows, drop_method="leace2attr")
    print(f"  [saved] +{len(mrows)} rows\n", flush=True)


print("[2] swap - add rff/tree/knn to A_kept", flush=True)
pr = pd.read_csv(RES / "probe_runs_rafdb_swap.csv")
need = pr[pr.target == "A_kept"][["method", "k", "seed"]].drop_duplicates()
have = set(map(tuple, pr[(pr.target == "A_kept") & pr.family.isin(
    ("rff", "tree", "knn"))][["method", "k", "seed"]].drop_duplicates().values))
todo = [t for t in map(tuple, need.values) if t not in have]
print(f"  {len(todo)} to do", flush=True)

rows, t0 = [], time.time()
for i, (meth, k, seed) in enumerate(todo, 1):
    if meth == "raw_U":
        Ztr, Zte = U[tr], U[~tr]
    else:
        f = ZDIR / f"rafdb_swap_{meth}_k{k}_s{seed}.npz"
        if not f.exists():
            print(f"  [skip] {meth} k={k} s={seed}", flush=True)
            continue
        z = np.load(f)
        Ztr, Zte = z["Ztr"], z["Zte"]
    mt, me = ktr, keep[~tr]
    rm = {c: i2 for i2, c in enumerate(sorted(np.unique(A[tr][mt])))}
    _, runs = lib.probe(Ztr[mt], np.array([rm[v] for v in A[tr][mt]]),
                        Zte[me], np.array([rm.get(v, 0) for v in A[~tr][me]]),
                        seed_base=seed, families=("rff", "tree", "knn"))
    rows += [{"dataset": "rafdb_swap", "encoder": cfg["encoder"],
              "method": meth, "D": D, "k": k, "transmitted_bytes": k * 4,
              "seed": seed, "target": "A_kept", **r} for r in runs]
    if i % 10 == 0 or i == len(todo):
        el = time.time() - t0
        print(f"  [{i}/{len(todo)}] {el/60:.0f} min, eta {el/i*len(todo)/60:.0f} min",
              flush=True)
        pd.concat([pr, pd.DataFrame(rows)], ignore_index=True).to_csv(
            RES / "probe_runs_rafdb_swap.csv", index=False)
pd.concat([pr, pd.DataFrame(rows)], ignore_index=True).to_csv(
    RES / "probe_runs_rafdb_swap.csv", index=False)
print(f"  [saved] +{len(rows)} rows\n", flush=True)


print("[3] DINOv2-large", flush=True)
emb = ROOT / "cache" / "embeddings" / "rafdb_dinol.npy"
if not emb.exists():
    sys.exit("rafdb_dinol.npy missing - run the embedding step first")

cfg = run07.load_rafdb(encoder="clip")
cfg["U"] = np.load(emb)
cfg["name"], cfg["encoder"] = "rafdb_dinol", "dinov2-l14"
U, tr, A = cfg["U"], cfg["tr"], cfg["A"]
D = U.shape[1]
holdout = run07.pick_holdout(A, tr, cfg["n_holdout"])
keep = ~np.isin(A, holdout)
ktr = keep[tr]
n_kept = len(np.unique(A[tr][ktr]))
K = n_kept - 1
b_cols = np.stack([cfg["B"][n][tr][ktr] for n in list(cfg["B"])], 1)
TG = targets_of(cfg, U, keep) + [("Aprime_heldout", A, ~keep)]


_rm = {c: i for i, c in enumerate(sorted(np.unique(A[tr][ktr])))}
ytr_k = np.array([_rm[v] for v in A[tr][ktr]])
print(f"  U {U.shape}   kept {n_kept} classes   k=C-1={K}", flush=True)

mrows, prows = [], []
for seed in SEEDS:
    t0 = time.time()
    measure(U[tr], U[~tr], TG, tr,
            dict(dataset="rafdb_dinol", encoder=cfg["encoder"], method="raw_U",
                 D=D, k=D, transmitted_bytes=D * 4, seed=seed, note=""),
            seed, mrows, prows)
    print(f"  raw_U seed {seed} ({time.time()-t0:.0f} s)", flush=True)
    for meth in ("lda_shrink", "vib_b1", "adv2attr_lam03", "leace2attr"):
        t0 = time.time()
        kw = {}
        if meth in lib.USES_B_LABELS:
            n = lib.N_ADV_ATTRS.get(meth, 1)
            kw["blabels"] = b_cols[:, :n] if n > 1 else b_cols[:, 0]
        kk = D if meth.startswith("leace") else K
        f = lib.METHODS[meth](U[tr][ktr], ytr_k, kk, n_kept, seed=seed, **kw)
        Ztr, Zte = f(U[tr]), f(U[~tr])
        np.savez_compressed(
            ZDIR / f"rafdb_dinol_{meth}_k{Ztr.shape[1]}_s{seed}.npz",
            Ztr=Ztr, Zte=Zte)
        measure(Ztr, Zte, TG, tr,
                dict(dataset="rafdb_dinol", encoder=cfg["encoder"],
                     method=meth, D=D, k=Ztr.shape[1],
                     transmitted_bytes=Ztr.shape[1] * 4, seed=seed, note=""),
                seed, mrows, prows)
        print(f"  {meth} seed {seed} ({time.time()-t0:.0f} s)", flush=True)
        pd.DataFrame(mrows).to_csv(RES / "main_rafdb_dinol.csv", index=False)
        pd.DataFrame(prows).to_csv(RES / "probe_runs_rafdb_dinol.csv",
                                   index=False)
print(f"\n[done] dinol {len(mrows)} rows")
