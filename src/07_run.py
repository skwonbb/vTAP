"""Train the methods and run the probes."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  


FAM_ATTACK = ("lr", "mlp", "rff", "tree", "knn")
FAM_UTILITY = ("lr", "mlp")

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
ZDIR = ROOT / "cache" / "Z"
RES.mkdir(exist_ok=True)
ZDIR.mkdir(parents=True, exist_ok=True)

SEEDS_STOCHASTIC = (0, 1, 2)


SEEDS_DETERMINISTIC = (0, 1, 2)


def load_rafdb(swap=False, encoder="clip", reduced=False):
    tag = {"clip": "rafdb_clip", "dino": "rafdb_dino"}[encoder]
    encname = {"clip": "clip-vit-l14", "dino": "dinov2-base"}[encoder]
    U = np.load(ROOT / "cache" / "embeddings" / f"{tag}.npy")
    df = pd.read_csv(ROOT / "data" / "face" / "raf_labels.csv")
    tr = (df.split == "train").values
    lm = df[[f"lm{i}" for i in range(10)]].values.astype(np.float32)

    ks = [4, 16, 64] if reduced else [2, 4, 8, 16, 64, 256]


    only = (["lda", "lda_shrink", "bottleneck", "random", "pca", "inlp", "leace",
             *lib.VIB_VARIANTS, *lib.ADV_VARIANTS] if reduced else None)
    if swap:
        return dict(
            name="rafdb_swap", encoder=encname, U=U, tr=tr, only=only,
            A=df.gender.values,                        
            A_exclude=(df.gender.values == 2),
            B={"emotion": df.emotion.values - 1, "race": df.race.values,
               "age": df.age.values},
            B_exclude={}, aux={"landmark": lm}, n_holdout=0, ks=ks,
        )
    return dict(
        name="rafdb" if encoder == "clip" else f"rafdb_{encoder}",
        encoder=encname, U=U, tr=tr, only=only,
        A=df.emotion.values - 1,                       
        B={"gender": df.gender.values, "race": df.race.values, "age": df.age.values},
        B_exclude={"gender": df.gender.values == 2},   
        aux={"landmark": lm},
        n_holdout=2, ks=ks,
    )


def load_bios(scrubbed=False):
    E = ROOT / "cache" / "embeddings"
    P = ROOT / "data" / "text" / "bios"
    tag = "scrub" if scrubbed else "bert"
    suffix = "_scrubbed" if scrubbed else ""
    Us, As, Gs, trs = [], [], [], []
    for split in ("train", "test"):
        u = np.load(E / f"bios_{tag}_{split}.npy")
        d = pd.read_parquet(P / f"{split}{suffix}.parquet")
        Us.append(u); As.append(d.profession.values); Gs.append(d.gender.values)
        trs.append(np.full(len(u), split == "train"))
    return dict(
        name="bios_scrub" if scrubbed else "bios",
        encoder="bert-base-cls" + ("-scrubbed" if scrubbed else ""),
        U=np.concatenate(Us), tr=np.concatenate(trs),
        A=np.concatenate(As), B={"gender": np.concatenate(Gs)},
        B_exclude={}, aux={},
        n_holdout=8,
        ks=[5, 10, 19, 32, 64, 128, 256, 512, 768],
    )


def pick_holdout(A, tr, n_holdout):
    if n_holdout == 0:
        return []
    vc = pd.Series(A[tr]).value_counts()          
    order = list(vc.index)
    pos = np.linspace(0, len(order) - 1, n_holdout + 2)[1:-1]
    return sorted(int(order[int(round(p))]) for p in pos)


def run_reconstruction(cfg, Ztr, Zte, base, main_rows, probe_rows):
    U, tr, A = cfg["U"], cfg["tr"], cfg["A"]
    met, Uhat_tr, Uhat_te = lib.reconstruct(Ztr, U[tr], Zte, U[~tr])
    main_rows.append({**base, "target": "recon_quality", "target_type": "regression",
                      "test_acc": met["recon_cosine"], "test_bal": met["recon_cosine"],
                      **met})


    ktr, kte = cfg["_keep_tr"], cfg["_keep_te"]
    a_tr0, a_te0 = A[tr][ktr], A[~tr][kte]
    rm = {c: i for i, c in enumerate(sorted(np.unique(a_tr0)))}
    for tname, ztr, ytr_, zte, yte_ in [
        ("recon_A", Uhat_tr[ktr], np.array([rm[v] for v in a_tr0]),
         Uhat_te[kte], np.array([rm[v] for v in a_te0])),
    ] + ([("recon_Aprime", Uhat_tr[cfg["_ho_tr"]], A[tr][cfg["_ho_tr"]],
           Uhat_te[cfg["_ho_te"]], A[~tr][cfg["_ho_te"]])]
         if cfg["_ho_tr"].sum() > 20 else []):
        if tname == "recon_Aprime":
            r2 = {c: i for i, c in enumerate(sorted(np.unique(ytr_)))}
            ytr_ = np.array([r2[v] for v in ytr_])
            yte_ = np.array([r2.get(v, 0) for v in yte_])
        best, runs = lib.probe(ztr, ytr_, zte, yte_, seed_base=0,
                               families=FAM_UTILITY)
        main_rows.append({**base, "target": tname, "target_type": "classification",
                          "train_bal": best["train_bal"], "test_bal": best["test_bal"],
                          "best_family": best["family"], **met})
    for bname, bval in cfg["B"].items():
        ex = cfg["B_exclude"].get(bname)
        mtr = ~ex[tr] if ex is not None else np.ones(tr.sum(), bool)
        mte = ~ex[~tr] if ex is not None else np.ones((~tr).sum(), bool)
        best, runs = lib.probe(Uhat_tr[mtr], bval[tr][mtr],
                               Uhat_te[mte], bval[~tr][mte], seed_base=0)
        main_rows.append({**base, "target": f"recon_B_{bname}",
                          "target_type": "classification",
                          "train_bal": best["train_bal"], "test_bal": best["test_bal"],
                          "train_acc": best["train_acc"], "test_acc": best["test_acc"],
                          "best_family": best["family"], **met})
        for r in runs:
            probe_rows.append({**base, "target": f"recon_B_{bname}", **r})

    return Uhat_tr, Uhat_te


def run_compat(cfg, Uhat_te, keep_te, base, main_rows):
    U, tr, A = cfg["U"], cfg["tr"], cfg["A"]
    a_tr0, a_te0 = A[tr][cfg["_keep_tr"]], A[~tr][keep_te]
    rm = {c: i for i, c in enumerate(sorted(np.unique(a_tr0)))}
    ytr = np.array([rm[v] for v in a_tr0])
    yte = np.array([rm[v] for v in a_te0])

    targets = [("A", U[tr][cfg["_keep_tr"]], ytr, U[~tr][keep_te],
                Uhat_te[keep_te], yte)]
    for bname, bval in cfg["B"].items():
        ex = cfg["B_exclude"].get(bname)
        mtr = ~ex[tr] if ex is not None else np.ones(tr.sum(), bool)
        mte = ~ex[~tr] if ex is not None else np.ones((~tr).sum(), bool)
        targets.append((f"B_{bname}", U[tr][mtr], bval[tr][mtr],
                        U[~tr][mte], Uhat_te[mte], bval[~tr][mte]))

    for tname, Xtr, ytr_, Xte_orig, Xte_hat, yte_ in targets:
        nc = int(max(ytr_.max(), yte_.max())) + 1
        sc = StandardScaler().fit(Xtr)          
        for arch, fit in (("linear", lib._fit_linear),
                          ("mlp256", lambda *a, **k: lib._fit_mlp(*a, hidden=256, **k)),
                          ("mlp1024", lambda *a, **k: lib._fit_mlp(*a, hidden=1024, **k))):
            m = fit(sc.transform(Xtr), ytr_, nc, seed=0)
            acc_orig = balanced_accuracy_score(yte_, lib._pred(m, sc.transform(Xte_orig)))
            acc_hat = balanced_accuracy_score(yte_, lib._pred(m, sc.transform(Xte_hat)))
            main_rows.append({**base, "target": f"compat_{tname}",
                              "target_type": "compat", "best_cfg": arch,
                              "test_bal": acc_hat, "compat_on_U": acc_orig,
                              "compat_on_Uhat": acc_hat,
                              "compat_drop": acc_orig - acc_hat})


def evaluate(cfg, Ztr, Zte, method, k, D, seed, keep_mask_tr, keep_mask_te,
             ho_mask_tr, ho_mask_te, main_rows, probe_rows, note=""):
    base = dict(
        dataset=cfg["name"], encoder=cfg["encoder"], method=method,
        D=D, k=k, transmitted_bytes=k * 4, seed=seed, note=note,
    )
    A, tr = cfg["A"], cfg["tr"]

    targets = []

    a_tr0, a_te0 = A[tr][keep_mask_tr], A[~tr][keep_mask_te]
    rm_a = {c: i for i, c in enumerate(sorted(np.unique(a_tr0)))}
    targets.append(("A_kept", "classification",
                    Ztr[keep_mask_tr], np.array([rm_a[v] for v in a_tr0]),
                    Zte[keep_mask_te], np.array([rm_a[v] for v in a_te0])))

    if ho_mask_tr.sum() > 20 and ho_mask_te.sum() > 20:
        ytr_h, yte_h = A[tr][ho_mask_tr], A[~tr][ho_mask_te]
        remap = {c: i for i, c in enumerate(sorted(np.unique(ytr_h)))}
        targets.append(("Aprime_heldout", "classification", Ztr[ho_mask_tr],
                        np.array([remap[v] for v in ytr_h]),
                        Zte[ho_mask_te],
                        np.array([remap.get(v, 0) for v in yte_h])))

    for bname, bval in cfg["B"].items():
        ex = cfg["B_exclude"].get(bname)
        mtr = ~ex[tr] if ex is not None else np.ones(tr.sum(), bool)
        mte = ~ex[~tr] if ex is not None else np.ones((~tr).sum(), bool)
        targets.append((f"B_{bname}", "classification", Ztr[mtr], bval[tr][mtr],
                        Zte[mte], bval[~tr][mte]))

    for tname, ttype, ztr, ytr_, zte, yte_ in targets:
        t0 = time.time()
        fams = FAM_ATTACK if tname.startswith("B_") else FAM_UTILITY
        best, runs = lib.probe(ztr, ytr_, zte, yte_, seed_base=seed, families=fams)
        chance = pd.Series(yte_).value_counts(normalize=True).iloc[0]
        main_rows.append({
            **base, "target": tname, "target_type": ttype,
            "n_classes": int(len(np.unique(ytr_))),
            "n_train": len(ztr), "n_test": len(zte),
            "chance": chance, "chance_balanced": 1.0 / len(np.unique(yte_)),
            "best_family": best["family"], "best_cfg": best["cfg"],
            "train_acc": best["train_acc"], "train_bal": best["train_bal"],
            "test_acc": best["test_acc"], "test_bal": best["test_bal"],
            "sec": time.time() - t0,
        })
        for r in runs:
            probe_rows.append({**base, "target": tname, **r})


    if "landmark" in cfg["aux"]:
        lm = cfg["aux"]["landmark"]
        t0 = time.time()
        reg = lib.probe_regression(Ztr, lm[tr], Zte, lm[~tr], seed=seed)
        main_rows.append({
            **base, "target": "Aprime_landmark", "target_type": "regression",
            "n_classes": np.nan, "n_train": len(Ztr), "n_test": len(Zte),
            "chance": np.nan, "chance_balanced": np.nan,
            "best_family": "mlp_reg", "best_cfg": "-",
            "train_acc": np.nan, "train_bal": np.nan,
            "test_acc": reg["landmark_r2"], "test_bal": reg["landmark_r2"],
            "landmark_mse": reg["landmark_mse"], "landmark_r2": reg["landmark_r2"],
            "sec": time.time() - t0,
        })


def main(which, baseline_only=False):
    if which.startswith("rafdb"):
        cfg = load_rafdb(swap=(which == "rafdb_swap"),
                         encoder="dino" if "dino" in which else "clip",
                         reduced=(which != "rafdb"))   
    else:
        cfg = load_bios(scrubbed=(which == "bios_scrub"))
    U, tr, A = cfg["U"], cfg["tr"], cfg["A"]
    d = U.shape[1]

    holdout = pick_holdout(A, tr, cfg["n_holdout"])


    a_ex = cfg.get("A_exclude")
    ok_tr = ~a_ex[tr] if a_ex is not None else np.ones(tr.sum(), bool)
    ok_te = ~a_ex[~tr] if a_ex is not None else np.ones((~tr).sum(), bool)
    keep_tr = ~np.isin(A[tr], holdout) & ok_tr
    keep_te = ~np.isin(A[~tr], holdout) & ok_te
    ho_tr = np.isin(A[tr], holdout) & ok_tr
    ho_te = np.isin(A[~tr], holdout) & ok_te
    n_kept = len(np.unique(A[tr][keep_tr]))
    cfg["ks"] = sorted(set(cfg["ks"]) | {max(1, n_kept - 1)})

    meta = dict(dataset=cfg["name"], encoder=cfg["encoder"], d=d,
                n_train=int(tr.sum()), n_test=int((~tr).sum()),
                holdout_classes=holdout, n_kept_classes=int(n_kept),
                ks=cfg["ks"], floor_dim=int(n_kept - 1))
    (RES / f"config_{cfg['name']}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)

    Utr_all, Ute_all = U[tr], U[~tr]


    Utr_k = Utr_all[keep_tr]
    kept_codes = sorted(np.unique(A[tr][keep_tr]))
    kept_remap = {c: i for i, c in enumerate(kept_codes)}
    ytr_k = np.array([kept_remap[v] for v in A[tr][keep_tr]])
    meta["kept_class_codes"] = [int(c) for c in kept_codes]
    (RES / f"config_{cfg['name']}.json").write_text(json.dumps(meta, indent=2))

    main_rows, probe_rows = [], []


    done_set = set()
    mpath, ppath = RES / f"main_{cfg['name']}.csv", RES / f"probe_runs_{cfg['name']}.csv"
    if mpath.exists():
        prev = pd.read_csv(mpath)
        main_rows = prev.to_dict("records")
        if ppath.exists():
            probe_rows = pd.read_csv(ppath).to_dict("records")
        done_set = {(r.method, int(r.k), int(r.seed))
                    for r in prev[["method", "k", "seed"]].dropna().itertuples()}
        print(f"[resume] {len(done_set)} configurations already present, skipped", flush=True)


    if ("raw_U", d, 0) not in done_set:
        print("\n[baseline] raw U", flush=True)
        evaluate(cfg, Utr_all, Ute_all, "raw_U", d, d, 0,
                 keep_tr, keep_te, ho_tr, ho_te, main_rows, probe_rows,
                 note="baseline")
        _flush(cfg, main_rows, probe_rows)

    if baseline_only:
        print("\n" + "=" * 72)
        print("BASELINE - how well each attribute reads from raw U")
        print("=" * 72)
        print(f"{'target':<20}{'n_cls':>6}{'chance':>9}{'test_acc':>10}"
              f"{'test_bal':>10}{'train_acc':>11}  best")
        print("-" * 72)
        for r in main_rows:
            if r.get("target_type") == "regression":
                print(f"{r['target']:<20}{'reg':>6}{'-':>9}"
                      f"{r['test_acc']:>10.4f}{'(R2)':>10}{'-':>11}  {r['best_family']}")
            else:
                print(f"{r['target']:<20}{r['n_classes']:>6}{r['chance']:>9.4f}"
                      f"{r['test_acc']:>10.4f}{r['test_bal']:>10.4f}"
                      f"{r['train_acc']:>11.4f}  {r['best_family']}/{r['best_cfg']}")
        print("-" * 72)
        print(f"held-out classes: {holdout}  (used as A-prime, not used to build Z)")
        return


    cfg["_keep_tr"], cfg["_keep_te"] = keep_tr, keep_te
    cfg["_ho_tr"], cfg["_ho_te"] = ho_tr, ho_te

    jobs = []
    jobs.append(("logits", n_kept, SEEDS_STOCHASTIC))


    jobs.append(("leace", d, SEEDS_DETERMINISTIC))         
    if len(cfg["B"]) >= 2:
        jobs.append(("leace2attr", d, SEEDS_DETERMINISTIC))  
    for k in cfg["ks"]:
        if k <= n_kept - 1:
            jobs.append(("lda", k, SEEDS_DETERMINISTIC))
            jobs.append(("lda_shrink", k, SEEDS_DETERMINISTIC))   
            jobs.append(("lda_rff", k, SEEDS_STOCHASTIC))
        jobs.append(("bottleneck", k, SEEDS_STOCHASTIC))


        for vname in lib.VIB_VARIANTS:              
            jobs.append((vname, k, SEEDS_STOCHASTIC))
        for advname in lib.ADV_VARIANTS:            
            jobs.append((advname, k, SEEDS_STOCHASTIC))
        jobs.append(("random", k, SEEDS_STOCHASTIC))
        jobs.append(("pca", k, SEEDS_DETERMINISTIC))


    b_names = list(cfg["B"])
    b_cols = np.stack([cfg["B"][n][tr][keep_tr] for n in b_names], 1)

    if cfg.get("only"):
        jobs = [j for j in jobs if j[0] in cfg["only"]]
    total = sum(len(s) for _, _, s in jobs)
    done = 0
    for method, k, seeds in jobs:
        for seed in seeds:
            kk_expect = d if method in ("inlp", "leace", "inlp2attr",
                                        "leace2attr") else (
                n_kept if method == "logits" else k)
            if (method, kk_expect, seed) in done_set:
                done += 1                        
                continue
            t0 = time.time()
            D = 2000 if method == "lda_rff" else d
            try:

                kw = {}
                if method in lib.USES_B_LABELS:

                    nattr = lib.N_ADV_ATTRS.get(method, 1)
                    kw["blabels"] = (b_cols[:, :nattr] if nattr > 1
                                     else b_cols[:, 0])
                f = lib.METHODS[method](Utr_k, ytr_k, k, n_kept, seed=seed, **kw)
                Ztr, Zte = f(Utr_all), f(Ute_all)
            except Exception as e:                       
                main_rows.append(dict(dataset=cfg["name"], method=method, k=k,
                                      seed=seed, target="FAILED", note=repr(e)))
                done += 1
                continue

            kk = Ztr.shape[1]
            np.savez_compressed(ZDIR / f"{cfg['name']}_{method}_k{kk}_s{seed}.npz",
                                Ztr=Ztr.astype(np.float32), Zte=Zte.astype(np.float32))
            evaluate(cfg, Ztr, Zte, method, kk, D, seed,
                     keep_tr, keep_te, ho_tr, ho_te, main_rows, probe_rows)


            _ks = cfg["ks"]
            _idx = sorted({int(round(p)) for p in
                           np.linspace(0, len(_ks) - 1, min(4, len(_ks)))})

            recon_ks = {_ks[i] for i in _idx} | {n_kept, max(1, n_kept - 1)}


            _cp = max(1, n_kept - 1)
            _rival = set(lib.VIB_VARIANTS) | {"adv1attr_lam10", "adv2attr_lam10"}
            do_recon = (seed == 0 and cfg["name"] == "rafdb" and (
                (kk in recon_ks and method in ("bottleneck", "lda", "lda_shrink",
                                               "random", "pca", "logits"))
                or (kk == _cp and method in _rival)))


            if (seed == 0 and cfg["name"] == "rafdb"
                    and method in ("leace", "leace2attr")):
                try:
                    run_compat(cfg, Zte, keep_te,
                               dict(dataset=cfg["name"], encoder=cfg["encoder"],
                                    method=method, D=D, k=kk, seed=seed,
                                    transmitted_bytes=kk * 4, note="compat_only"),
                               main_rows)
                except Exception as e:
                    main_rows.append(dict(dataset=cfg["name"], method=method,
                                          k=kk, seed=seed,
                                          target="COMPAT_FAILED", note=repr(e)))

            if do_recon:
                try:
                    rbase = dict(dataset=cfg["name"], encoder=cfg["encoder"],
                                 method=method, D=D, k=kk, seed=seed,
                                 transmitted_bytes=kk * 4, note="recon")
                    _, Uhat_te = run_reconstruction(cfg, Ztr, Zte, rbase,
                                                    main_rows, probe_rows)

                    run_compat(cfg, Uhat_te, keep_te, rbase, main_rows)
                except Exception as e:
                    main_rows.append(dict(dataset=cfg["name"], method=method, k=kk,
                                          seed=seed, target="RECON_FAILED",
                                          note=repr(e)))
            done += 1
            print(f"  [{done:>3}/{total}] {method:<11} k={kk:<4} seed={seed} "
                  f"({time.time()-t0:.1f}s)", flush=True)
            _flush(cfg, main_rows, probe_rows)

    print(f"\ndone. rows={len(main_rows)}  probe_runs={len(probe_rows)}")


def _flush(cfg, main_rows, probe_rows):
    pd.DataFrame(main_rows).to_csv(RES / f"main_{cfg['name']}.csv", index=False)
    pd.DataFrame(probe_rows).to_csv(RES / f"probe_runs_{cfg['name']}.csv", index=False)


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else "rafdb",
        baseline_only="baseline" in sys.argv[2:],
    )
