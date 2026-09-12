"""Per-family retention, and the family-averaged ablation table."""
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
CSV = ROOT / "results/probe_runs_rafdb.csv"

TARGETS = ["A_kept", "Aprime_heldout", "B_gender", "B_race", "B_age"]
CHANCE = np.array([.2, .5, .5, 1 / 3, .2])
FAM = ["lr", "mlp", "rff", "tree", "knn"]
FAM_DISPLAY = ["Linear", "MLP", "RFF", "GBM", "kNN"]


RECON_TARGETS = ["recon_A", "recon_Aprime", "recon_B_gender",
                 "recon_B_race", "recon_B_age"]


ROWS = {


    "raw U":            dict(method="raw_U", note="any"),
    "vTAP":              dict(method="lda_shrink", k=4, note="recon_linear",
                             targets=RECON_TARGETS),
    "w/o M":            dict(method="lda_shrink", k=4),
    "w/o M, alpha":     dict(method="lda", k=4),
    "w/o M, alpha, SW": dict(method="between", k=4),
    "VIB utility":      dict(method="vib_b1", k=16),
    "VIB privacy":      dict(method="vib_b01", k=2),
    "ADV utility":      dict(method="adv2attr_lam10", k=8),
    "ADV privacy":      dict(method="adv2attr_lam10", k=2),
    "LEACE":            dict(method="leace2attr", k=768),
}


DIM = {"vTAP": 768, "LEACE": 768, "VIB utility": 16, "VIB privacy": 2,
       "ADV utility": 8, "ADV privacy": 2, "raw U": 768,
       "w/o M": 4, "w/o M, alpha": 4, "w/o M, alpha, SW": 4}

_CACHE = {}


def _load():
    if "df" not in _CACHE:
        _CACHE["df"] = pd.read_csv(CSV)
    return _CACHE["df"]


def acc(name):
    spec = ROWS[name]
    df = _load()
    sel = df[df.method == spec["method"]]
    if "k" in spec:
        sel = sel[sel.k == spec["k"]]

    if spec.get("note") == "any":
        pass
    elif "note" in spec:
        sel = sel[sel.note == spec["note"]]
    else:
        sel = sel[sel.note.isna()]
    targets = spec.get("targets", TARGETS)

    out = np.full((len(FAM), len(targets)), np.nan)
    for i, f in enumerate(FAM):
        for j, t in enumerate(targets):
            d = sel[(sel.family == f) & (sel.target == t)]
            if len(d):
                out[i, j] = d.groupby("seed")["test_bal"].max().mean()
    if np.isnan(out).any():
        raise SystemExit(f"[tables] '{name}' has an empty cell: {spec}")
    return out


def retention(name, decimals=0):
    base = acc("raw U")
    r = (acc(name) - CHANCE) / (base - CHANCE) * 100
    return np.round(r, decimals) if decimals is not None else r


def collapse(name):
    return (acc(name).max(0) - CHANCE) / (acc("raw U").max(0) - CHANCE) * 100


ABL = ["vTAP", "w/o M", "w/o M, alpha", "w/o M, alpha, SW"]
TGT_SHORT = ["A", "A'", "B1", "B2", "B3"]


def ablation():
    rows = []
    for nm in ABL:
        rows.append(dict(method=nm, d=DIM[nm],
                         **{t: round(float(v), 1) for t, v in
                            zip(TGT_SHORT, retention(nm, None).mean(0))},
                         **{t + "_acc": round(float(v), 3) for t, v in
                            zip(TGT_SHORT, acc(nm).mean(0))}))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    for nm in ROWS:
        print(f"\n=== {nm}  (d={DIM[nm]}) ===")
        print(pd.DataFrame(retention(nm), index=FAM_DISPLAY,
                           columns=TGT_SHORT).astype(int))
    print("\n\n=== ablation, mean over five families (retention %, balanced accuracy in parentheses) ===")
    print(ablation().to_string(index=False))
