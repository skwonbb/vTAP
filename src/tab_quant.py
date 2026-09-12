"""Accuracy under fixed-point arithmetic at f=16."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import tables as T  

ROOT = SRC.parent
R = ROOT / "results"
DEPLOY_F = 16

ROWS_A = (4, 8, 12, 16, 20, 24)
TT = ["A_kept", "Aprime_heldout", "B_gender", "B_race", "B_age"]
LABEL = {"A_kept": "$A$", "Aprime_heldout": "$A'$", "B_gender": "$B_1$",
         "B_race": "$B_2$", "B_age": "$B_3$"}
FAM = ["lr", "mlp", "rff", "tree", "knn"]
CH = np.array([.2, .5, .5, 1 / 3, .2])


def table_b():
    q = pd.read_csv(R / "quant_sweep.csv")
    q = q[q.f == DEPLOY_F]
    fx = np.array([[q[(q.family == f) & (q.target == t)]
                    .groupby("seed")["test_bal"].max().mean()
                    for t in TT] for f in FAM])
    fl = T.acc("vTAP")                     
    raw = T.acc("raw U")

    ret = lambda A: (A.max(0) - CH) / (raw.max(0) - CH) * 100      
    out = pd.DataFrame({
        "target": [LABEL[t] for t in TT],


        "bal_float": fl.max(0).round(4), "bal_fixed": fx.max(0).round(4),
        "delta": (fx.max(0) - fl.max(0)).round(4),
        "ret_float": ret(fl).round(1), "ret_fixed": ret(fx).round(1),


        "seed_std": [round(float(q[q.target == t]
                                 .groupby(["family", "seed"])["test_bal"].max()
                                 .groupby(level=0).std(ddof=1).max()), 4) for t in TT],
    })
    out.to_csv(R / "quant_table_acc.csv", index=False)
    return out


if __name__ == "__main__":
    B = table_b()
    pd.set_option("display.width", 200)
    print(f"=== accuracy at f={DEPLOY_F} (results/quant_table_acc.csv) ===")
    print(B.to_string(index=False))
