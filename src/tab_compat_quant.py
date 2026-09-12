"""Receiver compatibility at f=16, against the real-valued transform.

The aggregation follows tab_quant.py: within a family take the best
configuration per seed, average over seeds, then take the best family. The
seed spread is the yardstick the arithmetization difference is read against.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

R = SRC.parent / "results"
SRCCSV = R / "compat_quant.csv"
TT = ["compat_A", "compat_B_gender", "compat_B_race", "compat_B_age"]
LABEL = {"compat_A": "$A$", "compat_B_gender": "$B_1$",
         "compat_B_race": "$B_2$", "compat_B_age": "$B_3$"}


def best(d, col, t):
    s = d[d.target == t]
    per = s.groupby(["family", "seed"])[col].max().groupby(level=0).mean()
    return float(per.max())


def spread(d, col, t):
    s = d[d.target == t]
    per = s.groupby(["family", "seed"])[col].max()
    return float(per.groupby(level=0).std(ddof=1).max())


def table():
    d = pd.read_csv(SRCCSV)
    rows = []
    for t in TT:
        real = best(d, "on_Uhat_real", t)
        fx = best(d, "on_Uhat_f", t)
        rows.append(dict(
            target=LABEL[t],
            on_U=round(best(d, "on_U", t), 4),
            bal_real=round(real, 4),
            bal_fixed=round(fx, 4),
            delta=round(fx - real, 4),
            seed_std=round(spread(d, "on_Uhat_real", t), 4)))
    out = pd.DataFrame(rows)
    out["of_U_real"] = (out.bal_real / out.on_U * 100).round(1)
    out["of_U_fixed"] = (out.bal_fixed / out.on_U * 100).round(1)
    return out


if __name__ == "__main__":
    T = table()
    pd.set_option("display.width", 200)
    print("=== receiver trained on raw U, given the transform at f=16 ===")
    print(T.to_string(index=False))
    print(f"\nlargest |delta| = {T.delta.abs().max():.4f}, "
          f"largest seed_std = {T.seed_std.max():.4f}")
