"""Label statistics for RAF-DB."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "face" / "raf_labels.csv")

train = df[df.split == "train"]
test = df[df.split == "test"]

print("=" * 66)
print(f"RAF-DB  n={len(df):,}  (train {len(train):,} / test {len(test):,})")
print("=" * 66)

print("\n[A] emotion")
print(df.emotion_name.value_counts(normalize=True).to_string(float_format=lambda x: f"{x:.4f}"))

for col in ("gender_name", "race_name", "age_name"):
    print(f"\n[B] {col.replace('_name','')}")
    print(df[col].value_counts(normalize=True).to_string(float_format=lambda x: f"{x:.4f}"))

print("\n" + "=" * 66)
print("A-implied leakage floor (predicting B from expression alone)")
print("=" * 66)
print(f"{'attribute':<10} {'chance':>8} {'floor':>8} {'lift':>8} {'floor_bal':>10}")
print("-" * 66)

for col in ("gender", "race", "age"):
    chance = test[col].value_counts(normalize=True).iloc[0]

    rule = train.groupby("emotion")[col].agg(lambda s: s.mode().iloc[0])
    pred = test["emotion"].map(rule)
    floor = (pred == test[col]).mean()
    bal = np.mean([(pred[test[col] == c] == c).mean() for c in sorted(test[col].unique())])
    print(f"{col:<10} {chance:>8.4f} {floor:>8.4f} {floor-chance:>+8.4f} {bal:>10.4f}")

print("-" * 66)
print("lift ~ 0  ->  A and B nearly independent; chance is an adequate baseline")
print("lift > 0  ->  keeping A structurally forces some B leakage; use the floor")
