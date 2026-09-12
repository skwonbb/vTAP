"""Build the RAF-DB label table."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAF = ROOT / "data" / "face" / "rafdb" / "rafdb_basic"
MANUAL = RAF / "Annotation" / "manual"

EMOTION = {
    1: "surprise", 2: "fear", 3: "disgust", 4: "happiness",
    5: "sadness", 6: "anger", 7: "neutral",
}
GENDER = {0: "male", 1: "female", 2: "unsure"}
RACE = {0: "caucasian", 1: "african_american", 2: "asian"}
AGE = {0: "0-3", 1: "4-19", 2: "20-39", 3: "40-69", 4: "70+"}

rows = []
missing = []

with open(RAF / "EmoLabel" / "list_patition_label.txt") as fh:
    entries = [ln.split() for ln in fh if ln.strip()]

for fname, emo in entries:
    stem = Path(fname).stem                      
    attr = MANUAL / f"{stem}_manu_attri.txt"
    if not attr.exists():
        missing.append(stem)
        continue

    vals = [ln.strip() for ln in attr.read_text().splitlines() if ln.strip()]
    lm = [float(v) for pair in vals[:5] for v in pair.split()]
    gender, race, age = (int(vals[5]), int(vals[6]), int(vals[7]))

    rows.append(
        {
            "image": fname,
            "stem": stem,
            "split": stem.split("_")[0],          
            "emotion": int(emo),
            "emotion_name": EMOTION[int(emo)],
            "gender": gender,
            "gender_name": GENDER[gender],
            "race": race,
            "race_name": RACE[race],
            "age": age,
            "age_name": AGE[age],
            **{f"lm{i}": v for i, v in enumerate(lm)},
        }
    )

df = pd.DataFrame(rows)
out = ROOT / "data" / "face" / "raf_labels.csv"
df.to_csv(out, index=False)

print(f"wrote {out}  n={len(df):,}  (missing attr: {len(missing)})")
print(df["split"].value_counts().to_string())
print("\ncolumns:", [c for c in df.columns if not c.startswith("lm")], "+ lm0..lm9")
