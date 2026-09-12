"""Groth16 cost of the commitment circuit over d."""
import argparse
import time

import pandas as pd

import bench_scaling as B

OUT = B.ROOT / "results" / "zk_commit_groth.csv"
FORM, REAL = "commit", False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    done = set()
    if OUT.exists():
        done = set(pd.read_csv(OUT).d.astype(int))

    pts = [d for d in B.DS if d not in done]
    if a.dry:
        for d in B.DS:
            print(f"  d={d:5}  r={B.BASE_R}  {FORM}  nbits={B.NBITS_FIXED}"
                  f"  {'have' if d in done else 'measure'}")
        print(f"\n{len(pts)} points to measure (of {len(B.DS)})")
        return
    if not B.PTAU.exists():
        raise SystemExit(f"ptau not found: {B.PTAU}")

    for i, d in enumerate(pts, 1):
        t0 = time.time()
        row = B.one(d, B.BASE_R, FORM, B.NBITS_FIXED, REAL)
        pd.DataFrame([row]).reindex(columns=B.SCHEMA).to_csv(
            OUT, mode="a" if OUT.exists() else "w",
            header=not OUT.exists(), index=False)
        print(f"[{i}/{len(pts)}] commit d={d:5d} -> "
              f"constraints {row.get('constraints', '-'):>7}  "
              f"({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
