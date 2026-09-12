"""Re-measure rows of zk_scaling.csv that failed a stage."""
import csv
import pathlib
import shutil

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
CSV = ROOT / "results" / "zk_scaling.csv"
SCHEMA = ["d", "r", "f", "form", "real_params", "nbits_requested", "nbits",
          "src_bytes", "compile_s", "nonlinear", "linear", "pub_out", "priv_in",
          "wires", "constraints", "r1cs_bytes", "wasm_bytes", "setup_s",
          "zkey_bytes", "vkey_bytes", "witness_s", "prove_s", "proof_bytes",
          "verify_ms", "stage_failed", "note"]

PREFIX = {23: 21, 8: 6}


def main():
    raw = list(csv.reader(CSV.open(encoding="utf-8")))
    hdr, body = raw[0], [r for r in raw[1:] if r]
    print(f"header {len(hdr)} cols, data {len(body)} rows")

    fixed, counts = [], {}
    for r in body:
        n = len(r)
        counts[n] = counts.get(n, 0) + 1
        row = dict.fromkeys(SCHEMA)
        if n >= 25:                       
            for k, v in zip(hdr, r):
                if k in row:
                    row[k] = v
        elif n in PREFIX:                 
            p = PREFIX[n]
            for k, v in zip(hdr[:p], r[:p]):
                row[k] = v
            row["stage_failed"], row["note"] = r[p], r[p + 1] if n > p + 1 else None
        else:
            raise SystemExit(f"unexpected field count {n}: {r[:6]}")
        fixed.append(row)

    print("field count distribution:", counts)
    shutil.copy(CSV, CSV.with_suffix(".csv.broken"))
    df = pd.DataFrame(fixed, columns=SCHEMA)
    for c in ("d", "r", "f", "nbits", "constraints", "nonlinear", "linear",
              "pub_out", "priv_in", "wires", "src_bytes", "r1cs_bytes",
              "wasm_bytes", "zkey_bytes", "vkey_bytes", "proof_bytes",
              "nbits_requested"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ("compile_s", "setup_s", "witness_s", "prove_s", "verify_ms"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values(["form", "r", "d"]).reset_index(drop=True)
    df.to_csv(CSV, index=False)

    ok = df.stage_failed.isna().sum()
    print(f"repaired: {ok} rows ok, {len(df) - ok} failed")
    print(df[df.stage_failed.notna()][
        ["d", "r", "form", "nbits", "constraints", "compile_s", "setup_s",
         "stage_failed", "note"]].to_string(index=False))


if __name__ == "__main__":
    main()
