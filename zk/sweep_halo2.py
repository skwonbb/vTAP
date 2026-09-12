"""halo2-KZG cost over d and r."""
import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
ZK2 = ROOT / "zk2"
PARAMS = ROOT / "results" / "tap_params"
OUT = ROOT / "results" / "zk_halo2_gated.csv"


MAX_COLS = 4
LOOKUP_BITS = 12


NBITS = 48

DS = [64, 128, 192, 256, 384, 512, 640, 768, 896, 1024]
RS = [1, 2, 8, 12, 16]
R_AT_D = 4      
D_AT_R = 768    

SCHEMA = [
    "system", "form", "d", "r", "f", "b_bits", "k", "lookup_bits", "max_cols",
    "advice_cells", "tap_advice", "pos_advice", "lookup_cells", "fixed",
    "advice_cols", "lookup_cols", "pos_cols",
    "srs_s", "keygen_s", "prove_s", "verify_ms",
    "proof_bytes", "srs_bytes", "pk_bytes", "vk_bytes", "instances",
    "stage_failed",
]


def gen(d, r):
    base = [sys.executable, str(ROOT / "zk" / "gen_circuits.py"),
            "--d", str(d), "--r", str(r), "--f", "16", "--random"]
    for extra in (["--nbits", str(NBITS)], []):
        if subprocess.run(base + extra, capture_output=True, text=True,
                          cwd=ROOT / "zk").returncode == 0:
            return json.loads((PARAMS / "quantized.json").read_text())["nbits"]
    return None


def run(binary, d, r, form=None):
    exe = ZK2 / "target" / "release" / (binary + (".exe" if sys.platform == "win32" else ""))
    cmd = [str(exe), "--params", str(PARAMS), "--d", str(d), "--r", str(r),
           "--f", "16", "--lookup_bits", str(LOOKUP_BITS), "--max_cols", str(MAX_COLS)]
    if form:
        cmd += ["--form", form]
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ZK2)
    if p.returncode != 0:
        tail = (p.stderr or p.stdout).strip().splitlines()[-1:] or ["?"]
        return {"stage_failed": tail[0][:200]}
    for line in reversed(p.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and '"system"' in line:
            row = json.loads(line)
            row["wall_s"] = round(time.time() - t, 1)
            return row
    return {"stage_failed": "no output"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    jobs = [(d, R_AT_D) for d in DS] + [(D_AT_R, r) for r in RS if r != R_AT_D]
    if a.dry:
        for d, r in jobs:
            print(f"  d={d:5} r={r:3}")
        print(f"\n{len(jobs)} points")
        return

    backup = PARAMS.parent / "tap_params.bak"
    if backup.exists():
        shutil.rmtree(backup)
    shutil.copytree(PARAMS, backup)

    rows = []
    for d, r in jobs:
        nbits = gen(d, r)
        if nbits is None:
            print(f"  d={d} r={r} parameter generation failed", flush=True)
            continue
        for binary, form in (("tap_halo2", None),):
            row = run(binary, d, r, form)
            row.setdefault("system", "?")
            row.setdefault("d", d)
            row.setdefault("r", r)
            row["max_cols"] = MAX_COLS
            row["b_bits"] = nbits

            if "tap_advice" in row:
                row["advice_cells"] = row["tap_advice"] + row.get("pos_advice", 0)
            rows.append({k: row.get(k) for k in SCHEMA})
            print(f"  d={d:5d} r={r:3d} {row.get('system','?'):18s} "
                  f"k={row.get('k')} cells={row.get('advice_cells')} "
                  f"B={nbits} cols={row.get('advice_cols')}+{row.get('lookup_cols')}+6 "
                  f"prove={row.get('prove_s')} proof={row.get('proof_bytes')} "
                  f"{row.get('stage_failed') or ''}", flush=True)

    if not rows:
        # Never overwrite a good file with nothing: a run that measured no
        # point measured nothing worth writing.
        shutil.rmtree(PARAMS, ignore_errors=True)
        backup.rename(PARAMS)
        raise SystemExit("no point was measured; results/ left untouched")

    import csv
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SCHEMA)
        w.writeheader()
        w.writerows(rows)
    print("[wrote]", OUT)
    shutil.rmtree(PARAMS)
    backup.rename(PARAMS)
    print("[restored]", PARAMS)


if __name__ == "__main__":
    main()
