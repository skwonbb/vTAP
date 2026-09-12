"""Groth16 cost over d and r."""
import argparse
import json
import pathlib
import random
import re
import shutil
import subprocess
import time

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
CIRC, BUILD = HERE / "circuits", HERE / "build_scale"
LIBS = HERE / "node_modules" / "circomlib" / "circuits"
PTAU = HERE / "ptau" / "pot17_final.ptau"
OUT = ROOT / "results" / "zk_scaling.csv"
P = 21888242871839275222246405745257275088548364400416034343698204186575808495617

F = 16
NBITS_FIXED = 48
DS = [64, 128, 192, 256, 384, 512, 640, 768, 896, 1024]
RS = [1, 2, 3, 4, 6, 8, 12, 16]
BASE_D, BASE_R = 768, 4
SHUFFLE_SEED = 0


SCHEMA = ["d", "r", "f", "form", "real_params", "nbits_requested", "nbits",
          "src_bytes", "compile_s", "nonlinear", "linear", "pub_out", "priv_in",
          "wires", "constraints", "r1cs_bytes", "wasm_bytes", "setup_s",
          "zkey_bytes", "vkey_bytes", "witness_s", "prove_s", "proof_bytes",
          "verify_ms", "stage_failed", "note"]


def plan():
    pts = []
    for d in DS:                       
        for form in ("tap_factored", "tap_direct", "tap_full"):
            pts.append((d, BASE_R, form, NBITS_FIXED, False))


    for r in RS:
        if r == BASE_R:
            continue                   
        for form in ("tap_factored", "tap_direct", "tap_full"):
            pts.append((BASE_D, r, form, NBITS_FIXED, False))

    for nb in (NBITS_FIXED, 0):
        for form in ("tap_factored", "commit", "tap_full"):
            pts.append((BASE_D, BASE_R, form, nb, True))
    random.Random(SHUFFLE_SEED).shuffle(pts)
    return pts


def sh(cmd, timeout=2400):
    t0 = time.perf_counter()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       cwd=str(HERE), timeout=timeout,
                       encoding="utf-8", errors="replace")
    return (time.perf_counter() - t0, r.returncode,
            (r.stdout or "") + (r.stderr or ""))


def counts(log):
    plain = re.sub(r"\x1b\[[0-9;]*m", "", log)
    out = {}
    for key, tag in (("non-linear constraints", "nonlinear"),
                     ("linear constraints", "linear"),
                     ("public outputs", "pub_out"),
                     ("private inputs", "priv_in"), ("wires", "wires")):
        pat = re.compile(r"^\s*" + re.escape(key) + r":\s*(\d+)")
        for line in plain.splitlines():
            if (m := pat.match(line)):
                out[tag] = int(m.group(1)); break
    return out


def make_input(form, d):
    U = np.load(ROOT / "cache" / "embeddings" / "rafdb_clip.npy")[0]
    U = np.resize(np.asarray(U, np.float64), d)     
    Uq = np.rint(U * (1 << F)).astype(np.int64)
    inp = {"U": [str(int(v) % P) for v in Uq]}
    if form in ("commit", "tap_full"):
        inp["rho"] = str(12345678901234567890 % P)
    p = BUILD / "in.json"
    p.write_text(json.dumps(inp), encoding="utf-8")
    return p


VERIFY_JS = """
const snarkjs = require("snarkjs"), fs = require("fs");
(async () => {{
  const a = process.argv.slice(2).map(p => JSON.parse(fs.readFileSync(p)));
  await snarkjs.groth16.verify(a[0], a[1], a[2]);
  let best = Infinity;
  for (let i = 0; i < 20; i++) {{
    const t = process.hrtime.bigint();
    const ok = await snarkjs.groth16.verify(a[0], a[1], a[2]);
    const dt = Number(process.hrtime.bigint() - t) / 1e6;
    if (!ok) {{ console.error("verify failed"); process.exit(1); }}
    if (dt < best) best = dt;
  }}
  console.log(JSON.stringify({{ms: best}}));
  process.exit(0);
}})();
"""


def one(d, r, form, nbits, real):
    tag = f"{form} d={d} r={r} nbits={nbits or 'auto'} {'real' if real else 'rand'}"
    row = dict(d=d, r=r, f=F, form=form, real_params=real,
               nbits_requested=nbits or None)
    BUILD.mkdir(parents=True, exist_ok=True)

    gen = (f'python "{HERE / "gen_circuits.py"}" --d {d} --r {r} --f {F}'
           + ("" if real else " --random")
           + (f" --nbits {nbits}" if nbits else ""))
    _, rc, log = sh(gen)
    if rc:
        row.update(stage_failed="generate", note=log.strip()[-120:])
        return row
    row["nbits"] = json.loads((CIRC / "params.json").read_text())["nbits"]
    src = CIRC / f"{form}.circom"
    row["src_bytes"] = src.stat().st_size

    t, rc, log = sh(f'circom "{src}" --r1cs --wasm -o "{BUILD}" -l "{LIBS}"')
    row["compile_s"] = round(t, 3)
    if rc:
        row.update(stage_failed="compile",
                   note="parser stack overflow" if "overflowed" in log else log.strip()[-120:])
        return row
    row.update(counts(log))
    row["constraints"] = row.get("nonlinear", 0) + row.get("linear", 0)
    row["r1cs_bytes"] = (BUILD / f"{form}.r1cs").stat().st_size
    row["wasm_bytes"] = (BUILD / f"{form}_js" / f"{form}.wasm").stat().st_size


    zkey, vkey = BUILD / f"{form}.zkey", BUILD / f"{form}.vkey.json"
    t, rc, log = sh(f'snarkjs groth16 setup "{BUILD}/{form}.r1cs" "{PTAU}" "{zkey}"')
    if rc:
        row.update(stage_failed="setup", note=log.strip()[-120:])
        return row
    row["setup_s"] = round(t, 2)
    row["zkey_bytes"] = zkey.stat().st_size
    sh(f'snarkjs zkey export verificationkey "{zkey}" "{vkey}"')
    row["vkey_bytes"] = vkey.stat().st_size

    inp = make_input(form, d)
    wtns = BUILD / f"{form}.wtns"
    t, rc, log = sh(f'node "{BUILD}/{form}_js/generate_witness.js" '
                    f'"{BUILD}/{form}_js/{form}.wasm" "{inp}" "{wtns}"')
    row["witness_s"] = round(t, 3)
    if rc:
        row.update(stage_failed="witness",
                   note=("WASM function size limit exceeded"
                         if "maximum function size" in log else log.strip()[-120:]))
        return row

    pf, pub = BUILD / f"{form}.proof.json", BUILD / f"{form}.public.json"
    t, rc, log = sh(f'snarkjs groth16 prove "{zkey}" "{wtns}" "{pf}" "{pub}"')
    if rc:
        row.update(stage_failed="prove", note=log.strip()[-120:])
        return row
    row["prove_s"] = round(t, 3)
    row["proof_bytes"] = pf.stat().st_size

    js = BUILD / "verify.js"
    js.write_text(VERIFY_JS.format(), encoding="utf-8")
    _, rc, o = sh(f'node "{js}" "{vkey}" "{pub}" "{pf}"')
    if rc:
        row.update(stage_failed="verify", note=o.strip()[-120:])
        return row
    row["verify_ms"] = round(json.loads(o.strip().splitlines()[-1])["ms"], 3)
    row["stage_failed"] = None
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    pts = plan()
    if a.dry:
        for p in pts:
            print(p)
        print(f"\n{len(pts)} points")
        return
    if not PTAU.exists():
        raise SystemExit(f"ptau not found: {PTAU}")

    done = set()
    if OUT.exists():
        old = pd.read_csv(OUT)
        done = set(map(tuple, old[["d", "r", "form", "nbits_requested",
                                   "real_params"]].fillna(-1).values))
    for i, (d, r, form, nb, real) in enumerate(pts, 1):
        if (d, r, form, nb if nb else -1, real) in done:
            print(f"[{i}/{len(pts)}] skip {form} d={d} r={r}", flush=True)
            continue
        t0 = time.time()
        row = one(d, r, form, nb, real)
        pd.DataFrame([row]).reindex(columns=SCHEMA).to_csv(
            OUT, mode="a" if OUT.exists() else "w",
            header=not OUT.exists(), index=False)
        print(f"[{i}/{len(pts)}] {form:13s} d={d:5d} r={r:2d} "
              f"nb={row.get('nbits','-')} {'real' if real else 'rand'} -> "
              f"constraints {row.get('constraints','-'):>7} "
              f"prove {row.get('prove_s','-')} verify {row.get('verify_ms','-')}ms "
              f"[{row.get('stage_failed') or 'ok'}] ({time.time()-t0:.0f}s)",
              flush=True)
    shutil.rmtree(BUILD, ignore_errors=True)
    print(f"\n[wrote] {OUT}")


if __name__ == "__main__":
    main()
