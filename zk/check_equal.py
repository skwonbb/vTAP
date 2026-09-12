"""Check that the two arithmetizations compute the same function."""
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ZK2 = ROOT / "zk2"
PARAMS = ROOT / "results" / "tap_params"
P = 21888242871839275222246405745257275088548364400416034343698204186575808495617


def signed(x):
    x %= P
    return x - P if x > P // 2 else x


def main():
    circom = json.loads((ROOT / "zk" / "build" / "tap_full.public.json").read_text())
    d = len(circom) - 1  
    want = [signed(int(v)) for v in circom[1:]]

    exe = ZK2 / "target" / "release" / ("tap_halo2" + (".exe" if sys.platform == "win32" else ""))
    p = subprocess.run(
        [str(exe), "--params", str(PARAMS), "--d", str(d), "--r", "4", "--f", "16",
         "--lookup_bits", "12", "--max_cols", "8"],
        capture_output=True, text=True, cwd=ZK2)
    if p.returncode != 0:
        print("halo2 run failed:", (p.stderr or p.stdout).strip()[-300:])
        return 1

    got = [signed(int(s, 16)) for s in
           json.loads((ZK2 / "halo2_output_factored.json").read_text())]

    if len(got) != len(want):
        print(f"length mismatch: circom {len(want)} vs halo2 {len(got)}")
        return 1
    bad = [(i, a, b) for i, (a, b) in enumerate(zip(want, got)) if a != b]
    print(f"d={d}  match {len(want) - len(bad)}/{len(want)}")
    for i, a, b in bad[:5]:
        print(f"  [{i}] circom {a} != halo2 {b}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
