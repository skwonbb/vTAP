"""Build zk/build/tap_full.* with the deployed parameters, for check_equal.py.

bench_scaling.py sweeps into a scratch directory and removes it when it is
done, so the artifacts check_equal.py reads are not left behind by the sweep.
This reruns the same steps once, at the deployed point, and keeps them.
"""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bench_scaling as B                                        # noqa: E402

D, R, NBITS = 768, 4, 48


def main():
    B.BUILD = HERE / "build"
    B.BUILD.mkdir(parents=True, exist_ok=True)
    if not B.PTAU.exists():
        raise SystemExit(f"ptau not found: {B.PTAU}")

    row = B.one(D, R, "tap_full", NBITS, real=True)
    failed = row.get("stage_failed")
    if failed:
        raise SystemExit(f"failed at {failed}: {row.get('note', '')}")

    pub = B.BUILD / "tap_full.public.json"
    print(f"[wrote] {pub}")
    print(f"  constraints {row['constraints']}  prove {row['prove_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
