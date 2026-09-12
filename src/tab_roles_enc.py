"""Role assignment and encoder tables, on Z and on Uhat."""
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
R = ROOT / "results"
MAIN = pd.read_csv(R / "probe_runs_rafdb.csv")
ROLES = pd.read_csv(R / "probe_runs_rafdb_roles.csv")
SWAP = pd.read_csv(R / "probe_runs_rafdb_swap.csv")
DINOL = pd.read_csv(R / "probe_runs_rafdb_dinol.csv")
UH_ROLE = pd.read_csv(R / "recon_appendix_roles.csv")
UH_ENC = pd.read_csv(R / "recon_appendix_enc.csv")

ATTR = ["emotion", "gender", "race", "age"]


CH_B = {"emotion": 1 / 7, "gender": .5, "race": 1 / 3, "age": .2}
CH_A = {"emotion": .2, "gender": .5, "race": 1 / 3, "age": 1 / 3}
CH_AP = .5                       


def best(df, target, method=None, note=None, seeds=(0, 1, 2)):
    q = df[df.target == target]
    if method is not None:
        q = q[q.method == method]
    if note is not None:
        q = q[q.note.astype(str) == note] if note else q[q.note.isna()]
    v = [q[q.seed == s].test_bal.max() for s in seeds if (q.seed == s).any()]
    return float(np.mean(v)) if v else np.nan


BASE_B = {"gender": best(MAIN, "B_gender", "raw_U"),
          "race": best(MAIN, "B_race", "raw_U"),
          "age": best(MAIN, "B_age", "raw_U"),
          "emotion": best(SWAP, "B_emotion", "raw_U")}
BASE_A = {"emotion": best(MAIN, "A_kept", "raw_U"),
          "gender": best(SWAP, "A_kept", "raw_U"),
          "race": best(ROLES[ROLES.note == "A=race"], "A_kept", "raw_U"),
          "age": best(ROLES[ROLES.note == "A=age"], "A_kept", "raw_U")}
BASE_AP = {"emotion": best(MAIN, "Aprime_heldout", "raw_U"),
           "age": best(ROLES[ROLES.note == "A=age"], "Aprime_heldout", "raw_U")}


ROLE_SRC = {
    "emotion": (4, MAIN, dict(method="lda_shrink"), "A=emotion"),
    "gender": (1, SWAP, dict(method="lda_shrink"), "A=gender"),
    "race": (2, ROLES[ROLES.note == "A=race"], dict(method="lda_shrink"), "A=race"),
    "age": (2, ROLES[ROLES.note == "A=age"], dict(method="lda_shrink"), "A=age"),
}
NAME = {"emotion": "Expr.", "gender": "Gender", "race": "Race", "age": "Age"}


RECON_NAME = {"A_kept": "recon_A", "Aprime_heldout": "recon_Aprime",
              "B_gender": "recon_B_gender", "B_race": "recon_B_race",
              "B_age": "recon_B_age"}


def uhat_acc(a, tag, tname):
    if a == "emotion":
        t = RECON_NAME.get(tname)
        return best(MAIN, t, "lda_shrink", "recon_linear") if t else float("nan")
    return best(UH_ROLE[UH_ROLE.setting == tag], tname)


def ret(acc, ch, base):
    return (acc - ch) / (base - ch) * 100


def role_rows(uhat):
    out = []
    for a in ATTR:
        r, src, sel, tag = ROLE_SRC[a]
        cells = []
        for col in ATTR:
            if col == a:
                tname, ch, base = "A_kept", CH_A[a], BASE_A[a]
            else:
                tname, ch, base = f"B_{col}", CH_B[col], BASE_B[col]
            acc = (uhat_acc(a, tag, tname) if uhat else best(src, tname, **sel))
            cells.append((ret(acc, ch, base), acc) if acc == acc else None)
        ap = None
        if a in BASE_AP:
            acc = (uhat_acc(a, tag, "Aprime_heldout") if uhat
                   else best(src, "Aprime_heldout", **sel))
            if acc == acc:
                ap = (ret(acc, CH_AP, BASE_AP[a]), acc)
        out.append((NAME[a], r, cells, ap))
    return out


def cell(v, bold=False):
    if v is None:
        return "      ---     "
    return f"{v[0]:4.0f}%{'*' if bold else ' '}({v[1]:.3f})"


def emit_roles(uhat):
    print(f"\n=== role assignment - {'Uhat' if uhat else 'Z'}"
          f"   (* marks the attribute that row assigns as the permitted task)")
    print(f"{'assigned':>8} {'r':>2}  "
          + "  ".join(f"{NAME[a]:>13}" for a in ATTR) + f"  {'A-prime':>13}")
    for nm, r, cells, ap in role_rows(uhat):
        diag = ATTR[[NAME[a] for a in ATTR].index(nm)]
        body = "  ".join(cell(c, bold=(ATTR[i] == diag))
                         for i, c in enumerate(cells))
        print(f"{nm:>8} {r:>2}  {body}  {cell(ap)}")


def emit_enc(uhat):
    print(f"\n=== encoder - {'Uhat' if uhat else 'Z'}")
    print(f"{'encoder':>16} " + "  ".join(
        f"{t:>13}" for t in ("A", "A-prime", "gender", "race", "age")))
    TG = [("A_kept", CH_A["emotion"], BASE_A["emotion"]),
          ("Aprime_heldout", CH_AP, BASE_AP["emotion"]),
          ("B_gender", CH_B["gender"], BASE_B["gender"]),
          ("B_race", CH_B["race"], BASE_B["race"]),
          ("B_age", CH_B["age"], BASE_B["age"])]
    for label, src, uh, sel in [
            ("CLIP ViT-L/14", MAIN, UH_ROLE[UH_ROLE.setting == "A=emotion"],
             dict(method="lda_shrink")),
            ("DINOv2 ViT-L/14", DINOL, UH_ENC, dict(method="lda_shrink"))]:
        base = {t: (best(src, t, "raw_U") if label.startswith("DINO") else b)
                for t, _, b in TG}
        cells = []
        for t, ch, _ in TG:
            acc = (best(uh, t if not uhat else t) if uhat
                   else best(src, t, **sel))
            if uhat and label.startswith("CLIP"):
                acc = best(MAIN, "recon_" + t.replace("_kept", "")
                           .replace("_heldout", "").replace("B_", "B_"),
                           "lda_shrink", "recon_linear")
            cells.append((ret(acc, ch, base[t]), acc) if acc == acc else None)
        print(f"{label:>16} " + "  ".join(cell(c) for c in cells))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    print("baseline A :", {k: round(v, 3) for k, v in BASE_A.items()})
    print("baseline B :", {k: round(v, 3) for k, v in BASE_B.items()})
    print("baseline A':", {k: round(v, 3) for k, v in BASE_AP.items()})
    if which in ("z", "both"):
        emit_roles(False)
    if which in ("uhat", "both"):
        emit_roles(True)

    if which in ("uhat", "both"):
        emit_enc(True)
    if which in ("z", "both"):
        emit_enc(False)
