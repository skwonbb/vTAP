"""Export every value shown in the paper's figures and tables to paper_data/.

The figure scripts are imported rather than re-implemented, so the exported
numbers are the ones actually drawn. Table values come from the same scripts
that produce the table CSVs under results/.
"""
import contextlib
import importlib
import io
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "paper_data"
sys.path.insert(0, str(ROOT / "src"))

D, R, F = 768, 4, 16


def load(mod):
    with contextlib.redirect_stdout(io.StringIO()):
        return importlib.import_module(mod)


def write(df, name):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{name}.csv", index=False)
    print(f"  {name + '.csv':16} {len(df):5} rows")


def figure2():
    m = load("fig2")
    acc = m.T.acc("raw U")
    rows = [dict(target=name, best=round(float(acc[:, j].max()), 4),
                 **{f: round(float(v), 4)
                    for f, v in zip(m.T.FAM_DISPLAY, acc[:, j])})
            for j, name, _ in m.ROWS]
    write(pd.DataFrame(rows), "figure2")


def figure4():
    m = load("fig4")
    rows = []
    for name, arr in m.data.items():
        for i, fam in enumerate(m.fams):
            for j, tgt in enumerate(m.TGT):
                rows.append(dict(
                    panel="abcdef"[m.keep.index(name)],
                    method=m.RENAME.get(name, name),
                    end=m.SUB.get(name, ""),
                    d=m.DIM[name],
                    probe_family=fam,
                    target=tgt.strip("$"),
                    retention_pct=round(float(arr[i, j]), 1)))
    write(pd.DataFrame(rows), "figure4")

    g = [dict(item="vTAP", task_retention_pct=round(float(m.TU), 2),
              leakage_pct=round(float(m.TL), 2))]
    for k, v in m.edge.items():
        g.append(dict(item=f"{m.SNAME[k]} frontier at vTAP leakage",
                      task_retention_pct=round(float(v), 2),
                      leakage_pct=round(float(m.TL), 2)))
    g.append(dict(item="vTAP advantage over best (pp)",
                  task_retention_pct=round(float(m.TU - m.best), 2),
                  leakage_pct=np.nan))
    write(pd.DataFrame(g), "figure4g")


def figure5():
    m = load("fig5")
    rows = []
    for k, (label, _) in enumerate(m.ROWS):
        for i, fam in enumerate(m.FAMS):
            for j, tgt in enumerate(m.TGT):
                rows.append(dict(
                    variant=label.replace("$", "").replace("\\", ""),
                    probe_family=fam,
                    target=tgt.strip("$"),
                    panel="abcde"[j],
                    measures=m.WHAT[j],
                    retention_pct=round(float(m.D[k, i, j]), 1)))
    write(pd.DataFrame(rows), "figure5")


def figure6():
    m = load("fig6")
    rows = []
    for j, (col, _, tgt) in enumerate(m.TGT):
        for i, fam in enumerate(m.FAMN):
            rows.append(dict(
                probe_family=fam, target=tgt.strip("$"),
                acc_raw_U=round(float(m.RAW[j, i]), 4),
                acc_Uhat=round(float(m.UHAT[j, i]), 4),
                retention_pct=round(float(m.RET[j, i]), 1),
                delta_acc=round(float(m.DIFF[j, i]), 4)))
        rows.append(dict(
            probe_family="mean of 5", target=tgt.strip("$"),
            acc_raw_U=round(float(m.RAW[j].mean()), 4),
            acc_Uhat=round(float(m.UHAT[j].mean()), 4),
            retention_pct=round(float(m.RET[j].mean()), 1),
            delta_acc=round(float(m.DIFF[j].mean()), 4)))
    write(pd.DataFrame(rows), "figure6")


def figure7():
    m = load("fig7")
    rows = []
    for i, role in enumerate(m.ROLE):
        for j, col in enumerate(m.COL):
            rows.append(dict(panel="a", assigned_task=role, measured=col,
                             retention_pct=float(m.R[i, j])))
    for enc, vals in m.ENC.items():
        for tgt, v in zip(m.TGT, vals):
            rows.append(dict(panel="b", assigned_task=enc,
                             measured=tgt.strip("$"), retention_pct=float(v)))
    write(pd.DataFrame(rows), "figure7")


def figure8():
    m = load("fig8")
    a = pd.read_csv(ROOT / "results" / "quant_error_ablation.csv")
    cum = a.pivot(index="f", columns="stage", values="cum_rel_l2").sort_index()
    inc = a.pivot(index="f", columns="stage", values="inc_rel_l2").sort_index()
    cum, inc = cum.loc[4:], inc.loc[4:]
    share = inc.div(cum[3], axis=0) * 100
    label = {st: lab.replace("$", "").replace("\\!", "").replace("\\to", "->")
                    .replace("\\tilde ", "~").replace("\\hat ", "^")
             for st, lab, _ in m.STAGE}
    rows = []
    for f in cum.index:
        for st in (1, 2, 3):
            rows.append(dict(
                f=int(f), stage=st, step=label[st],
                total_rel_l2=float(cum.loc[f, 3]),
                stage_rel_l2=float(inc.loc[f, st]),
                share_pct=round(float(share.loc[f, st]), 2)))
    write(pd.DataFrame(rows), "figure8")


def figure10():
    m = load("fig10")
    g, h = m.load()
    rd = lambda v: round(float(v), 2)                            # noqa: E731
    rows = []
    for axis, vals in (("d", m.DS), ("r", m.RS)):
        gg, hh = m.series(g, h, axis, vals)
        for col, (title, short, unit, log, fg, fh, fh_lo) in enumerate(m.PANELS):
            tag = "abcdefghijkl"[(0 if axis == "d" else 6) + col]
            for v, xg, xh in zip(vals, gg, hh):
                d, r = (v, R) if axis == "d" else (D, v)
                rows.append(dict(panel=tag, quantity=title, unit=unit,
                                 sweep=axis, d=d, r=r,
                                 groth16=rd(fg(xg)), halo2=rd(fh(xh)),
                                 halo2_srs=rd(fh_lo(xh)) if fh_lo else np.nan,
                                 halo2_keygen=(rd(fh(xh) - fh_lo(xh))
                                               if fh_lo else np.nan)))
    write(pd.DataFrame(rows), "figure10")


def figure9():
    m = load("fig9")
    gt, gcm, ht, hcm = m.bars(*m.load())
    pts = [("d", d, R) for d in m.DS] + [("r", D, r) for r in m.RS]
    rows = []
    for i, (sweep, d, r) in enumerate(pts):
        for panel, unit, tap, com in (("a", "R1CS constraints", gt[i], gcm[i]),
                                      ("b", "PLONKish advice cells", ht[i], hcm[i])):
            rows.append(dict(panel=panel, unit=unit, sweep=sweep, d=d, r=r,
                             tap=int(tap), commitment=int(com),
                             total=int(tap + com),
                             tap_share_pct=round(100 * tap / (tap + com), 1)))
    write(pd.DataFrame(rows), "figure9")


def table5():
    write(load("tables").ablation(), "table5")


def table6():
    q = pd.read_csv(ROOT / "results" / "quant_form.csv").set_index("f").loc[F]
    rows = [
        dict(quantity="rescalings", factored=R + D, direct=D,
             ratio=round((R + D) / D, 3)),
        dict(quantity="constant multiplications", factored=2 * D * R,
             direct=D * D, ratio=round(D * D / (2 * D * R), 1)),
        dict(quantity="quantization error (rel. L2)",
             factored=float(q.fac_rel_l2), direct=float(q.dir_rel_l2),
             ratio=round(float(q.ratio), 1)),
        dict(quantity="published parameters (entries)",
             factored=2 * D * R + D, direct=D * D + D,
             ratio=round((D * D + D) / (2 * D * R + D), 1)),
    ]
    write(pd.DataFrame(rows), "table6")


def table7():
    write(load("tab_quant").table_b(), "table7")


def table8():
    write(load("tab_compat_quant").table(), "table8")


if __name__ == "__main__":
    for fn in (figure2, figure4, figure5, figure6, figure7, figure8,
               figure9, figure10, table5, table6, table7, table8):
        fn()
    print(f"-> {OUT}")
