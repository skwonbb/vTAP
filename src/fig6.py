"""Figure 6 - receiver models trained on raw U, given Uhat."""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.linewidth": .7,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": .7, "ytick.major.width": .7,
})

ROOT = pathlib.Path(__file__).resolve().parent.parent
C = pd.read_csv(ROOT / "results/compat_5fam.csv")
C = C[C.method == "lda_shrink"]

FAM = ["lr", "mlp", "rff", "tree", "knn"]
FAMN = ["Lin.", "MLP", "RFF", "GBM", "kNN"]   
TGT = [("compat_A", .2, "$A$"), ("compat_B_gender", .5, "$B_1$"),
       ("compat_B_race", 1 / 3, "$B_2$"), ("compat_B_age", .2, "$B_3$")]
SEEDS = sorted(C.seed.unique())
TC = ["#12666e", "#8f2f4d", "#c06a86", "#e3adba"]      

ARROW = [r"$\;(\uparrow)$"] + [r"$\;(\downarrow)$"] * 3


def acc(col, fam, tgt):
    return np.mean([C[(C.seed == s) & (C.family == fam) & (C.target == tgt)][col].max()
                    for s in SEEDS])


RAW = np.array([[acc("compat_on_U", f, t) for f in FAM] for t, _, _ in TGT])
UHAT = np.array([[acc("compat_on_Uhat", f, t) for f in FAM] for t, _, _ in TGT])
CH = np.array([c for _, c, _ in TGT])
RET = (UHAT - CH[:, None]) / (RAW - CH[:, None]) * 100
DIFF = UHAT - RAW


PANEL = [("retention  (%)", RET, (0, 125), [0, 25, 50, 75, 100, 125],
          "relative to raw $U$"),
         ("$\\Delta$ accuracy", DIFF, (-.62, .12), [-.6, -.4, -.2, 0],
          "absolute change")]


fig = plt.figure(figsize=(3.5, 1.55))
gs = fig.add_gridspec(1, 2, left=.125, right=.995, top=.80, bottom=.24,
                      wspace=.50)
axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
x, W = np.arange(len(FAM)), .80 / len(TGT)
H = []
for n, (ylab, V, ylim, yticks, sub) in enumerate(PANEL):
    ax = axes[n]
    for i in range(len(TGT)):
        b = ax.bar(x + (i - (len(TGT) - 1) / 2) * W, V[i], W * .9,
                   color=TC[i], zorder=2)
        if n == 0:
            H.append(b[0])
    ax.axhline(100 if n == 0 else 0, color="#555", lw=.9,
               ls=(0, (5, 4)) if n == 0 else "-", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(FAMN, fontsize=6.5)
    ax.set_xlabel(f"({'ab'[n]})  {sub}", fontsize=8.5, labelpad=4)
    ax.set_ylabel(ylab, fontsize=8.5)
    ax.set_ylim(*ylim); ax.set_yticks(yticks)
    if n == 1:                     
        ax.set_yticklabels([f"{v:.2f}" for v in yticks])
    ax.set_xlim(-.6, len(FAM) - .4)
    ax.grid(axis="y", color="#eee", lw=.7); ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color("#bbb")

fig.legend(H, [t + a for (_, _, t), a in zip(TGT, ARROW)], ncol=4, loc="upper center",
           bbox_to_anchor=(.5, 1.02), frameon=False, fontsize=8.5,
           handlelength=1.5, columnspacing=1.2)
for ax in axes:
    ax.tick_params(labelsize=7)
out = ROOT / "results/figures/figure6.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=220, bbox_inches="tight")
print("[wrote]", out.name)
print("retention, mean over families:", np.round(RET.mean(1), 0))
print("accuracy change, mean over families:", np.round(DIFF.mean(1), 3))
