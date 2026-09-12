"""Figure 4 - retention per method, and the dense sweep."""
import pathlib
import sys
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.linewidth": .7,
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.major.width": .7, "ytick.major.width": .7,
})

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import tables as T                                                 


KEEP = [("vTAP",         "vTAP",   None),
        ("VIB utility", "VIB",   "utility end"),
        ("VIB privacy", "VIB",   "privacy end"),
        ("ADV utility", "ADV",   "utility end"),
        ("ADV privacy", "ADV",   "privacy end"),
        ("LEACE",       "LEACE", None)]


DIM = T.DIM
fams = T.FAM_DISPLAY
NT = len(T.TARGETS)
TGT = ["$A$", "$A'$", "$B_1$", "$B_2$", "$B_3$"]

LEG = [t + (r"$\;(\uparrow)$" if i < NT - 3 else r"$\;(\downarrow)$")
       for i, t in enumerate(TGT)]

keep = [k for k, _, _ in KEEP]
data = {k: T.retention(k) for k in keep}
SUB = {k: s for k, _, s in KEEP if s}
RENAME = {k: d for k, d, _ in KEEP}


TC = ["#12666e", "#5fa8ae", "#8f2f4d", "#c06a86", "#e3adba"][:NT]


S = pd.read_csv(ROOT / "results/sweep_dense.csv")
P = pd.read_csv(ROOT / "results/probe_runs_rafdb.csv")
TT = ["A_kept", "Aprime_heldout", "B_gender", "B_race", "B_age"]
CH = dict(zip(TT, [.2, .5, .5, 1 / 3, .2]))
BASE = {t: P[(P.method == "raw_U") & (P.target == t)]["test_bal"].max() for t in TT}
for t in TT:
    S["r_" + t] = (S[t] - CH[t]) / (BASE[t] - CH[t]) * 100
S["leak"] = S[["r_B_gender", "r_B_race", "r_B_age"]].mean(1)
S["util"] = S[["r_A_kept", "r_Aprime_heldout"]].mean(1)


tap = T.collapse("vTAP")
TL, TU = tap[2:].mean(), (tap[0] + tap[1]) / 2


CS = {"vib": "#9aa856", "adv": "#9483c9"}     
SNAME = {"vib": "VIB", "adv": "ADV"}
U0, U1, L1 = 56, 94, 55        
RED = "#c62828"


def frontier(g):
    q = g.sort_values("leak")
    fy, u = q.leak.values, q.util.values
    hi = np.maximum.accumulate(u)
    rec = np.r_[True, np.diff(hi) > 0]             
    ky, kx = fy[rec], hi[rec]
    gl = np.linspace(ky[0], fy[-1], 400)
    return gl, PchipInterpolator(ky, kx, extrapolate=True)(np.clip(gl, ky[0], ky[-1]))


fig = plt.figure(figsize=(12.6, 6.0))

gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, .0, 1.02],
                      left=.045, right=.985, top=.855, bottom=.12,
                      wspace=.26, hspace=.55)
ABC = "abcdefg"

x, W = np.arange(len(fams)), .78 / NT
H, ax0, bax = [], None, []
for n, nm in enumerate(keep):
    ax = fig.add_subplot(gs[n // 3, n % 3])
    ax0 = ax0 or ax
    ax.axhline(100, color="#555", lw=1.0, ls=(0, (5, 4)), zorder=1)
    for t in range(NT):
        b = ax.bar(x + (t - (NT - 1) / 2) * W, data[nm][:, t], W * .88,
                   color=TC[t], zorder=2)
        if n == 0:
            H.append(b[0])
    ax.set_xticks(x); ax.set_xticklabels(fams, fontsize=9.5)
    tag = (f"{SUB[nm]}, $d$={DIM[nm]}" if nm in SUB else f"$d$={DIM[nm]}")
    ax.set_xlabel(f"({ABC[n]})  {RENAME.get(nm, nm)}  ({tag})", fontsize=11,
                  fontweight="bold" if nm == "vTAP" else "normal", labelpad=7)
    ax.set_ylim(0, 152); ax.set_yticks([0, 50, 100, 150])
    ax.set_xlim(-.6, len(fams) - .4)
    ax.grid(axis="y", color="#eee", lw=.7); ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color("#bbb")
    if nm == "vTAP":
        for s in ax.spines.values():
            s.set_color("#111"); s.set_linewidth(1.4)
    if n >= 3:
        bax.append(ax)                
    if n % 3 == 0:
        ax.set_ylabel("retention (%)", fontsize=10.5)

fig.legend(H, LEG, ncol=NT, fontsize=10.5, loc="upper center",
           bbox_to_anchor=(.37, .925), frameon=False, handlelength=1.5,
           columnspacing=2.0)

axs = fig.add_subplot(gs[:, 4])


L0 = min(S[S.kind == k].leak.min() for k in ("vib", "adv"))
_p = axs.get_position()
_h = _p.height - .052               
_cut = _h * (L0 - 13) / (L1 - 13)   
axs.set_position([_p.x0 - .022, _p.y0 + .052 + _cut, _p.width, _h - _cut])
for ax in bax:
    q = ax.get_position()
    ax.set_position([q.x0, q.y0 + _cut, q.width, q.height])
axd = bax[0]                        
edge = {}
for kind in ("vib", "adv"):
    g = S[S.kind == kind]
    gl, fx = frontier(g)
    axs.scatter(g.util, g.leak, s=6, color=CS[kind], alpha=.5, lw=0, zorder=2)
    axs.plot(fx, gl, color=CS[kind], lw=2.0, zorder=3, label=SNAME[kind])

    edge[kind] = np.interp(TL, gl, fx)

axs.scatter(TU, TL, marker="o", s=95, facecolor=RED, edgecolor="white",
            lw=1.3, zorder=6, label="vTAP")
axs.plot([TU, TU], [L0, TL], color=RED, lw=.7, ls=(0, (2, 3)), zorder=2)
axs.plot([U0, TU], [TL, TL], color=RED, lw=.7, ls=(0, (2, 3)), zorder=2)
best = max(edge.values())
axs.plot([best, TU], [TL, TL], color=RED, lw=1.1, solid_capstyle="butt", zorder=5)
axs.text((best + TU) / 2, TL - 1.0, "$+%.1f$ pp" % (TU - best),
         fontsize=11, color=RED, ha="center", va="top")

axs.set_xlabel("task retention  (%)", fontsize=10.5)
axs.set_ylabel("protected-attribute leakage  (%)", fontsize=10.5)
axs.set_xlim(U0, U1); axs.set_ylim(L0, L1)
axs.set_xticks([60, 70, 80, 90]); axs.set_yticks([20, 30, 40, 50])
axs.grid(color="#ececec", lw=.6, zorder=0); axs.set_axisbelow(True)
for s in axs.spines.values():
    s.set_color("#999")


_px = axs.get_position()
fig.legend(*axs.get_legend_handles_labels(), ncol=3, fontsize=10,
           loc="upper center", bbox_to_anchor=((_px.x0 + _px.x1) / 2, .925),
           frameon=False, handlelength=1.6, columnspacing=1.4)


fig.canvas.draw()
yc = axd.xaxis.get_label().get_window_extent().transformed(
    fig.transFigure.inverted()).y0
px = axs.get_position()
fig.text((px.x0 + px.x1) / 2, yc, "(g)  dense sweep", ha="center",
         va="bottom", fontsize=11)

out = ROOT / "results/figures/figure4.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight")   
print("[wrote]", out.name, "| panels:", keep)
print(f"vTAP  leakage {TL:.1f}  utility {TU:.1f}   band edge {best:.1f}  (+{TU-best:.1f}pp)")
