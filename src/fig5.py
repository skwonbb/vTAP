"""Figure 5 - ablation of the construction."""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.linewidth": .7,
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.major.width": .7, "ytick.major.width": .7,
})

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import tables as T                                                 

FAMS = T.FAM_DISPLAY
TGT = ["$A$", "$A'$", "$B_1$", "$B_2$", "$B_3$"]

WHAT = ["expression", "held-out classes", "gender", "race", "age"]
NT = len(TGT)


MC = ["#c62828", "#5b4a8f", "#8f80b8", "#c4bcdd"]


ROWS = [("vTAP",                           "vTAP"),
        ("vTAP w/o $M$",                   "w/o M"),
        ("vTAP w/o $M$, $\\alpha$",        "w/o M, alpha"),
        ("vTAP w/o $M$, $\\alpha$, $S_W$", "w/o M, alpha, SW")]


SPEC = [((39, 150), (50, 75, 100, 125, 150), 50, None),
        ((43, 80), (50, 60, 70, 80), 50, None),
        ((0, 80), (0, 10, 20, 30, 70), None, (30, .35, .075)),
        ((0, 30), (0, 10, 20, 30), None, None),
        ((0, 70), (0, 10, 20, 30, 50, 70), None, (30, .45, .09))]

BANDS = [(0, 2, "Permitted task"), (2, 5, "Protected attributes")]


D = np.stack([T.retention(nm) for _, nm in ROWS])          


def axis_break(ax, frac, w=.028, gap=.020):
    k = dict(transform=ax.transAxes, clip_on=False, zorder=7)
    ax.add_patch(plt.Rectangle((-.004, frac - gap * .8), .008, gap * 1.6,
                               facecolor="white", edgecolor="none", **k))
    xs = np.linspace(-w, w, 60)
    for dy in (-gap / 2, gap / 2):
        ax.plot(xs, frac + dy + gap * .28 * np.sin(2 * np.pi * xs / w),
                color="#111", lw=.8, solid_capstyle="butt", **k)


def squash(th, k):
    def f(y):
        y = np.asarray(y, dtype=float)
        return np.where(y <= th, y, th + (y - th) * k)

    def g(v):
        v = np.asarray(v, dtype=float)
        return np.where(v <= th, v, th + (v - th) / k)
    return f, g


fig = plt.figure(figsize=(11.4, 4.6))

gs = fig.add_gridspec(2, 3, left=.10, right=.99, top=.965, bottom=.095,
                      wspace=.42, hspace=.44)
CELL = [gs[0, 0], gs[0, 1], gs[1, 0], gs[1, 1], gs[1, 2]]

x, W = np.arange(len(FAMS)), .78 / len(ROWS)
H, axes = [], []
for n, cell in enumerate(CELL):
    ax = fig.add_subplot(cell)
    axes.append(ax)
    ylim, yticks, cut, comp = SPEC[n]
    for i in range(len(ROWS)):
        b = ax.bar(x + (i - (len(ROWS) - 1) / 2) * W, D[i, :, n], W * .88,
                   color=MC[i], zorder=2)
        if n == 0:
            H.append(b[0])
    if ylim[1] > 100:
        ax.axhline(100, color="#555", lw=1.0, ls=(0, (5, 4)), zorder=1)
    if comp:
        ax.set_yscale("function", functions=squash(*comp[:2]))
    ax.set_ylim(*ylim)
    ax.set_xlim(-.6, len(FAMS) - .4)
    ax.set_xticks(x); ax.set_xticklabels(FAMS, fontsize=9.5)
    ax.set_xlabel(f"({'abcde'[n]})  {TGT[n]}   ({WHAT[n]})",
                  fontsize=11, labelpad=7)
    if comp:
        f, _ = squash(*comp[:2])
        lo, hi = (float(f(v)) for v in ylim)
        ax.set_yticks(list(yticks))
        axis_break(ax, (float(f(comp[0])) - lo) / (hi - lo) + comp[2])
    elif cut is None:
        ax.set_yticks(list(yticks))
    else:
        ax.set_yticks([ylim[0]] + list(yticks))
        ax.set_yticklabels(["0"] + [str(v) for v in yticks])
        axis_break(ax, (cut - ylim[0]) / (ylim[1] - ylim[0]) / 2)
    ax.grid(axis="y", color="#eee", lw=.7); ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color("#bbb")
    ax.set_ylabel("retention (%)", fontsize=10.5)


fig.canvas.draw()
for a, b, text in BANDS:
    box = [axes[i].get_position() for i in range(a, b)]
    yc = (min(p.y0 for p in box) + max(p.y1 for p in box)) / 2
    fig.text(.008, yc, text, rotation=90, va="center", ha="left",
             fontsize=13.5, fontweight="bold", color="#222")


_lab = axes[4].yaxis.get_label().get_window_extent().transformed(
    fig.transFigure.inverted())
_top = axes[1].get_position()
fig.legend(H, [r[0] for r in ROWS], loc="center left", ncol=1, fontsize=11.5,
           bbox_to_anchor=(_lab.x0, (_top.y0 + _top.y1) / 2 - .055), frameon=False,
           handlelength=1.7, labelspacing=1.0)
out = ROOT / "results/figures/figure5.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight")
print("[wrote]", out.name)
