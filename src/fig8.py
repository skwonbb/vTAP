"""Figure 8 - arithmetization error, magnitude and composition."""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      
import numpy as np                   
import pandas as pd                  

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.linewidth": .6,
    "xtick.major.size": 0, "ytick.major.size": 2.5,
    "ytick.major.width": .6, "ytick.minor.size": 0,
})

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "results" / "figures"
INK, GRID, RULE = "#222", "#ececec", "#bbb"


TOTAL = "#A8003A"
STAGE = [(1, r"$U\!\to\!\tilde U$", "#FFCBD8"),
         (2, r"$\tilde U\!\to\!\tilde Z$", "#F76A85"),
         (3, r"$\tilde Z\!\to\!\hat U_f$", "#DC0036")]
DEPLOY_F = 16


def main():
    a = pd.read_csv(ROOT / "results" / "quant_error_ablation.csv")
    cum = a.pivot(index="f", columns="stage", values="cum_rel_l2").sort_index()
    inc = a.pivot(index="f", columns="stage", values="inc_rel_l2").sort_index()
    cum, inc = cum.loc[4:], inc.loc[4:]
    fs = cum.index.to_numpy()

    fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.42))
    fig.subplots_adjust(left=.155, right=.995, top=.965, bottom=.44, wspace=.42)
    BW = 1.15


    FLOOR = 5e-8
    ax[0].set_yscale("log")
    ax[0].bar(fs, cum[3].to_numpy() - FLOOR, bottom=FLOOR, width=BW,
              color=TOTAL, linewidth=0, zorder=3)
    ax[0].set_ylim(FLOOR, .35)
    ax[0].yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=5))
    ax[0].yaxis.set_minor_locator(matplotlib.ticker.LogLocator(
        subs=tuple(np.arange(2, 10) * 1.0), numticks=40))
    ax[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[0].set_ylabel(r"$\varepsilon$  (relative $L_2$)", fontsize=7.0, labelpad=1.5)


    share = inc.div(cum[3], axis=0) * 100
    GAP = 0.8
    bottom = np.zeros(len(fs))
    for st, lab, col in STAGE:
        h = share[st].to_numpy()
        cut = h - GAP if st != STAGE[-1][0] else h
        ax[1].bar(fs, cut, bottom=bottom, width=BW, color=col, label=lab,
                  linewidth=0, zorder=3)
        bottom += h
    ax[1].set_ylim(0, 100)
    ax[1].set_yticks([0, 50, 100])
    ax[1].set_ylabel(r"share of $\varepsilon$ (\%)".replace("\\%", "%"),
                     fontsize=7.0, labelpad=1.5)

    for axis in ax:
        axis.set_xlabel("$f$", fontsize=7.6, labelpad=2)
        axis.set_xticks(fs)
        axis.set_xticklabels([str(x) for x in fs], fontsize=5.4)
        axis.tick_params(axis="x", pad=2)
        axis.tick_params(axis="y", labelsize=6.2, pad=1)
        axis.set_xlim(fs[0] - 1.4, fs[-1] + 1.4)
        axis.grid(axis="y", color=GRID, lw=.5, zorder=0)
        axis.set_axisbelow(True)
        for sp in axis.spines.values():
            sp.set_color(RULE)


    fig.canvas.draw()
    lb = ax[0].xaxis.label.get_window_extent(fig.canvas.get_renderer())
    ly = fig.transFigure.inverted().transform((0, lb.y0))[1]
    for axis, name in zip(ax, ("(a)  magnitude", "(b)  composition")):
        bb = axis.get_position()
        fig.text((bb.x0 + bb.x1) / 2, ly - .022, name, ha="center", va="top",
                 fontsize=7.6, color=INK)


    lx = (ax[0].get_position().x0 + ax[1].get_position().x1) / 2
    hs = [plt.Rectangle((0, 0), 1, 1, fc=c, lw=0)
          for c in (TOTAL, *(c for _, _, c in STAGE))]
    fig.legend(hs, [r"total $\varepsilon$", *(l for _, l, _ in STAGE)],
               fontsize=6.4, frameon=False, loc="upper center",
               bbox_to_anchor=(lx, ly - .105), ncol=4, handlelength=.9,
               handleheight=.75, columnspacing=1.0, handletextpad=.4,
               borderpad=0)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    p = OUTDIR / "figure8.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[wrote]", p.name)


if __name__ == "__main__":
    main()
