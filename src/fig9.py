"""Figure 9 - circuit size split into vTAP and the commitment."""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      
import numpy as np                   
import pandas as pd                  

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.linewidth": .6,
    "xtick.major.size": 0, "ytick.major.size": 2.5, "ytick.major.width": .6,
})

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "figures"
INK, GRID, RULE = "#222", "#ececec", "#bbb"


DARK, LIGHT = "#2E2721", "#B8ADA0"

DS = [384, 512, 768, 1024]
RS = [2, 4, 8, 16]


def load():
    g = pd.read_csv(ROOT / "results" / "zk_scaling.csv")
    g = g[(~g.real_params) & (g.form == "tap_full") & g.stage_failed.isna()]
    gc = pd.read_csv(ROOT / "results" / "zk_commit_groth.csv").set_index("d").constraints
    h = pd.read_csv(ROOT / "results" / "zk_halo2_gated.csv")
    return g, gc, h


def bars(g, gc, h):
    gt, gcm, ht, hcm = [], [], [], []
    for d in DS:
        row = g[(g.d == d) & (g.r == 4)].iloc[0]
        c = int(gc.loc[d])
        gt.append(int(row.constraints) - c)
        gcm.append(c)
        x = h[(h.d == d) & (h.r == 4)].iloc[0]
        ht.append(int(x.tap_advice))
        hcm.append(int(x.pos_advice))
    for r in RS:
        row = g[(g.d == 768) & (g.r == r)].iloc[0]
        c = int(gc.loc[768])
        gt.append(int(row.constraints) - c)
        gcm.append(c)
        x = h[(h.d == 768) & (h.r == r)].iloc[0]
        ht.append(int(x.tap_advice))
        hcm.append(int(x.pos_advice))
    return map(np.array, (gt, gcm, ht, hcm))


def draw(ax, tap, com, title):
    xs = np.arange(8) + np.r_[np.zeros(4), np.full(4, .55)]


    top = (tap + com).max() * 1.06
    gap = top * .014
    ax.bar(xs, tap - gap, width=.68, color=DARK, linewidth=0, zorder=3, label="vTAP")
    ax.bar(xs, com, bottom=tap, width=.68, color=LIGHT, linewidth=0,
           zorder=3, label="commitment")
    ax.axvline(3.775, color=RULE, lw=.6, zorder=2)

    ax.set_xlim(-.62, xs[-1] + .62)
    ax.set_ylim(0, top)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{d}\n{4}" for d in DS] + [f"{768}\n{r}" for r in RS],
                       fontsize=5.4, linespacing=1.45)
    ax.tick_params(axis="x", pad=2)
    ax.tick_params(axis="y", labelsize=6.2, pad=1)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k" if v else "0"))
    ax.set_xlabel(title, fontsize=7.6, labelpad=3)
    ax.grid(axis="y", color=GRID, lw=.5, zorder=0)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color(RULE)


def main():
    gt, gcm, ht, hcm = bars(*load())
    fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.62))
    fig.subplots_adjust(left=.135, right=.995, top=.965, bottom=.44, wspace=.26)

    draw(ax[0], gt, gcm, "(a)  R1CS constraints")
    draw(ax[1], ht, hcm, "(b)  PLONKish advice cells")


    fig.canvas.draw()
    for a in ax:
        bb = a.get_position()
        t = a.get_xticklabels()[0].get_window_extent(fig.canvas.get_renderer())
        y0, y1 = fig.transFigure.inverted().transform(
            [(0, t.y0), (0, t.y1)])[:, 1]
        for y, name in ((y1 - (y1 - y0) * .25, "$d$"), (y0 + (y1 - y0) * .25, "$r$")):
            fig.text(bb.x0 - .009, y, name, ha="right", va="center",
                     fontsize=5.8, color=INK)


    lb = ax[0].xaxis.label.get_window_extent(fig.canvas.get_renderer())
    ly = fig.transFigure.inverted().transform((0, lb.y0))[1]
    lx = (ax[0].get_position().x0 + ax[1].get_position().x1) / 2

    hs = [plt.Rectangle((0, 0), 1, 1, fc=c, lw=0) for c in (DARK, LIGHT)]
    fig.legend(hs, ["vTAP", "commitment"],
               fontsize=6.8, frameon=False, loc="upper center",
               bbox_to_anchor=(lx, ly - .012), ncol=2, handlelength=.95,
               handleheight=.8, columnspacing=1.8, handletextpad=.5,
               borderpad=0)

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "figure9.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[wrote]", p.name)
    for nm, t, c in (("R1CS", gt, gcm), ("PLONKish", ht, hcm)):
        sh = np.round(100 * t / (t + c)).astype(int)
        print(f"  {nm:9} vTAP share  d: {sh[:4]}   r: {sh[4:]}")


if __name__ == "__main__":
    main()
