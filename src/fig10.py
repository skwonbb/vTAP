"""Figure 10 - deployment cost of the two proof systems."""
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
OUT = ROOT / "results" / "figures"
INK, GRID, RULE = "#222", "#ececec", "#bbb"


G16, H2, H2K = "#00A24E", "#1E40BD", "#8FA5EC"

DS = [384, 512, 768, 1024]
RS = [2, 4, 8, 16]


GROTH_PROOF = 128


def load():
    g = pd.read_csv(ROOT / "results" / "zk_scaling.csv")


    g = g[(~g.real_params) & (g.form == "tap_full") & g.stage_failed.isna()]
    h = pd.read_csv(ROOT / "results" / "zk_halo2_gated.csv")
    return g, h


def series(g, h, axis, vals):
    gg = [g[(g.d == v) & (g.r == 4)].iloc[0] if axis == "d"
          else g[(g.d == 768) & (g.r == v)].iloc[0] for v in vals]
    hh = [h[(h.d == v) & (h.r == 4)].iloc[0] if axis == "d"
          else h[(h.d == 768) & (h.r == v)].iloc[0] for v in vals]
    return gg, hh


PANELS = [
    ("setup time", "setup", "s", True,
     lambda x: x.setup_s, lambda x: x.srs_s + x.keygen_s, lambda x: x.srs_s),
    ("proving time", "prove", "s", False,
     lambda x: x.prove_s, lambda x: x.prove_s, None),
    ("verification time", "verify", "ms", False,
     lambda x: x.verify_ms, lambda x: x.verify_ms, None),
    ("proving key size", "pk", "MB", False,
     lambda x: x.zkey_bytes / 1e6, lambda x: x.pk_bytes / 1e6, None),
    ("verification key size", "vk", "KB", True,
     lambda x: x.vkey_bytes / 1e3, lambda x: x.vk_bytes / 1e3, None),
    ("proof size", "proof", "B", True,
     lambda x: GROTH_PROOF, lambda x: x.proof_bytes, None),
]


def draw(ax, gg, hh, spec, labels, tag, xname):
    title, short, unit, log, fg, fh, fh_lo = spec
    gv = np.array([fg(x) for x in gg], float)
    hv = np.array([fh(x) for x in hh], float)
    xs = np.arange(8) + np.r_[np.zeros(4), np.full(4, .6)]

    if log:
        floor = 10 ** (np.floor(np.log10(np.min(np.r_[gv, hv]))) - .35)
        ax.set_yscale("log")
        ax.set_ylim(floor, np.max(np.r_[gv, hv]) * 10 ** .26)
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=5))
        ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(
            subs=tuple(np.arange(2, 10) * 1.0), numticks=40))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    else:
        floor = 0.0
        ax.set_ylim(0, np.max(np.r_[gv, hv]) * 1.08)

    ax.bar(xs[:4], gv - floor, bottom=floor, width=.82, color=G16,
           edgecolor="white", linewidth=.4, zorder=3)
    if fh_lo is None:
        ax.bar(xs[4:], hv - floor, bottom=floor, width=.82, color=H2,
               edgecolor="white", linewidth=.4, zorder=3)
    else:


        mid = np.array([fh_lo(x) for x in hh], float)
        ax.bar(xs[4:], mid - floor, bottom=floor, width=.82, color=H2,
               edgecolor="white", linewidth=.4, zorder=3)
        ax.bar(xs[4:], hv - mid, bottom=mid, width=.82, color=H2K,
               edgecolor="white", linewidth=.4, zorder=3)

    ax.axvline(3.8, color=RULE, lw=.6, zorder=2)
    ax.set_xlim(-.7, xs[-1] + .7)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels * 2, fontsize=5.8, rotation=90)
    ax.tick_params(axis="x", pad=1)
    ax.tick_params(axis="y", labelsize=6.4, pad=1)
    ax.set_ylabel(f"{short} ({unit})", fontsize=7.2, labelpad=1.5)
    ax.set_xlabel(f"${xname}$\n({tag})  {title} ({unit})", fontsize=7.2, labelpad=1.5)
    ax.grid(axis="y", color=GRID, lw=.5, zorder=0)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color(RULE)


def main():
    g, h = load()
    fig, ax = plt.subplots(2, 6, figsize=(7.16, 2.78))
    fig.subplots_adjust(left=.058, right=.997, top=.99, bottom=.20,
                        wspace=.46, hspace=.72)

    for row, (axis, vals) in enumerate((("d", DS), ("r", RS))):
        gg, hh = series(g, h, axis, vals)
        labels = [str(v) for v in vals]
        for col, spec in enumerate(PANELS):
            draw(ax[row][col], gg, hh, spec, labels,
                 "abcdefghijkl"[row * 6 + col], axis)


    hs = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in (G16, H2)]
    fig.legend(hs, ["Groth16", "halo2-KZG"], fontsize=6.8, frameon=False,
               loc="lower center", bbox_to_anchor=(.5, -.012), ncol=2,
               handlelength=1.0, handleheight=.8, columnspacing=2.0,
               borderpad=0, handletextpad=.5)

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "figure10.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[wrote]", p.name)


if __name__ == "__main__":
    main()
