"""Figure 2 - what one embedding gives up: the permitted task and three more."""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.linewidth": .6,
    "xtick.major.size": 2.5, "xtick.major.width": .6, "ytick.major.size": 0,
})

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "figures"
sys.path.insert(0, str(ROOT / "src"))
import tables as T                                                 # noqa: E402

INK, GRID, RULE = "#222", "#ececec", "#bbb"
ROWS = [(0, "expression", "#12666e"), (2, "gender", "#8f2f4d"),
        (3, "race", "#a8536b"), (4, "age", "#c07d92")]


def main():
    acc = T.acc("raw U")

    fig, ax = plt.subplots(figsize=(2.55, 1.26))
    fig.subplots_adjust(left=.175, right=.995, top=.97, bottom=.29)

    for k, (j, name, col) in enumerate(ROWS):
        y = len(ROWS) - 1 - k
        all_v = acc[:, j]
        v = all_v.max()
        ax.barh(y, v, height=.56, color=col, linewidth=0, zorder=2)
        ax.text(.016, y, f"{v:.4f}", va="center_baseline", ha="left",
                fontsize=7.2, color="white", fontweight="bold", zorder=5)

    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([n for _, n, _ in ROWS][::-1], fontsize=7.8)
    ax.tick_params(axis="y", pad=2)
    ax.tick_params(axis="x", labelsize=6.4, pad=2)
    ax.set_ylim(-.55, len(ROWS) - .45)
    ax.set_xlim(0, 1.00)
    ax.set_xticks([0, .25, .5, .75, 1.0])
    ax.set_xlabel("balanced accuracy", fontsize=7.2, labelpad=2)
    ax.grid(axis="x", color=GRID, lw=.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(RULE)
    ax.spines["left"].set_visible(False)

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "figure2.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[wrote]", p.name)
    for j, name, _ in ROWS:
        print(f"  {name:11} {acc[:, j].max():.4f}")


if __name__ == "__main__":
    main()
