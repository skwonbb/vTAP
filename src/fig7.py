"""Figure 7 - role assignment and encoder swap."""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.linewidth": .7,
    "xtick.major.size": 0, "ytick.major.size": 0,
})

ROLE = ["Expr.", "Gender", "Race", "Age"]
COL = ["Expr.", "Gend.", "Race", "Age"]   
R = np.array([[105, 20, 9, 25],
              [12, 100, 12, 27],
              [6, 10, 101, 15],
              [9, 21, 18, 105]], float)
TGT = ["$A$", "$A'$", "$B_1$", "$B_2$", "$B_3$"]
ENC = {"CLIP ViT-L/14": [105, 71, 20, 9, 25],
       "DINOv2 ViT-L/14": [100, 45, 19, 13, 28]}

EC = ["#c62828", "#9483c9"]

CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "tap", ["#ffffff", "#cfe3e5", "#7fb4b9", "#2d7c85", "#0d4f57"])


fig = plt.figure(figsize=(3.5, 1.62))
gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1], left=.105, right=.995,
                      top=.84, bottom=.255, wspace=.70)


ax = fig.add_subplot(gs[0, 0])
grid = R
im = ax.imshow(grid, cmap=CMAP, vmin=0, vmax=110, aspect="auto")
for i in range(len(ROLE)):
    for j in range(grid.shape[1]):
        v = grid[i, j]
        ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6,
                color="white" if v > 60 else "#222",
                fontweight="bold" if i == j else "normal")
ax.set_xticks(range(grid.shape[1]))
ax.set_xticklabels(COL, fontsize=6.5)
ax.set_yticks(range(len(ROLE))); ax.set_yticklabels(ROLE, fontsize=6.5)
ax.set_title("measured attribute", fontsize=8, pad=3)
ax.set_xlabel("(a)  varying the role assignment", fontsize=8, labelpad=3)

cb = fig.colorbar(im, ax=ax, fraction=.042, pad=.03)
cb.set_label("retention (%)", fontsize=8, labelpad=2)
cb.ax.tick_params(labelsize=6); cb.outline.set_linewidth(.7)
ax.set_ylabel("permitted task", fontsize=8)


ax2 = fig.add_subplot(gs[0, 1])
x, W = np.arange(len(TGT)), .38
for i, (nm, v) in enumerate(ENC.items()):
    ax2.bar(x + (i - .5) * W, v, W * .9, color=EC[i], label=nm, zorder=2)
ax2.axhline(100, color="#555", lw=.9, ls=(0, (5, 4)), zorder=1)
ax2.set_xticks(x); ax2.set_xticklabels(TGT, fontsize=7)


for t in ax2.get_xticklabels():
    t.set_va("baseline"); t.set_y(-.065)
ax2.set_xlabel("(b)  varying the encoder", fontsize=8, labelpad=3)
ax2.set_ylabel("retention (%)", fontsize=8)

ax2.set_ylim(0, 138); ax2.set_yticks([0, 25, 50, 75, 100])
ax2.set_xlim(-.6, len(TGT) - .4)
ax2.grid(axis="y", color="#eee", lw=.7); ax2.set_axisbelow(True)
ax2.tick_params(labelsize=6.5)
for sp in ax2.spines.values():
    sp.set_color("#bbb")
ax2.legend(fontsize=6, loc="upper right", frameon=False, handlelength=1.4,
           borderaxespad=.2)

for a in (ax, ax2):
    a.xaxis.set_label_coords(.5, -.20)

out = ROOT / "results/figures/figure7.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=220, bbox_inches="tight")
print("[wrote]", out.name)
