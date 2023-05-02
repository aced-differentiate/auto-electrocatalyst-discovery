import os
import json

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
from adjustText import adjust_text


thisdir = os.path.dirname(os.path.abspath(__file__))
volc_param_path = os.path.join(thisdir, "..", "..", "data", "raw_volc_m_b.csv")
RAW_VOLC_PARAMS = pd.read_csv(volc_param_path).to_numpy()

CANDIDATES_TO_LABEL = [
    "Hf$_\mathregular{1}$Cr",
    "Zr$_\mathregular{1}$Cr",
    "Ti$_\mathregular{1}$Fe",
    "Au$_\mathregular{1}$Re",
    "Ag$_\mathregular{1}$Re",
]


HHI = np.loadtxt("HHI_CANDIDATES.txt")
SEG_ENER = np.loadtxt("SEG_CANDIDATES.txt")
dGN = np.loadtxt("dGN_CANDIDATES.txt")
HHI_init = np.loadtxt("HHI_INIT.txt")
SEG_ENER_init = np.loadtxt("SEG_INIT.txt")
dGN_init = np.loadtxt("dGN_INIT.txt")

with open("FORMULAS.json", "r") as f:
    formulas = json.load(f)


def volcano(dG):
    strong_leg = RAW_VOLC_PARAMS[0, 1] + RAW_VOLC_PARAMS[0, 0] * dG
    weak_leg = RAW_VOLC_PARAMS[1, 1] + RAW_VOLC_PARAMS[1, 0] * dG
    return np.minimum(strong_leg, weak_leg)


plt.style.use("seaborn-ticks")
rcParams.update(
    {
        "font.family": "sans-serif",
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "xtick.major.size": 7,
        "ytick.major.size": 7,
        "xtick.major.width": 2.0,
        "ytick.major.width": 2.0,
        "font.size": 20,
        "axes.linewidth": 2.0,
        "lines.dashed_pattern": (5, 2.5),
        "lines.markersize": 10,
        "lines.linewidth": 3,
        "lines.markeredgewidth": 2,
        "lines.markeredgecolor": "k",
        "legend.fontsize": 16,
        "legend.frameon": True,
    }
)

fig, (axins, ax) = plt.subplots(1, 2, figsize=(16, 6))
cmap = mpl.cm.autumn_r

ax.scatter(HHI_init, SEG_ENER_init, c="grey", alpha=0.5)
ax.scatter(HHI, SEG_ENER, c=range(len(HHI)), cmap=cmap, ec="k")
ax.set_xlabel("Normalized HHI")
ax.set_ylabel("Segregation Energy (eV)")
ax.text(
    0.3,
    0.6,
    "Stable",
    c="k",
    rotation=90,
    bbox=dict(boxstyle="larrow,pad=0.3", lw=2, fc="mediumseagreen"),
    transform=ax.transAxes,
)
ax.text(
    0.3,
    0.9,
    "Economical",
    c="k",
    bbox=dict(boxstyle="rarrow,pad=0.3", lw=2, fc="mediumseagreen"),
    transform=ax.transAxes,
)
xlims = ax.get_xlim()
ylims = ax.get_ylim()
ax.scatter(
    [0.9807015550000001],
    [-2.1695],
    marker="*",
    s=500,
    c="mediumseagreen",
    ec="k",
    lw=1.5,
)
ax.set_xlim(0.08526725125, 1.0233412837500002)
ax.set_ylim(-2.3449750000000003, 1.5154750000000001)

x = np.linspace(-2.5, 1.0, 3000)
axins.plot(x, volcano(x), c="k")
axins.scatter(dGN, volcano(dGN), zorder=100, ec="k", c=range(len(dGN)), cmap=cmap)
axins.scatter(dGN_init, volcano(dGN_init), zorder=50, c="grey", alpha=0.5)
axins.set_xlim(-2.5, 1)
axins.set_ylim(-1.75, -0.5)
axins.set_ylabel("U$_\mathrm{L}$ (V)")
axins.set_xlabel("$\Delta G_{\mathrm{N}}$ (eV)")
axins.axvspan(-1.1767, -0.5767, facecolor="mediumseagreen", alpha=0.3)

norm = mpl.colors.Normalize(vmin=1, vmax=len(dGN))
fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Iteration Count"
)

# Volcano Annotation
volcano_texts = []
for idx, name in enumerate(formulas):
    if name in CANDIDATES_TO_LABEL:
        volcano_texts.append(
            axins.text(
                s=" " + name,
                x=dGN[idx],
                y=volcano(dGN[idx]),
                fontsize="x-small",
                fontweight="heavy",
            ),
        )
adjust_text(
    volcano_texts,
    x=x,
    y=volcano(x),
    expand_text=(1.5, 1.5),
    # expand_points=(2.4, 2.4),
    expand_points=(3.0, 3.0),
    ax=axins,
    force_points=(0.0005, 0.0005),
    arrowprops=dict(arrowstyle="-|>", color="k", lw=1.5),
)

# HHI vs SEG annotation
hhi_seg_texts = []
for idx, name in enumerate(formulas):
    if name in CANDIDATES_TO_LABEL:
        hhi_seg_texts.append(
            ax.text(
                s=" " + name,
                x=HHI[idx],
                y=SEG_ENER[idx],
                fontsize="x-small",
                fontweight="heavy",
            ),
        )
adjust_text(
    hhi_seg_texts,
    x=np.concatenate((HHI, HHI_init)),
    y=np.concatenate((SEG_ENER, SEG_ENER_init)),
    expand_points=(2.8, 3.0),
    expand_text=(1.5, 3.5),
    ax=ax,
    force_points=(0.1, 0.01),
    force_text=(0.7, 1.0),
    arrowprops=dict(arrowstyle="-|>", color="k", lw=1.5),
)


plt.savefig("search_trajectory.png", bbox_inches="tight", dpi=200)
