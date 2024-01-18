import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

plt.style.use("seaborn-v0_8-ticks")
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

with open("SCORE_DATA.json", "r") as f:
    data_dict = json.load(f)

# All
# ranked_SAAs = [saa for saa, d in sorted(data_dict.items(), key=lambda x: x[1].get("score"), reverse=True)]

# Top 5
ranked_SAAs = [
    saa
    for saa, d in sorted(
        data_dict.items(), key=lambda x: x[1].get("score"), reverse=True
    )
][:5]

rs_j = [data_dict[saa].get("score") for saa in ranked_SAAs]
c_j = [data_dict[saa].get("c") for saa in ranked_SAAs]
S_j = [data_dict[saa].get("seg_e") for saa in ranked_SAAs]
C_j = [data_dict[saa].get("hhi") for saa in ranked_SAAs]

# reformat SAA names to Dpt$_1$Hst fmt
fmt_SAA_names = [
    "$_{\mathregular{1}}$".join(saa.split("-")[::-1]) for saa in ranked_SAAs
]

width = 0.25

X_axis = np.arange(len(ranked_SAAs))
X_axis = np.array(range(0, 2 * len(ranked_SAAs), 2))

fig, ax = plt.subplots(figsize=(8, 6))

ax.bar(X_axis, rs_j, width, label="$RS_j$", color="k", edgecolor="k")
ax.bar(
    X_axis + width,
    c_j,
    width,
    label="$c^{\mathrm{active}}_j$",
    color="#0e9594",
    edgecolor="k",
)
ax.bar(X_axis + 2 * width, S_j, width, label="$S_j$", color="#f5dfbb", edgecolor="k")
ax.bar(X_axis + 3 * width, C_j, width, label="$C_j$", color="#f2542d", edgecolor="k")

ax.set_xticks(X_axis + 1.5 * width, fmt_SAA_names)
ax.set_ylim(0.0, 1.0)
ax.legend(
    bbox_to_anchor=(0.525, 1.0),
    loc="upper center",
    frameon=True,
    fancybox=True,
    framealpha=1.0,
    fontsize=14,
    ncol=4,
)

plt.savefig("RANKING_BAR.png", bbox_inches="tight", dpi=200)
