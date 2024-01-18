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

max_aq_hist = np.loadtxt("MAX_AQ_HIST.txt")

unc_cand_hist = np.loadtxt("UNC_CAND_HIST.txt")

fig, ax = plt.subplots()
c1 = "#2a9d8f"
ax.plot(range(1, len(max_aq_hist) + 1), max_aq_hist, marker="o", c=c1)
ax.set_xlabel("Iteration Count")
ax.set_ylabel("AQ", color=c1)
ax.set_ylim(0.0, 0.5)
ax.set_xticks(list(range(0, len(max_aq_hist) + 1, 2)))
ax.tick_params(axis="y", labelcolor=c1)
ax.annotate(
    "",
    xy=(4.5, 0.3),
    xytext=(0.5, 0.3),
    arrowprops=dict(arrowstyle="<|-", color=c1, lw=1.5),
)

ax2 = ax.twinx()
c2 = "#e76f51"
ax2.set_ylabel("Candidate $\sigma_{j,\mathrm{N}}^{\mathrm{pred}}$ (eV)", color=c2)
ax2.plot(range(1, len(max_aq_hist) + 1), unc_cand_hist, marker="o", c=c2)
ax2.tick_params(axis="y", labelcolor=c2)
ax2.annotate(
    "",
    xy=(16.5, 0.3),
    xytext=(13.5, 0.3),
    arrowprops=dict(arrowstyle="-|>", color=c2, lw=1.5),
)

fig.savefig("SCORES_PLOT.png", bbox_inches="tight", dpi=200)
