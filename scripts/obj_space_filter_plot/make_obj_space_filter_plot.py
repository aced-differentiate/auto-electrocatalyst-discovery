import os
import json

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams

from autocat.data.segregation_energies import RABAN1999_SEGREGATION_ENERGIES
from autocat.learning.sequential import DesignSpace
from autocat.learning.sequential import calculate_hhi_scores


def get_segregation_energy(structures):
    seg_eners = np.zeros(len(structures))
    for idx, struct in enumerate(structures):
        el_counts = struct.symbols.formula.count()
        assert len(el_counts) == 2
        for el in el_counts:
            if el_counts[el] == 1:
                dsp = el
            else:
                hsp = el
        seg_eners[idx] = RABAN1999_SEGREGATION_ENERGIES[hsp][dsp]
    return seg_eners


thisdir = os.path.dirname(os.path.abspath(__file__))
volc_param_path = os.path.join(thisdir, "..", "..", "data", "raw_volc_m_b.csv")
RAW_VOLC_PARAMS = pd.read_csv(volc_param_path).to_numpy()

ds_data_path = os.path.join(thisdir, "..", "..", "data", "acds.json")
ds = DesignSpace.from_json(ds_data_path)

CANDIDATES_TO_LABEL = [
    "Hf$_\mathregular{1}$Cr",
    "Zr$_\mathregular{1}$Cr",
    "Ti$_\mathregular{1}$Fe",
    "Au$_\mathregular{1}$Re",
    "Ag$_\mathregular{1}$Re",
]


all_HHI = calculate_hhi_scores(ds.design_space_structures, hhi_type="reserves")
all_seg_ener = get_segregation_energy(ds.design_space_structures)


HHI = np.loadtxt("HHI_CANDIDATES.txt")
SEG_ENER = np.loadtxt("SEG_CANDIDATES.txt")
dGN = np.loadtxt("dGN_CANDIDATES.txt")
HHI_init = np.loadtxt("HHI_INIT.txt")
SEG_ENER_init = np.loadtxt("SEG_INIT.txt")
dGN_init = np.loadtxt("dGN_INIT.txt")

with open("FORMULAS.json", "r") as f:
    formulas = json.load(f)

dGN_peak = (RAW_VOLC_PARAMS[1, 1] - RAW_VOLC_PARAMS[0, 1]) / (
    RAW_VOLC_PARAMS[0, 0] - RAW_VOLC_PARAMS[1, 0]
)


def volcano(dG):
    strong_leg = RAW_VOLC_PARAMS[0, 1] + RAW_VOLC_PARAMS[0, 0] * dG
    weak_leg = RAW_VOLC_PARAMS[1, 1] + RAW_VOLC_PARAMS[1, 0] * dG
    return np.minimum(strong_leg, weak_leg)


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

fig, ax = plt.subplots(figsize=(8, 6))
cmap = mpl.cm.summer

ax.scatter(
    all_HHI,
    all_seg_ener,
    c="grey",
    ec="k",
    alpha=0.2
    #    marker="s",
    #    s=55,
)
ax.scatter(HHI_init, SEG_ENER_init, c=abs(dGN_init - dGN_peak), cmap=cmap, ec="k")
ax.scatter(HHI, SEG_ENER, c=abs(dGN - dGN_peak), cmap=cmap, ec="k")
ax.set_xlabel("Normalized HHI")
ax.set_ylabel("Segregation Energy (eV)")
xlims = ax.get_xlim()
ylims = ax.get_ylim()
ax.set_xlim(0.08526725125, 1.0233412837500002)
ax.set_ylim(-2.3449750000000003, 1.5154750000000001)

# segregation energy filter
ax.axhline(0.0, ls="--", c="k")

# hhi filter
ax.axvline(0.8, ls="--", c="k")

x = np.linspace(-2.5, 1.0, 3000)

norm = mpl.colors.Normalize(vmin=0, vmax=1.5)
fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Volcano Peak Distance"
)

# HHI vs SEG annotation
arrowprops = dict(arrowstyle="-|>", color="k", lw=1.5)

ax.annotate(
    text="Hf$_\mathregular{1}$Cr",
    xy=(0.5862403100775193, -1.55),
    xytext=(0.3, -1.5),
    arrowprops=arrowprops,
    fontsize="x-small",
    fontweight="heavy",
)

ax.annotate(
    text="Zr$_\mathregular{1}$Cr",
    xy=(0.5862403100775193, -2.05),
    xytext=(0.4, -2.15),
    arrowprops=arrowprops,
    fontsize="x-small",
    fontweight="heavy",
)

ax.annotate(
    text="Ag$_\mathregular{1}$Re",
    xy=(0.6805555555555555, -1.24),
    xytext=(0.65, -2.0),
    arrowprops=arrowprops,
    fontsize="x-small",
    fontweight="heavy",
)

ax.annotate(
    text="Au$_\mathregular{1}$Re",
    xy=(0.6818475452196382, -1.05),
    xytext=(0.81, -1.0),
    arrowprops=arrowprops,
    fontsize="x-small",
    fontweight="heavy",
)

ax.annotate(
    text="Ti$_\mathregular{1}$Fe",
    xy=(0.894702842377261, -0.39),
    xytext=(0.92, -0.65),
    arrowprops=arrowprops,
    fontsize="x-small",
    fontweight="heavy",
)

plt.savefig("filter_plot.png", bbox_inches="tight", dpi=200)
