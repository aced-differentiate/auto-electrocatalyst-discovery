import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.ensemble import RandomForestRegressor
from adjustText import adjust_text

from autocat.learning.sequential import SequentialLearner

thisdir = os.path.dirname(os.path.abspath(__file__))
sl_data_path = os.path.join(thisdir, "..", "..", "data", "acsl.json")
sl = SequentialLearner.from_json(sl_data_path)
ds = sl.design_space

CANDIDATES_TO_LABEL = [
    "Pt$_1$Au",
    "Re$_1$Au",
    "Fe$_1$Au",
    "Ru$_1$Au",
    "Pd$_1$Au",
    "Ru$_1$Pd",
    "Fe$_1$Ru",
    "Fe$_1$Cu",
    "Mo$_1$Au",
    "Fe$_1$Ni",
]

embedding = np.loadtxt("L1_EMBEDDING.txt")

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
        "font.size": 18,
        "axes.linewidth": 2.0,
        "lines.dashed_pattern": (5, 2.5),
    }
)

fig, ax = plt.subplots()
cmap = mpl.cm.autumn_r

init_data_mask = sl.train_idx_history[0]
ax.scatter(
    embedding[:, 0][init_data_mask],
    embedding[:, 1][init_data_mask],
    zorder=700,
    label="Initial Training Data",
    edgecolors="k",
    c="blue",
)


names = []
for idx, struct in enumerate(ds.design_space_structures):
    if init_data_mask[idx]:
        formula_dict = struct.symbols.formula.count()
        assert len(formula_dict) == 2
        for sp in formula_dict:
            if formula_dict[sp] == 1:
                dopant_species = sp
            else:
                host_species = sp
        names.append(dopant_species + "$_1$" + host_species)

remaining_mask = ~init_data_mask

ax.scatter(
    embedding[:, 0][remaining_mask],
    embedding[:, 1][remaining_mask],
    zorder=500,
    alpha=0.25,
    c="k",
)
ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")

norm = mpl.colors.Normalize(vmin=1, vmax=sl.iteration_count - 1)
fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Iteration Count"
)

texts = []
for idx, name in enumerate(names):
    if name in CANDIDATES_TO_LABEL:
        texts.append(
            ax.text(
                s=" " + name,
                x=embedding[:, 0][init_data_mask][idx],
                y=embedding[:, 1][init_data_mask][idx],
                fontsize="x-small",
                fontweight="heavy",
                zorder=1500,
            ),
        )

adjust_text(
    texts,
    x=embedding[:, 0],
    y=embedding[:, 1],
    ax=ax,
    force_points=(0.005, 0.005),
    extend_text=(1.4, 1.4),
)

fig.savefig("UMAP_initial_L1_LABEL.png", bbox_inches="tight", dpi=200)
# plt.show()
