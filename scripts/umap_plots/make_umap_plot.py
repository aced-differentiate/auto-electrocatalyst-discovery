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
    "Hf$_\mathregular{1}$Cr",
    "Zr$_\mathregular{1}$Cr",
    "Ti$_\mathregular{1}$Fe",
    "Au$_\mathregular{1}$Re",
    "Ag$_\mathregular{1}$Re",
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
)

candidate_index_history = sl.candidate_index_history[:-1]

cand_mask = np.zeros(len(ds), dtype=bool)
cand_embed = np.zeros((len(candidate_index_history), 3))
names = []
for i, c in enumerate(candidate_index_history):
    cand_mask[c[0]] += 1

    # get name for labelling
    formula_dict = ds.design_space_structures[c[0]].symbols.formula.count()
    assert len(formula_dict) == 2
    for sp in formula_dict:
        if formula_dict[sp] == 1:
            dopant_species = sp
        else:
            host_species = sp
    names.append(dopant_species + "$_\mathregular{1}$" + host_species)

    cand_embed[i] = (embedding[c[0], 0], embedding[c[0], 1], i + 1)

ax.scatter(
    cand_embed[:, 0],
    cand_embed[:, 1],
    c=cand_embed[:, 2],
    cmap=cmap,
    edgecolors="k",
    zorder=1000,
)


remaining_mask = ~cand_mask + ~init_data_mask

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
                x=cand_embed[idx, 0],
                y=cand_embed[idx, 1],
                fontsize="x-small",
                fontweight="heavy",
                zorder=1500,
            ),
        )

adjust_text(
    texts,
    x=embedding[:, 0],
    y=embedding[:, 1],
    expand_points=(1.2, 1.2),
    ax=ax,
    force_points=(1.0, 2.0),
    force_text=(0.7, 0.8),
    arrowprops=dict(arrowstyle="-|>", color="k", lw=1.5),
)

fig.savefig("UMAP.png", bbox_inches="tight", dpi=200)
# plt.show()
