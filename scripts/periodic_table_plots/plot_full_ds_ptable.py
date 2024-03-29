import os
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.colors
from ptplotter import ElementDataPlotter
from ptplotter import elt_data
from matplotlib import rcParams

plt.style.use("seaborn-v0_8-ticks")
rcParams.update(
    {
        "font.family": "sans-serif",
        "font.weight": "heavy",
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "xtick.major.size": 7,
        "ytick.major.size": 7,
        "xtick.major.width": 2.0,
        "ytick.major.width": 2.0,
        "axes.linewidth": 2.0,
        "lines.dashed_pattern": (5, 2.5),
        "lines.markersize": 10,
        "lines.linewidth": 3,
        "lines.markeredgewidth": 2,
        "lines.markeredgecolor": "k",
        "patch.force_edgecolor": True,
    }
)

thisdir = os.path.dirname(os.path.abspath(__file__))
ele_data_path = os.path.join(thisdir, "..", "..", "data", "ELEMENTS.json")
with open(ele_data_path, "r") as f:
    ds_ele = json.load(f)

init_host_ele = ["Au", "Cu", "Fe", "Ni", "Ru", "Pd"]
init_dop_ele = ["Fe", "Mo", "Pd", "Pt", "Re", "Ru", "Ni"]


def custom_data():
    return defaultdict(custom_data)


data = custom_data()
for elt in elt_data:
    if elt in ds_ele:
        data[elt]["full_ds"] = True
    else:
        data[elt]["full_ds"] = False

    if elt in init_host_ele:
        data[elt]["init_host"] = True
    else:
        data[elt]["init_host"] = False

    if elt in init_dop_ele:
        data[elt]["init_dop"] = True
    else:
        data[elt]["init_dop"] = False


def in_ds(elt):
    """Design Space"""
    if elt["full_ds"]:
        return 1
    else:
        return 0


def init_host(elt):
    """Host Species \n in Initial Training Set"""
    if elt["init_host"]:
        return 1
    else:
        return 0


def init_dop(elt):
    """Dopant Species in Initial Training Set"""
    if elt["init_dop"]:
        return 1
    else:
        return 0


cm = matplotlib.colors.LinearSegmentedColormap.from_list("", ["tomato", "limegreen"])

epd = ElementDataPlotter(data=data)
epd.ptable(
    [in_ds],
    colorbars=False,
    cmaps=[cm],
    font={"color": "k", "size": "large"},
    elem_labels=True,
    guide=False,
)
plt.savefig("FULL_DESIGN_SPACE.png", bbox_inches="tight", dpi=200)
plt.show()
