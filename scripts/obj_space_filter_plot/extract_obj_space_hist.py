import os
import sys
import json

# Adds drivers to path since we use a couple of the functions
# defined there
thisdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(thisdir, "..", "drivers"))
from sl_driver import get_binding_energy
from sl_driver import pif_to_atoms

import numpy as np
import ase.db
from ase import Atom
from pypif import pif
from sklearn.ensemble import RandomForestRegressor as SklearnRandomForestRegressor
from autocat.learning.sequential import SequentialLearner
from autocat.learning.sequential import calculate_hhi_scores
from autocat.data.segregation_energies import SEGREGATION_ENERGIES


class RandomForestRegressor(SklearnRandomForestRegressor):
    """Custom wrapper for sklearn RandomForest regressor which returns random
    forest predictions along with std in the tree estimates."""

    def predict(self, X, return_std=True):
        preds = SklearnRandomForestRegressor.predict(self, X)
        est_preds = np.empty((len(X), len(self.estimators_)))
        # loop over each tree in the forest and use it to make a prediction
        for ind, est in enumerate(self.estimators_):
            est_preds[:, ind] = est.predict(X)
        # assert np.allclose(np.mean(est_preds, axis=1), preds)
        if return_std:
            return preds, np.std(est_preds, axis=1)
        else:
            return preds



CANDIDATES_TO_LABEL = [
    "Hf$_\mathregular{1}$Cr",
    "Zr$_\mathregular{1}$Cr",
    "Ti$_\mathregular{1}$Fe",
    "Au$_\mathregular{1}$Re",
    "Ag$_\mathregular{1}$Re",
]

db_path = os.path.join(thisdir, "..", "..", "data", "dft_data.db")
dft_db = ase.db.connect(db_path, type="json")

sl_data_path = os.path.join(thisdir, "..", "..", "data", "acsl.json")
sl = SequentialLearner.from_json(sl_data_path)

# get candidate formulas
cand_formulas = []
for cand_idx in sl.candidate_index_history:
    candidate_struct = sl.design_space.design_space_structures[cand_idx[0]]
    candidate_struct.append(Atom("N"))
    form = candidate_struct.get_chemical_formula()
    cand_formulas.append(form)

# collect all data
HHI_init = []
SEG_ENER_init = []
dGN_init = []
sl_idx = []
HHI_unsorted = []
SEG_ENER_unsorted = []
dGN_unsorted = []
formulas_unsorted = []

print("Volcano peak at dGN = -0.8767 eV")

for i, row in enumerate(dft_db.select(adsorbate="N")):
    # get binding energy
    pifo = pif.loado(row.data)
    be = get_binding_energy(pifo)

    candidate_struct = pif_to_atoms(pifo)
    # rm adsorbate N
    assert candidate_struct[-1].symbol == "N"
    del candidate_struct[-1]

    # get hhi
    hhi = calculate_hhi_scores([candidate_struct], hhi_type="reserves")

    # get segregation energy
    names = row.data.names[0].split(" ")
    hsp = names[0]
    dsp = names[1]
    seg_e = SEGREGATION_ENERGIES["raban1999"][hsp][dsp]

    if row.data.get("chemicalFormula") in cand_formulas:
        # is a candidate
        formulas_unsorted.append(dsp + "$_\mathregular{1}$" + hsp)
        HHI_unsorted.append(hhi)
        SEG_ENER_unsorted.append(seg_e)
        dGN_unsorted.append(be)
        sl_idx.append(cand_formulas.index(row.data.get("chemicalFormula")))
        # print(be, row.data.get("chemicalFormula"))
        if be > -1.1767 and be < -0.5767:
            print(
                row.data.get("chemicalFormula"),
                be,
                hhi[0],
                seg_e,
                cand_formulas.index(row.data.get("chemicalFormula")),
            )
    else:
        # is from initial data
        HHI_init.append(hhi)
        SEG_ENER_init.append(seg_e)
        dGN_init.append(be)


# sort by sl index
HHI = [hhi for _, hhi in sorted(zip(sl_idx, HHI_unsorted))]
SEG_ENER = [se for _, se in sorted(zip(sl_idx, SEG_ENER_unsorted))]
dGN = [dgn for _, dgn in sorted(zip(sl_idx, dGN_unsorted))]
formulas = [formula for _, formula in sorted(zip(sl_idx, formulas_unsorted))]


HHI = np.array(HHI)
SEG_ENER = np.array(SEG_ENER)
dGN = np.array(dGN)
HHI_init = np.array(HHI_init)
SEG_ENER_init = np.array(SEG_ENER_init)
dGN_init = np.array(dGN_init)

np.savetxt("HHI_CANDIDATES.txt", HHI)
np.savetxt("dGN_CANDIDATES.txt", dGN)
np.savetxt("SEG_CANDIDATES.txt", SEG_ENER)
np.savetxt("HHI_INIT.txt", HHI_init)
np.savetxt("dGN_INIT.txt", dGN_init)
np.savetxt("SEG_INIT.txt", SEG_ENER_init)

with open("FORMULAS.json", "w") as f:
    json.dump(formulas, f)
