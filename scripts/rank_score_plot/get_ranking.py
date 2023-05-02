import os
import json
from typing import Sequence

import numpy as np
from sklearn.ensemble import RandomForestRegressor as SklearnRandomForestRegressor
from autocat.learning.sequential import SequentialLearner
from autocat.learning.sequential import calculate_hhi_scores
from autocat.learning.sequential import calculate_segregation_energy_scores


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


def calculate_activity_confidence(ensemble: Sequence):
    """Calculate confidence that adsorption energy is within activity window"""
    c = 0
    for e in ensemble:
        if abs(e - (-0.8767)) <= 0.3:
            c += 1
    return c / len(ensemble)


thisdir = os.path.dirname(os.path.abspath(__file__))
sl_data_path = os.path.join(thisdir, "..", "..", "data", "acsl.json")
sl = SequentialLearner.from_json(sl_data_path)
ds = sl.design_space

SAAs = [
    "Fe-Zr",
    "Fe-Hf",
    "Cr-Hf",
    "Cr-Zr",
    "V-Zr",
    "Cr-Ti",
    "Ni-Zr",
    "Fe-Ti",
    "Cr-Co",
    "V-Hf",
    "Cr-Ru",
    "Re-Au",
    "Re-Ta",
    "Re-Ag",
    "Re-Pd",
    "Re-Cu",
]
iter_counts = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

# assert len(SAAs) == len(iter_counts)

print("SAA", "Score", "c", "HHI score", "SEG ENER score")

data_dict = {}
for iter_num, saa in zip(iter_counts, SAAs):

    ens_path = os.path.join(
        thisdir, "..", "..", "data", "bee_ensembles", f"{saa}_ens.txt"
    )
    ensemble = np.loadtxt(ens_path)
    # ensemble = np.loadtxt(f"../ensembles_nrr_pipeline_data/{saa}_ens.txt")

    idx = sl.candidate_index_history[iter_num][0]
    hhi = calculate_hhi_scores([ds.design_space_structures[idx]], hhi_type="reserves")
    seg_e_score = calculate_segregation_energy_scores(
        [ds.design_space_structures[idx]], data_source="raban1999"
    )

    c = calculate_activity_confidence(ensemble)
    score = c * hhi * seg_e_score
    print(saa, score, c, hhi, seg_e_score)

    data_dict[saa] = {
        "c": c,
        "hhi": hhi[0],
        "seg_e": seg_e_score[0],
        "score": score[0],
        "iter_num": iter_num,
    }

with open("SCORE_DATA.json", "w") as f:
    json.dump(data_dict, f, indent=4)
