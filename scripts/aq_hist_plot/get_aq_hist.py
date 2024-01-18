import os

import numpy as np
from sklearn.ensemble import RandomForestRegressor as SklearnRandomForestRegressor
from autocat.learning.sequential import SequentialLearner


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


thisdir = os.path.dirname(os.path.abspath(__file__))
sl_data_path = os.path.join(thisdir, "..", "..", "data", "acsl.json")
sl = SequentialLearner.from_json(sl_data_path)
ds = sl.design_space
cs = sl.candidate_selector

# get full aq scores
aq_hist = []
max_aq_hist = []
unc_cand_hist = []
for i, preds in enumerate(sl.predictions_history):
    _, max_aq_score, aq_scores = cs.choose_candidate(
        design_space=ds,
        allowed_idx=sl.train_idx_history[i],
        predictions=preds,
        uncertainties=sl.uncertainties_history[i],
    )
    aq_hist.append(aq_scores)
    max_aq_hist.append(max_aq_score[0])

    unc_cand_hist.append(
        sl.uncertainties_history[i][np.where(aq_scores == max_aq_score[0])]
    )


aq_hist = np.array(aq_hist)
max_aq_hist = np.array(max_aq_hist)


# save arrays to disk
np.savetxt("AQ_HIST.txt", aq_hist[:-1])
np.savetxt("MAX_AQ_HIST.txt", max_aq_hist[:-1])
np.savetxt("UNC_CAND_HIST.txt", unc_cand_hist[:-1])
