import os

import numpy as np
import umap
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from autocat.learning.sequential import SequentialLearner

thisdir = os.path.dirname(os.path.abspath(__file__))
sl_data_path = os.path.join(thisdir, "..", "..", "data", "acsl.json")
sl = SequentialLearner.from_json(sl_data_path)
ds = sl.design_space

rf = sl.predictor

X = rf.featurizer.featurize_multiple(ds.design_space_structures)
scaler = StandardScaler().fit(X)
X = scaler.transform(X)


reducer = umap.UMAP(n_neighbors=20, min_dist=0.1, metric="manhattan")
embedding = reducer.fit_transform(X)

np.savetxt("L1_EMBEDDING.txt", embedding)
