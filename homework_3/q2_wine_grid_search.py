"""Problem 2(c): grid search on white-wine quality with a 70/10/20 train/validation/test split.

Models: RandomForestRegressor and KNeighborsRegressor, 600 configurations each
(more than the 490 validation points). Selection uses validation MSE only; the
selected model is then evaluated once on the test set.

Run:  python q2_wine_grid_search.py      (about 6 minutes on 2 CPU cores)
Output: outputs/q2_results.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
SEED = 0

df = pd.read_csv(HERE / "data" / "winequality-white.csv", sep=";")
X = df.drop(columns="quality").values
y = df["quality"].values.astype(float)

# 70% train, then split the remaining 30% into 10% validation and 20% test.
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, random_state=SEED)
X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=2 / 3, random_state=SEED)
print(f"train={len(y_tr)}  val={len(y_va)}  test={len(y_te)}")

GRIDS = {
    "RandomForest": (
        lambda p: RandomForestRegressor(random_state=SEED, n_jobs=-1, **p),
        dict(n_estimators=[50, 100], max_depth=[None, 4, 8, 12, 16, 24],
             min_samples_leaf=[1, 2, 4, 8, 16], max_features=[0.2, 0.33, 0.5, 0.75, 1.0],
             bootstrap=[True, False]),
    ),
    # Scaler is fit inside the pipeline on training data only.
    "KNN": (
        lambda p: make_pipeline(StandardScaler(), KNeighborsRegressor(**p)),
        dict(n_neighbors=list(range(1, 101)), weights=["uniform", "distance"], p=[1, 1.5, 2]),
    ),
}

results = {}
for name, (make_model, grid) in GRIDS.items():
    configs = list(ParameterGrid(grid))
    assert len(configs) > len(y_va)
    val_mse = [mse(y_va, make_model(p).fit(X_tr, y_tr).predict(X_va)) for p in configs]
    best = configs[int(np.argmin(val_mse))]
    test_mse = mse(y_te, make_model(best).fit(X_tr, y_tr).predict(X_te))
    results[name] = dict(n_configs=len(configs), best_params=best,
                         val_mse=float(min(val_mse)), test_mse=float(test_mse))
    print(name, results[name])

(HERE / "outputs").mkdir(exist_ok=True)
(HERE / "outputs" / "q2_results.json").write_text(json.dumps(results, indent=2, default=str))
