"""
Broader model benchmark (supporting evidence): repeat the MP -> Alexandria
transfer test with several genuinely different regressors, identical splits,
for formation_energy_per_atom (primary target) and band_gap.
Checks whether the transfer pattern found with XGBoost (mostly successful,
sometimes exceeding in-distribution performance) is consistent across model
families, or an XGBoost-specific artifact.
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                               GradientBoostingRegressor, AdaBoostRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

PROJ = Path("/glade/work/ajalali/Perovskite-CrossDB-ML-project")
DATA = PROJ / "data"
OUT = PROJ / "models" / "crossdb"

TARGETS = ["formation_energy_per_atom", "band_gap"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_feature_cols(df):
    magpie = [c for c in df.columns if c.startswith("MagpieData")]
    struct = ["space_group_number", "tolerance_factor", "a", "b", "c",
              "alpha", "beta", "gamma", "volume", "nsites"]
    struct = [c for c in struct if c in df.columns]
    return magpie + struct


MODELS = {
    "XGBoost": lambda: XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
                                     n_jobs=8, random_state=42),
    "RandomForest": lambda: RandomForestRegressor(n_estimators=300, max_depth=None,
                                                   n_jobs=8, random_state=42),
    "ExtraTrees": lambda: ExtraTreesRegressor(n_estimators=300, max_depth=None,
                                               n_jobs=8, random_state=42),
    "GradientBoosting": lambda: GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                                           learning_rate=0.05, subsample=0.8,
                                                           random_state=42),
    "AdaBoost": lambda: AdaBoostRegressor(n_estimators=200, learning_rate=0.05,
                                          random_state=42),
    "SVR": lambda: SVR(kernel="rbf", C=10.0, epsilon=0.01),
    "kNN": lambda: KNeighborsRegressor(n_neighbors=8, weights="distance"),
    "GaussianProcess": lambda: GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0),
        normalize_y=True, random_state=42),
    "Ridge": lambda: Ridge(alpha=1.0),
    "Lasso": lambda: Lasso(alpha=0.001, max_iter=10000),
    "MLP": lambda: MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000,
                                 random_state=42, early_stopping=True),
}
NEEDS_SCALING = {"SVR", "kNN", "Ridge", "Lasso", "MLP", "GaussianProcess"}


def main():
    t0 = time.time()
    mp_df = pd.read_parquet(DATA / "mp_perovskites.parquet")
    feature_cols = get_feature_cols(mp_df)
    alex = {f: pd.read_parquet(DATA / f"alexandria_{f}.parquet") for f in ["pbe", "pbesol", "scan"]}

    results = {}
    for target in TARGETS:
        log(f"=== Target: {target} ===")
        results[target] = {}
        X_train_raw = mp_df[feature_cols].to_numpy(dtype=float)
        col_mean = np.nanmean(X_train_raw, axis=0)
        X_train_raw = np.where(np.isnan(X_train_raw), col_mean, X_train_raw)
        y_train = mp_df[target].to_numpy(dtype=float)

        scaler = StandardScaler().fit(X_train_raw)
        X_train_scaled = scaler.transform(X_train_raw)

        for name, ctor in MODELS.items():
            model = ctor()
            Xtr = X_train_scaled if name in NEEDS_SCALING else X_train_raw
            model.fit(Xtr, y_train)
            model_results = {}
            for functional, adf in alex.items():
                adf_valid = adf.dropna(subset=[target]).reset_index(drop=True)
                X_test_raw = adf_valid[feature_cols].to_numpy(dtype=float)
                X_test_raw = np.where(np.isnan(X_test_raw), col_mean, X_test_raw)
                Xte = scaler.transform(X_test_raw) if name in NEEDS_SCALING else X_test_raw
                y_test = adf_valid[target].to_numpy(dtype=float)
                pred = model.predict(Xte)
                m = {
                    "mae": float(mean_absolute_error(y_test, pred)),
                    "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
                    "r2": float(r2_score(y_test, pred)) if len(y_test) > 1 else None,
                }
                model_results[functional] = m
            results[target][name] = model_results
            log(f"  {name}: {model_results}")

    with open(OUT / "benchmark_models.json", "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    log(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
