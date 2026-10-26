# Computes both internal (MP 80/20 split) and external (Alexandria transfer) metrics
# for all 11 models in a single consistent pass, using the identical model configs as
# benchmark_models.py and the identical MP 80/20 split as the matched-training-size
# control in feature_tier_and_robustness_checks.py.
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path("/glade/work/ajalali/13 - Perovskite-CrossDB-ML-project")
DATA = ROOT / "data"
RNG = 42
TARGETS = ["formation_energy_per_atom", "band_gap"]

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def get_feature_cols(df):
    magpie = [c for c in df.columns if c.startswith("MagpieData")]
    struct = ["space_group_number", "tolerance_factor", "a", "b", "c",
              "alpha", "beta", "gamma", "volume", "nsites"]
    return magpie + [c for c in struct if c in df.columns]

MODELS = {
    "XGBoost": lambda: XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
                                     n_jobs=8, random_state=RNG),
    "RandomForest": lambda: RandomForestRegressor(n_estimators=300, max_depth=None,
                                                   n_jobs=8, random_state=RNG),
    "ExtraTrees": lambda: ExtraTreesRegressor(n_estimators=300, max_depth=None,
                                               n_jobs=8, random_state=RNG),
    "GradientBoosting": lambda: GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                                           learning_rate=0.05, subsample=0.8,
                                                           random_state=RNG),
    "AdaBoost": lambda: AdaBoostRegressor(n_estimators=200, learning_rate=0.05,
                                          random_state=RNG),
    "SVR": lambda: SVR(kernel="rbf", C=10.0, epsilon=0.01),
    "kNN": lambda: KNeighborsRegressor(n_neighbors=8, weights="distance"),
    "GaussianProcess": lambda: GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0),
        normalize_y=True, random_state=RNG),
    "Ridge": lambda: Ridge(alpha=1.0),
    "Lasso": lambda: Lasso(alpha=0.001, max_iter=10000),
    "MLP": lambda: MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000,
                                 random_state=RNG, early_stopping=True),
}
NEEDS_SCALING = {"SVR", "kNN", "Ridge", "Lasso", "MLP", "GaussianProcess"}

def metrics(y, p):
    return {"mae": float(mean_absolute_error(y, p)),
            "rmse": float(np.sqrt(mean_squared_error(y, p))), "r2": float(r2_score(y, p))}

mp_df = pd.read_parquet(DATA / "mp_perovskites.parquet")
feature_cols = get_feature_cols(mp_df)
alex = {f: pd.read_parquet(DATA / f"alexandria_{f}.parquet") for f in ["pbe", "pbesol", "scan"]}

mp_train80, mp_test20 = train_test_split(mp_df, test_size=0.2, random_state=RNG)

results = {}
for target in TARGETS:
    log(f"=== Target: {target} ===")
    results[target] = {}

    # external model: fit on full MP set (matches the main train_crossdb.py/
    # benchmark_models.py protocol)
    Xfull_raw = mp_df[feature_cols].to_numpy(dtype=float)
    col_mean_full = np.nanmean(Xfull_raw, axis=0)
    Xfull_raw = np.where(np.isnan(Xfull_raw), col_mean_full, Xfull_raw)
    yfull = mp_df[target].to_numpy(dtype=float)
    scaler_full = StandardScaler().fit(Xfull_raw)
    Xfull_scaled = scaler_full.transform(Xfull_raw)

    # internal model: fit on MP 80% split only, evaluate on MP held-out 20%
    Xtr80_raw = mp_train80[feature_cols].to_numpy(dtype=float)
    col_mean_80 = np.nanmean(Xtr80_raw, axis=0)
    Xtr80_raw = np.where(np.isnan(Xtr80_raw), col_mean_80, Xtr80_raw)
    ytr80 = mp_train80[target].to_numpy(dtype=float)
    Xte20_raw = mp_test20[feature_cols].to_numpy(dtype=float)
    Xte20_raw = np.where(np.isnan(Xte20_raw), col_mean_80, Xte20_raw)
    yte20 = mp_test20[target].to_numpy(dtype=float)
    scaler_80 = StandardScaler().fit(Xtr80_raw)
    Xtr80_scaled = scaler_80.transform(Xtr80_raw)
    Xte20_scaled = scaler_80.transform(Xte20_raw)

    for name, ctor in MODELS.items():
        model_ext = ctor()
        Xtr = Xfull_scaled if name in NEEDS_SCALING else Xfull_raw
        model_ext.fit(Xtr, yfull)
        external = {}
        for functional, adf in alex.items():
            adf_valid = adf.dropna(subset=[target]).reset_index(drop=True)
            Xte_raw = adf_valid[feature_cols].to_numpy(dtype=float)
            Xte_raw = np.where(np.isnan(Xte_raw), col_mean_full, Xte_raw)
            Xte = scaler_full.transform(Xte_raw) if name in NEEDS_SCALING else Xte_raw
            yte = adf_valid[target].to_numpy(dtype=float)
            pred = model_ext.predict(Xte)
            external[functional] = metrics(yte, pred)

        model_int = ctor()
        Xtr80 = Xtr80_scaled if name in NEEDS_SCALING else Xtr80_raw
        Xte20 = Xte20_scaled if name in NEEDS_SCALING else Xte20_raw
        model_int.fit(Xtr80, ytr80)
        pred_int = model_int.predict(Xte20)
        internal = metrics(yte20, pred_int)

        results[target][name] = {"internal_80_20": internal, "external": external}
        log(f"  {name}: internal_r2={internal['r2']:.3f} "
            f"external_pbe_r2={external['pbe']['r2']:.3f} external_pbe_mae={external['pbe']['mae']:.4f}")

with open(DATA / "benchmark_models_internal_and_external.json", "w") as fh:
    json.dump(results, fh, indent=2, default=str)
log("DONE, wrote data/benchmark_models_internal_and_external.json")
