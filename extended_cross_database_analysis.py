# Extended cross-database analysis: composition overlap, PBEsol offset calibration,
# Hubbard+U classification, feature-tier comparison, distribution matching, extended
# 11-model benchmark (internal + external), and bootstrap confidence intervals.
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition
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

PROJ = Path("/glade/work/ajalali/13 - Perovskite-CrossDB-ML-project")
DATA = PROJ / "data"
OUT = PROJ / "models" / "crossdb"
RNG0 = 42

TARGETS = ["formation_energy_per_atom", "band_gap"]
FUNCTIONALS = ["pbe", "pbesol", "scan"]

# Hubbard+U element set validated against 15 real MP GGA+U entries' actual
# is_hubbard/hubbards metadata (pymatgen MPRelaxSet LDAUU oxide table).
MP_HUBBARD_ELEMENTS = {"Co", "Cr", "Fe", "Mn", "Mo", "Ni", "V", "W"}
OLD_PROXY_TM = set(
    "Sc Ti V Cr Mn Fe Co Ni Cu Zn Y Zr Nb Mo Tc Ru Rh Pd Ag Cd "
    "Hf Ta W Re Os Ir Pt Au Hg".split()
)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_feature_cols(df, tier="full"):
    magpie = [c for c in df.columns if c.startswith("MagpieData")]
    tol = ["tolerance_factor"]
    struct = ["space_group_number", "a", "b", "c", "alpha", "beta", "gamma",
              "volume", "nsites"]
    tol = [c for c in tol if c in df.columns]
    struct = [c for c in struct if c in df.columns]
    if tier == "composition_only":
        return magpie
    if tier == "plus_tolerance":
        return magpie + tol
    return magpie + tol + struct  # full


def align(train_df, test_df, feature_cols):
    X_train = train_df[feature_cols].to_numpy(dtype=float)
    X_test = test_df[feature_cols].to_numpy(dtype=float)
    col_mean = np.nanmean(X_train, axis=0)
    X_train = np.where(np.isnan(X_train), col_mean, X_train)
    X_test = np.where(np.isnan(X_test), col_mean, X_test)
    return X_train, X_test, col_mean


def xgb():
    return XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
                         n_jobs=8, random_state=RNG0)


def metrics(y_true, y_pred):
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else None,
    }


def bootstrap_ci(y_true, y_pred, n_boot=2000, seed=0, metric="mae"):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        if metric == "mae":
            vals.append(mean_absolute_error(yt, yp))
        elif metric == "rmse":
            vals.append(np.sqrt(mean_squared_error(yt, yp)))
        elif metric == "r2":
            if len(np.unique(yt)) < 2:
                continue
            vals.append(r2_score(yt, yp))
    vals = np.array(vals)
    return {"mean": float(vals.mean()), "lo95": float(np.percentile(vals, 2.5)),
            "hi95": float(np.percentile(vals, 97.5)), "n_boot_used": int(len(vals))}


def reduced_formula(f):
    try:
        return Composition(f).reduced_formula
    except Exception:
        return f


# ---------------------------------------------------------------------
log("Loading data")
mp_df = pd.read_parquet(DATA / "mp_perovskites.parquet")
alex = {f: pd.read_parquet(DATA / f"alexandria_{f}.parquet") for f in FUNCTIONALS}
feature_cols_full = get_feature_cols(mp_df, "full")
log(f"MP: {mp_df.shape}, Alexandria: {[(f, d.shape) for f, d in alex.items()]}")

mp_df["reduced_formula"] = mp_df["formula"].apply(reduced_formula)
mp_reduced_set = set(mp_df["reduced_formula"])
for f in FUNCTIONALS:
    alex[f]["reduced_formula"] = alex[f]["formula"].apply(reduced_formula)

results = {}

# ======================================================================
# Composition overlap: MP training set vs each Alexandria test set
# ======================================================================
log("=== Composition overlap, MP vs Alexandria ===")
overlap = {}
for f in FUNCTIONALS:
    adf = alex[f]
    is_overlap = adf["reduced_formula"].isin(mp_reduced_set)
    overlap[f] = {
        "n_total": int(len(adf)), "n_overlap": int(is_overlap.sum()),
        "n_novel": int((~is_overlap).sum()),
        "pct_overlap": float(100 * is_overlap.mean()),
    }
    log(f"  {f}: {overlap[f]}")
results["composition_overlap"] = overlap

# Overlap BETWEEN the three Alexandria functional sets (matched-compound check)
log("=== Matched-compound overlap across Alexandria functionals ===")
pairs = [("pbe", "pbesol"), ("pbe", "scan"), ("pbesol", "scan")]
cross_func_overlap = {}
sets = {f: set(alex[f]["reduced_formula"]) for f in FUNCTIONALS}
for a, b in pairs:
    shared = sets[a] & sets[b]
    cross_func_overlap[f"{a}_vs_{b}"] = {"n_shared": len(shared), "formulas": sorted(shared)[:50]}
    log(f"  {a} vs {b}: {len(shared)} shared reduced formulas")
results["cross_functional_overlap"] = cross_func_overlap

# ======================================================================
# Train final model (all MP compounds) + internal-validation model (80/20) --
# matches train_crossdb.py's protocol, extended here with more analysis.
# ======================================================================
log("=== Training final (100% MP) and internal-validation (80/20 MP) models, full feature tier ===")
models_per_target = {}
internal_preds = {}
transfer_preds = {}
for target in TARGETS:
    y_mp = mp_df[target].to_numpy(dtype=float)
    rng = np.random.RandomState(RNG0)
    idx = rng.permutation(len(mp_df))
    n_test = int(0.2 * len(mp_df))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    X_all = mp_df[feature_cols_full].to_numpy(dtype=float)
    col_mean = np.nanmean(X_all[train_idx], axis=0)
    X_all_filled = np.where(np.isnan(X_all), col_mean, X_all)
    m_int = xgb()
    m_int.fit(X_all_filled[train_idx], y_mp[train_idx])
    pred_int = m_int.predict(X_all_filled[test_idx])
    internal_preds[target] = {"y_true": y_mp[test_idx], "y_pred": pred_int,
                               "test_idx": test_idx}

    m_full = xgb()
    X_train_full, _, col_mean_full = align(mp_df, mp_df, feature_cols_full)
    m_full.fit(X_train_full, y_mp)
    models_per_target[target] = m_full

    tpreds = {}
    for f in FUNCTIONALS:
        adf = alex[f].dropna(subset=[target]).reset_index(drop=True)
        _, X_test, _ = align(mp_df, adf, feature_cols_full)
        pred = m_full.predict(X_test)
        tpreds[f] = pd.DataFrame({
            "reduced_formula": adf["reduced_formula"], "y_true": adf[target].to_numpy(float),
            "y_pred": pred,
        })
    transfer_preds[target] = tpreds

results["internal_vs_final_clarification"] = {
    "note": ("'MP-internal' metrics use a model trained on the 80% random-split "
             "training subset only. 'Alexandria transfer' metrics use a SEPARATE "
             "final model retrained on ALL MP compounds (0% held out), since no "
             "Alexandria label is ever used for training/selection either way. The "
             "two are not directly comparable sample-size-for-sample-size, which is "
             "stated explicitly in the manuscript.")
}

# ======================================================================
# Composition-overlap-split accuracy
# ======================================================================
log("=== Accuracy split by composition-overlap status ===")
overlap_accuracy = {}
for target in TARGETS:
    overlap_accuracy[target] = {}
    for f in FUNCTIONALS:
        tp = transfer_preds[target][f]
        is_ov = tp["reduced_formula"].isin(mp_reduced_set)
        ov_m = metrics(tp.loc[is_ov, "y_true"], tp.loc[is_ov, "y_pred"]) if is_ov.sum() > 1 else None
        nov_m = metrics(tp.loc[~is_ov, "y_true"], tp.loc[~is_ov, "y_pred"]) if (~is_ov).sum() > 1 else None
        overlap_accuracy[target][f] = {"overlapping": ov_m, "novel": nov_m}
        log(f"  {target}/{f}: overlap(n={ov_m['n'] if ov_m else 0})={ov_m}, "
            f"novel(n={nov_m['n'] if nov_m else 0})={nov_m}")
results["overlap_split_accuracy"] = overlap_accuracy

# ======================================================================
# PBEsol offset: calibration/held-out protocol, repeated, + covariate check
# ======================================================================
log("=== PBEsol offset calibration/held-out protocol (30 seeds) ===")
pbe_target = "formation_energy_per_atom"
pbesol_df = transfer_preds[pbe_target]["pbesol"].copy()
adf_pbesol_full = alex["pbesol"].dropna(subset=[pbe_target]).reset_index(drop=True)
assert len(adf_pbesol_full) == len(pbesol_df)
pbesol_df["a_site"] = adf_pbesol_full["a_site"].values
pbesol_df["b_site"] = adf_pbesol_full["b_site"].values
pbesol_df["has_tm_old_proxy"] = adf_pbesol_full.apply(
    lambda r: int(r["a_site"] in OLD_PROXY_TM or r["b_site"] in OLD_PROXY_TM), axis=1)
pbesol_df["has_hubbard_u"] = adf_pbesol_full.apply(
    lambda r: int(r["a_site"] in MP_HUBBARD_ELEMENTS or r["b_site"] in MP_HUBBARD_ELEMENTS), axis=1)
pbesol_df["residual"] = pbesol_df["y_pred"] - pbesol_df["y_true"]

raw_mae = mean_absolute_error(pbesol_df["y_true"], pbesol_df["y_pred"])
raw_r2 = r2_score(pbesol_df["y_true"], pbesol_df["y_pred"])
raw_mean_resid = float(pbesol_df["residual"].mean())

cal_test_results = []
rng = np.random.RandomState(0)
n = len(pbesol_df)
for seed in range(30):
    perm = rng.permutation(n)
    n_cal = n // 2
    cal_idx, test_idx = perm[:n_cal], perm[n_cal:]
    cal = pbesol_df.iloc[cal_idx]
    test = pbesol_df.iloc[test_idx]
    offset = cal["residual"].mean()  # pred - true, mean on calibration half
    corrected_pred_test = test["y_pred"] - offset  # subtract mean residual
    mae_corr = mean_absolute_error(test["y_true"], corrected_pred_test)
    r2_corr = r2_score(test["y_true"], corrected_pred_test) if len(test) > 1 else None
    mae_raw_test = mean_absolute_error(test["y_true"], test["y_pred"])
    cal_test_results.append({"offset": float(offset), "mae_corrected": float(mae_corr),
                              "r2_corrected": float(r2_corr) if r2_corr is not None else None,
                              "mae_raw_on_test_half": float(mae_raw_test)})

offsets = np.array([r["offset"] for r in cal_test_results])
maes_corr = np.array([r["mae_corrected"] for r in cal_test_results])
maes_raw = np.array([r["mae_raw_on_test_half"] for r in cal_test_results])

results["pbesol_offset_calibrated"] = {
    "raw_mean_residual_full_set_in_sample": raw_mean_resid,
    "raw_mae_full_set": float(raw_mae),
    "raw_r2_full_set": float(raw_r2),
    "n_splits": 30,
    "offset_mean": float(offsets.mean()), "offset_std": float(offsets.std()),
    "mae_corrected_mean": float(maes_corr.mean()), "mae_corrected_std": float(maes_corr.std()),
    "mae_raw_on_held_out_half_mean": float(maes_raw.mean()),
    "note": ("Calibration (offset estimation) and evaluation are done on disjoint "
             "random halves of the PBEsol set, repeated 30x. The correction "
             "direction is: corrected_pred = pred - mean_residual, where "
             "mean_residual = pred-true computed on the CALIBRATION half only. "
             "Since mean_residual is negative, this numerically ADDS to each "
             "prediction."),
}
log(f"  Calibrated (30 random cal/test splits): offset={offsets.mean():.4f}+-{offsets.std():.4f}, "
    f"corrected MAE={maes_corr.mean():.4f}+-{maes_corr.std():.4f} "
    f"(vs raw MAE on held-out half {maes_raw.mean():.4f})")

# Does the offset vary with TM content / formation-energy magnitude?
tm_resid = pbesol_df.loc[pbesol_df["has_hubbard_u"] == 1, "residual"]
non_tm_resid = pbesol_df.loc[pbesol_df["has_hubbard_u"] == 0, "residual"]
corr_resid_vs_Eform = float(np.corrcoef(pbesol_df["y_true"], pbesol_df["residual"])[0, 1])
results["pbesol_offset_covariates"] = {
    "mean_residual_hubbard_u_compounds": float(tm_resid.mean()) if len(tm_resid) else None,
    "mean_residual_non_hubbard_u_compounds": float(non_tm_resid.mean()) if len(non_tm_resid) else None,
    "n_hubbard_u": int(len(tm_resid)), "n_non_hubbard_u": int(len(non_tm_resid)),
    "pearson_corr_residual_vs_formation_energy": corr_resid_vs_Eform,
}
log(f"  Offset by +U-applicable group: U-group={tm_resid.mean():.4f} (n={len(tm_resid)}), "
    f"non-U-group={non_tm_resid.mean():.4f} (n={len(non_tm_resid)}); "
    f"corr(residual, E_form)={corr_resid_vs_Eform:.3f}")

# ======================================================================
# Corrected Hubbard+U classification vs old crude transition-metal proxy
# ======================================================================
log("=== Corrected +U classification (validated vs real MP metadata) ===")
hubbard_check = {}
for target in TARGETS:
    hubbard_check[target] = {}
    for f in FUNCTIONALS:
        adf = alex[f].dropna(subset=[target]).reset_index(drop=True)
        tp = transfer_preds[target][f]
        has_u = adf.apply(lambda r: int(r["a_site"] in MP_HUBBARD_ELEMENTS or
                                         r["b_site"] in MP_HUBBARD_ELEMENTS), axis=1).to_numpy()
        has_old_tm = adf.apply(lambda r: int(r["a_site"] in OLD_PROXY_TM or
                                              r["b_site"] in OLD_PROXY_TM), axis=1).to_numpy()
        err = np.abs(tp["y_true"].to_numpy() - tp["y_pred"].to_numpy())
        u_mask = has_u.astype(bool)
        old_mask = has_old_tm.astype(bool)
        entry = {
            "n_hubbard_u_applicable": int(u_mask.sum()), "n_non_hubbard_u": int((~u_mask).sum()),
            "mae_hubbard_u": float(err[u_mask].mean()) if u_mask.sum() > 1 else None,
            "mae_non_hubbard_u": float(err[~u_mask].mean()) if (~u_mask).sum() > 1 else None,
            "n_old_proxy_tm": int(old_mask.sum()), "n_old_proxy_non_tm": int((~old_mask).sum()),
            "mae_old_proxy_tm": float(err[old_mask].mean()) if old_mask.sum() > 1 else None,
            "mae_old_proxy_non_tm": float(err[~old_mask].mean()) if (~old_mask).sum() > 1 else None,
        }
        hubbard_check[target][f] = entry
        log(f"  {target}/{f}: +U(n={entry['n_hubbard_u_applicable']}) MAE={entry['mae_hubbard_u']}, "
            f"non-U(n={entry['n_non_hubbard_u']}) MAE={entry['mae_non_hubbard_u']} | "
            f"old-proxy-TM(n={entry['n_old_proxy_tm']}) MAE={entry['mae_old_proxy_tm']}, "
            f"old-proxy-non-TM(n={entry['n_old_proxy_non_tm']}) MAE={entry['mae_old_proxy_non_tm']}")
results["hubbard_u_corrected"] = hubbard_check
results["hubbard_u_validation"] = {
    "method": ("Queried mp-api get_entries() for 15 random MP training compounds, "
               "inspected actual is_hubbard/hubbards fields on GGA/GGA+U-type "
               "entries. Every GGA+U entry's hubbards dict matched exactly the "
               "pymatgen MPRelaxSet LDAUU oxide table "
               "{Co,Cr,Fe,Mn,Mo,Ni,V,W}; every plain-GGA entry's A/B-site "
               "elements were outside that set. Rule confirmed empirically, "
               "not assumed."),
    "elements": sorted(MP_HUBBARD_ELEMENTS),
}

# ======================================================================
# Three-tier feature comparison
# ======================================================================
log("=== Three-tier feature comparison ===")
tier_results = {}
for tier in ["composition_only", "plus_tolerance", "full"]:
    fcols = get_feature_cols(mp_df, tier)
    tier_results[tier] = {"n_features": len(fcols), "targets": {}}
    for target in TARGETS:
        y_mp = mp_df[target].to_numpy(dtype=float)
        X_train, _, _ = align(mp_df, mp_df, fcols)
        m = xgb()
        m.fit(X_train, y_mp)
        per_func = {}
        for f in FUNCTIONALS:
            adf = alex[f].dropna(subset=[target]).reset_index(drop=True)
            _, X_test, _ = align(mp_df, adf, fcols)
            pred = m.predict(X_test)
            per_func[f] = metrics(adf[target].to_numpy(float), pred)
        tier_results[tier]["targets"][target] = per_func
        log(f"  tier={tier} ({len(fcols)} feats) target={target}: "
            f"{ {k: round(v['r2'],3) if v['r2'] else None for k,v in per_func.items()} }")
results["feature_tier_comparison"] = tier_results

# ======================================================================
# Distribution-matching: evaluate MP model on MP subset matched to Alexandria
# ======================================================================
log("=== Distribution-matching check (formation energy) ===")
dist_match = {}
target = "formation_energy_per_atom"
y_test_int = internal_preds[target]["y_true"]
pred_test_int = internal_preds[target]["y_pred"]
for f in FUNCTIONALS:
    adf = alex[f].dropna(subset=[target]).reset_index(drop=True)
    alex_vals = adf[target].to_numpy(float)
    lo, hi = np.percentile(alex_vals, [1, 99])
    mask_match = (y_test_int >= lo) & (y_test_int <= hi)
    if mask_match.sum() > 5:
        m_matched = metrics(y_test_int[mask_match], pred_test_int[mask_match])
    else:
        m_matched = None
    dist_match[f] = {
        "alexandria_Eform_p1_p99": [float(lo), float(hi)],
        "mp_internal_test_full": metrics(y_test_int, pred_test_int),
        "mp_internal_test_distribution_matched": m_matched,
        "n_matched": int(mask_match.sum()),
    }
    log(f"  {f}: MP-internal full MAE={dist_match[f]['mp_internal_test_full']['mae']:.4f}, "
        f"distribution-matched (n={mask_match.sum()}) MAE="
        f"{m_matched['mae'] if m_matched else None}")
results["distribution_matching"] = dist_match

# ======================================================================
# Extended 11-model benchmark: internal + external, MAE/RMSE/R2 for all
# ======================================================================
log("=== Extended 11-model benchmark (internal + external) ===")
MODELS = {
    "XGBoost": lambda: XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
                                     n_jobs=8, random_state=RNG0),
    "RandomForest": lambda: RandomForestRegressor(n_estimators=300, max_depth=None,
                                                   n_jobs=8, random_state=RNG0),
    "ExtraTrees": lambda: ExtraTreesRegressor(n_estimators=300, max_depth=None,
                                               n_jobs=8, random_state=RNG0),
    "GradientBoosting": lambda: GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                                           learning_rate=0.05, subsample=0.8,
                                                           random_state=RNG0),
    "AdaBoost": lambda: AdaBoostRegressor(n_estimators=200, learning_rate=0.05,
                                          random_state=RNG0),
    "SVR": lambda: SVR(kernel="rbf", C=10.0, epsilon=0.01),
    "kNN": lambda: KNeighborsRegressor(n_neighbors=8, weights="distance"),
    "GaussianProcess": lambda: GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0),
        normalize_y=True, random_state=RNG0),
    "Ridge": lambda: Ridge(alpha=1.0),
    "Lasso": lambda: Lasso(alpha=0.001, max_iter=10000),
    "MLP": lambda: MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000,
                                 random_state=RNG0, early_stopping=True),
}
NEEDS_SCALING = {"SVR", "kNN", "Ridge", "Lasso", "MLP", "GaussianProcess"}

bench = {}
bench_preds_for_bootstrap = {}
for target in TARGETS:
    bench[target] = {}
    y_mp = mp_df[target].to_numpy(dtype=float)
    rng = np.random.RandomState(RNG0)
    idx = rng.permutation(len(mp_df))
    n_test = int(0.2 * len(mp_df))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    X_all = mp_df[feature_cols_full].to_numpy(dtype=float)
    col_mean_int = np.nanmean(X_all[train_idx], axis=0)
    X_all_filled = np.where(np.isnan(X_all), col_mean_int, X_all)
    X_tr_int, X_te_int = X_all_filled[train_idx], X_all_filled[test_idx]
    y_tr_int, y_te_int = y_mp[train_idx], y_mp[test_idx]
    scaler_int = StandardScaler().fit(X_tr_int)

    X_train_full_raw = mp_df[feature_cols_full].to_numpy(dtype=float)
    col_mean_full = np.nanmean(X_train_full_raw, axis=0)
    X_train_full_raw = np.where(np.isnan(X_train_full_raw), col_mean_full, X_train_full_raw)
    scaler_full = StandardScaler().fit(X_train_full_raw)

    for name, ctor in MODELS.items():
        model_int = ctor()
        Xtr_i = scaler_int.transform(X_tr_int) if name in NEEDS_SCALING else X_tr_int
        Xte_i = scaler_int.transform(X_te_int) if name in NEEDS_SCALING else X_te_int
        model_int.fit(Xtr_i, y_tr_int)
        pred_int_m = model_int.predict(Xte_i)
        internal_m = metrics(y_te_int, pred_int_m)

        model_full = ctor()
        Xtr_f = scaler_full.transform(X_train_full_raw) if name in NEEDS_SCALING else X_train_full_raw
        model_full.fit(Xtr_f, y_mp)

        per_func = {}
        for f in FUNCTIONALS:
            adf = alex[f].dropna(subset=[target]).reset_index(drop=True)
            X_test_raw = adf[feature_cols_full].to_numpy(dtype=float)
            X_test_raw = np.where(np.isnan(X_test_raw), col_mean_full, X_test_raw)
            Xte_f = scaler_full.transform(X_test_raw) if name in NEEDS_SCALING else X_test_raw
            y_test_f = adf[target].to_numpy(dtype=float)
            pred_f = model_full.predict(Xte_f)
            per_func[f] = metrics(y_test_f, pred_f)
            if target == pbe_target:
                bench_preds_for_bootstrap.setdefault(f, {})[name] = (y_test_f, pred_f)

        bench[target][name] = {"internal": internal_m, "external": per_func}
        log(f"  {target}/{name}: internal R2={internal_m['r2']:.3f}, "
            f"external R2={ {k: round(v['r2'],3) if v['r2'] else None for k,v in per_func.items()} }")
results["extended_model_benchmark"] = bench

# ======================================================================
# Bootstrap CIs for principal metrics + paired bootstrap (XGBoost vs runner-up)
# ======================================================================
log("=== Bootstrap 95% CIs ===")
boot = {}
for target in TARGETS:
    boot[target] = {"mp_internal": {}, "transfer": {}}
    yt, yp = internal_preds[target]["y_true"], internal_preds[target]["y_pred"]
    for metric in ["mae", "rmse", "r2"]:
        boot[target]["mp_internal"][metric] = bootstrap_ci(yt, yp, metric=metric)
    for f in FUNCTIONALS:
        tp = transfer_preds[target][f]
        boot[target]["transfer"][f] = {}
        for metric in ["mae", "rmse", "r2"]:
            boot[target]["transfer"][f][metric] = bootstrap_ci(
                tp["y_true"].to_numpy(), tp["y_pred"].to_numpy(), metric=metric)
    log(f"  {target} MP-internal MAE 95% CI: {boot[target]['mp_internal']['mae']}")

# Paired bootstrap: XGBoost vs RandomForest on identical PBE test compounds --
# same resample indices both models.
paired = {}
if "pbe" in bench_preds_for_bootstrap and "XGBoost" in bench_preds_for_bootstrap["pbe"]:
    yt_x, yp_x = bench_preds_for_bootstrap["pbe"]["XGBoost"]
    yt_r, yp_r = bench_preds_for_bootstrap["pbe"].get("RandomForest", (None, None))
    if yp_r is not None:
        rng = np.random.RandomState(1)
        n = len(yt_x)
        diffs = []
        for _ in range(2000):
            idx = rng.randint(0, n, n)
            mae_x = mean_absolute_error(yt_x[idx], yp_x[idx])
            mae_r = mean_absolute_error(yt_r[idx], yp_r[idx])
            diffs.append(mae_x - mae_r)
        diffs = np.array(diffs)
        paired["xgboost_minus_randomforest_mae_pbe_formation_energy"] = {
            "mean_diff": float(diffs.mean()), "lo95": float(np.percentile(diffs, 2.5)),
            "hi95": float(np.percentile(diffs, 97.5)),
        }
        log(f"  Paired bootstrap XGB-RF MAE diff: {paired['xgboost_minus_randomforest_mae_pbe_formation_energy']}")
results["bootstrap_cis"] = boot
results["paired_bootstrap"] = paired

# ======================================================================
# Save everything
# ======================================================================
with open(DATA / "extended_cross_database_analysis.json", "w") as fh:
    json.dump(results, fh, indent=2, default=str)

for target in TARGETS:
    for f in FUNCTIONALS:
        transfer_preds[target][f].to_csv(
            DATA / f"final_model_predictions_{target}_{f}.csv", index=False)
    pd.DataFrame({"y_true": internal_preds[target]["y_true"],
                  "y_pred": internal_preds[target]["y_pred"]}).to_csv(
        DATA / f"final_model_predictions_{target}_mp_internal.csv", index=False)

pbesol_df.to_csv(DATA / "pbesol_offset_detail.csv", index=False)

log("ALL DONE. Saved data/extended_cross_database_analysis.json and final_model_predictions_*.csv")
