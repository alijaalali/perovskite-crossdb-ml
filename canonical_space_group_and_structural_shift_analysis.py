# Canonical-space-group-restricted retrain (the 5 unambiguous corner-sharing space
# groups only); bootstrap CIs on novel-composition subsets and composition-grouped CV;
# lattice-parameter/volume/space-group differences for composition-matched MP-vs-
# Alexandria compounds; cross-functional paired residual comparison on matched
# compounds; and a normalized transfer-degradation metric per model.
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = Path("/glade/work/ajalali/13 - Perovskite-CrossDB-ML-project")
DATA = ROOT / "data"
RNG = 42
N_BOOT = 2000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


OUT = {}

mp = pd.read_parquet(DATA / "mp_perovskites.parquet")
alex = {f: pd.read_parquet(DATA / f"alexandria_{f}.parquet") for f in ["pbe", "pbesol", "scan"]}
TARGETS = ["formation_energy_per_atom", "band_gap"]


def reduced_formula(f):
    try:
        return Composition(f).reduced_formula
    except Exception:
        return None


mp["reduced_formula"] = mp["formula"].apply(reduced_formula)
for f in alex:
    alex[f]["reduced_formula"] = alex[f]["formula"].apply(reduced_formula)


def get_feature_cols(df, mode="full"):
    magpie = [c for c in df.columns if c.startswith("MagpieData")]
    if mode == "composition_only":
        return magpie
    struct = ["space_group_number", "tolerance_factor", "a", "b", "c",
              "alpha", "beta", "gamma", "volume", "nsites"]
    return magpie + [c for c in struct if c in df.columns]


def align(train_df, test_df, cols):
    Xtr = train_df[cols].to_numpy(dtype=float)
    Xte = test_df[cols].to_numpy(dtype=float)
    m = np.nanmean(Xtr, axis=0)
    Xtr = np.where(np.isnan(Xtr), m, Xtr)
    Xte = np.where(np.isnan(Xte), m, Xte)
    return Xtr, Xte


def fit_xgb(Xtr, ytr):
    model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
                          n_jobs=8, random_state=RNG)
    model.fit(Xtr, ytr)
    return model


def metrics(y, p):
    return {"n": int(len(y)), "mae": float(mean_absolute_error(y, p)),
            "rmse": float(np.sqrt(mean_squared_error(y, p))), "r2": float(r2_score(y, p))}


def bootstrap_metric_ci(y, p, n=N_BOOT, seed=RNG, metric="r2"):
    rng = np.random.RandomState(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        idx = rng.randint(0, len(y), len(y))
        if metric == "r2":
            vals.append(r2_score(y[idx], p[idx]))
        else:
            vals.append(mean_absolute_error(y[idx], p[idx]))
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ============================================================
# Canonical-space-group-restricted retrain
# ============================================================
log("=== Canonical-space-group-restricted retrain ===")
CANONICAL_SG = [221, 225, 62, 167, 140]
mp_canon = mp[mp["space_group_number"].isin(CANONICAL_SG)].reset_index(drop=True)
log(f"Canonical subset: {len(mp_canon)} of {len(mp)} MP compounds")
assert len(mp_canon) == 1375, f"Canonical count mismatch: got {len(mp_canon)}, expected 1375"

canonical_retrain = {"n_canonical": int(len(mp_canon)), "n_full": int(len(mp)), "targets": {}}
for target in TARGETS:
    canonical_retrain["targets"][target] = {"full_model": {}, "canonical_model": {}}
    cols = get_feature_cols(mp, "full")
    ytr_canon = mp_canon[target].to_numpy(dtype=float)
    Xtr_canon, _ = align(mp_canon, mp_canon, cols)
    model_canon = fit_xgb(Xtr_canon, ytr_canon)

    for f in alex:
        # Reuse the exact saved final-model predictions rather than retraining, so
        # this matches the headline table/figure exactly instead of reintroducing
        # retrain-vs-table floating-point drift (XGBoost with n_jobs>1 is not
        # bit-exact reproducible run-to-run even with a fixed random_state).
        full_preds = pd.read_csv(DATA / f"final_model_predictions_{target}_{f}.csv")
        canonical_retrain["targets"][target]["full_model"][f] = metrics(
            full_preds["y_true"].to_numpy(dtype=float),
            full_preds["y_pred"].to_numpy(dtype=float))

        test_df = alex[f].dropna(subset=[target])
        yte = test_df[target].to_numpy(dtype=float)
        _, Xte_canon = align(mp_canon, test_df, cols)
        pred_canon = model_canon.predict(Xte_canon)
        canonical_retrain["targets"][target]["canonical_model"][f] = metrics(yte, pred_canon)
        log(f"  {target} {f}: full(n={len(mp)}) R2={canonical_retrain['targets'][target]['full_model'][f]['r2']:.3f} "
            f"canonical(n=1375) R2={metrics(yte,pred_canon)['r2']:.3f}")
OUT["canonical_space_group_retrain"] = canonical_retrain
json.dump(OUT, open(DATA / "extended_robustness_and_sensitivity_checks.json", "w"), indent=2, default=str)

# ============================================================
# Bootstrap CIs on the novel-composition subsets. Uses the exact saved
# final-model predictions (the same source make_figures.py reads) rather than
# retraining a new model, to avoid reintroducing retrain-vs-table numeric drift.
# ============================================================
log("=== Bootstrap CIs on novel-composition subsets ===")
mp_reduced_set = set(mp["reduced_formula"])
novel_subset_ci = {"targets": {}}
for target in TARGETS:
    novel_subset_ci["targets"][target] = {}
    for f in alex:
        preds = pd.read_csv(DATA / f"final_model_predictions_{target}_{f}.csv")
        is_overlap = preds["reduced_formula"].isin(mp_reduced_set)
        novel = preds[~is_overlap]
        yte_novel = novel["y_true"].to_numpy(dtype=float)
        pred_novel = novel["y_pred"].to_numpy(dtype=float)
        r2_mean, r2_lo, r2_hi = bootstrap_metric_ci(yte_novel, pred_novel, metric="r2")
        mae_mean, mae_lo, mae_hi = bootstrap_metric_ci(yte_novel, pred_novel, metric="mae")
        novel_subset_ci["targets"][target][f] = {
            "n_novel": int(len(novel)),
            "point_estimate": metrics(yte_novel, pred_novel),
            "r2_bootstrap_ci95": [r2_lo, r2_hi],
            "mae_bootstrap_ci95": [mae_lo, mae_hi],
        }
        log(f"  {target} {f} novel (n={len(novel)}): "
            f"R2={metrics(yte_novel,pred_novel)['r2']:.3f} CI=[{r2_lo:.3f},{r2_hi:.3f}] "
            f"MAE CI=[{mae_lo:.3f},{mae_hi:.3f}]")
OUT["novel_subset_bootstrap_ci"] = novel_subset_ci
json.dump(OUT, open(DATA / "extended_robustness_and_sensitivity_checks.json", "w"), indent=2, default=str)

# ============================================================
# Composition-grouped CV on MP (every reduced-composition stays in one fold --
# tests generalization to genuinely new MP compositions, rather than same
# composition appearing in both train and test)
# ============================================================
log("=== Composition-grouped 5-fold CV on MP ===")
comp_grouped_cv = {"targets": {}}
groups = mp["reduced_formula"].to_numpy()
gkf = GroupKFold(n_splits=5)
cols = get_feature_cols(mp, "full")
for target in TARGETS:
    y_all = mp[target].to_numpy(dtype=float)
    X_all, _ = align(mp, mp, cols)
    fold_r2, fold_mae = [], []
    for tr_idx, te_idx in gkf.split(X_all, y_all, groups=groups):
        model = fit_xgb(X_all[tr_idx], y_all[tr_idx])
        pred = model.predict(X_all[te_idx])
        fold_r2.append(r2_score(y_all[te_idx], pred))
        fold_mae.append(mean_absolute_error(y_all[te_idx], pred))
    comp_grouped_cv["targets"][target] = {
        "fold_r2": [float(x) for x in fold_r2], "fold_mae": [float(x) for x in fold_mae],
        "mean_r2": float(np.mean(fold_r2)), "std_r2": float(np.std(fold_r2)),
        "mean_mae": float(np.mean(fold_mae)), "std_mae": float(np.std(fold_mae)),
    }
    log(f"  {target}: composition-grouped 5-fold R2={np.mean(fold_r2):.3f}+-{np.std(fold_r2):.3f} "
        f"MAE={np.mean(fold_mae):.3f}+-{np.std(fold_mae):.3f}")
OUT["composition_grouped_cv"] = comp_grouped_cv
json.dump(OUT, open(DATA / "extended_robustness_and_sensitivity_checks.json", "w"), indent=2, default=str)

# ============================================================
# Lattice-parameter / volume / space-group differences for composition-matched
# MP-vs-Alexandria compounds
# ============================================================
log("=== Structural-feature differences for composition-matched compounds ===")
structural_shift = {}
for f in alex:
    adf = alex[f]
    merged = mp.merge(adf, on="reduced_formula", suffixes=("_mp", "_alex"))
    if len(merged) == 0:
        continue
    numeric_cols = ["volume_mp", "volume_alex", "nsites_mp", "nsites_alex",
                    "space_group_number_mp", "space_group_number_alex"]
    for c in numeric_cols:
        merged[c] = merged[c].astype(float)
    same_sg = (merged["space_group_number_mp"] == merged["space_group_number_alex"])
    # MP and Alexandria store cells with different numbers of formula units, so raw
    # cell volume/lattice constants are not directly comparable -- normalize to
    # volume-per-atom and a pseudo-cubic lattice constant derived from an ABO3
    # (5-atom) formula unit before comparing.
    vpa_mp = merged["volume_mp"] / merged["nsites_mp"]
    vpa_alex = merged["volume_alex"] / merged["nsites_alex"]
    vol_pct_diff = 100 * (vpa_alex - vpa_mp) / vpa_mp
    a_pseudo_mp = (5.0 * vpa_mp) ** (1.0 / 3.0)
    a_pseudo_alex = (5.0 * vpa_alex) ** (1.0 / 3.0)
    mean_abc_diff = (a_pseudo_alex - a_pseudo_mp).abs()
    structural_shift[f] = {
        "n_matched": int(len(merged)),
        "pct_same_space_group": float(same_sg.mean() * 100),
        "mean_abs_volume_per_atom_pct_diff": float(vol_pct_diff.abs().mean()),
        "median_abs_volume_per_atom_pct_diff": float(vol_pct_diff.abs().median()),
        "mean_abs_pseudocubic_lattice_const_diff_angstrom": float(mean_abc_diff.mean()),
        "median_abs_pseudocubic_lattice_const_diff_angstrom": float(mean_abc_diff.median()),
    }
    log(f"  {f}: n={len(merged)} matched, same-SG={same_sg.mean()*100:.1f}%, "
        f"|dV|/V={vol_pct_diff.abs().mean():.1f}%, "
        f"mean |d(a,b,c)|={mean_abc_diff.mean():.4f} A")
OUT["structural_feature_shift"] = structural_shift
json.dump(OUT, open(DATA / "extended_robustness_and_sensitivity_checks.json", "w"), indent=2, default=str)

# ============================================================
# Cross-functional paired comparison on matched compounds: prediction-residual
# differences (model error under functional A vs model error under functional B
# for the SAME composition), which isolates the functional's effect on model
# error from population-composition effects -- distinct from the raw difference
# in the two functionals' own DFT target values (covered by the PBEsol-offset
# analysis).
# ============================================================
log("=== Cross-functional paired comparison (prediction residuals) ===")
extended_analysis = json.load(open(DATA / "extended_cross_database_analysis.json"))
# Reuse the exact saved final-model predictions/residuals for every functional,
# rather than retraining, to avoid reintroducing retrain-vs-table numeric drift
# (no duplicate reduced_formula values within any one functional's set, verified
# separately, so the merge below is one-to-one).
sg_lookup = {f: alex[f][["reduced_formula", "space_group_number"]]
             .assign(space_group_number=lambda d: d["space_group_number"].astype(float))
             for f in alex}

cross_func_paired = {}
pairs = [("pbe", "pbesol"), ("pbe", "scan"), ("pbesol", "scan")]
for a, b in pairs:
    key = f"{a}_vs_{b}"
    formulas = set(alex[a]["reduced_formula"]) & set(alex[b]["reduced_formula"])
    sub = {}
    for target in TARGETS:
        da = pd.read_csv(DATA / f"final_model_predictions_{target}_{a}.csv")
        db = pd.read_csv(DATA / f"final_model_predictions_{target}_{b}.csv")
        da = da[da["reduced_formula"].isin(formulas)].copy()
        db = db[db["reduced_formula"].isin(formulas)].copy()
        da["residual"] = da["y_pred"] - da["y_true"]
        db["residual"] = db["y_pred"] - db["y_true"]
        da = da.merge(sg_lookup[a], on="reduced_formula")
        db = db.merge(sg_lookup[b], on="reduced_formula")
        merged = da[["reduced_formula", "residual", "space_group_number"]].merge(
            db[["reduced_formula", "residual", "space_group_number"]],
            on="reduced_formula", suffixes=(f"_{a}", f"_{b}"))
        if len(merged) == 0:
            continue
        resid_diff = merged[f"residual_{a}"] - merged[f"residual_{b}"]
        same_sg = (merged[f"space_group_number_{a}"] == merged[f"space_group_number_{b}"])
        sub[target] = {
            "n_matched_with_target": int(len(merged)),
            "mean_signed_residual_diff": float(resid_diff.mean()),
            "mean_abs_residual_diff": float(resid_diff.abs().mean()),
            "median_abs_residual_diff": float(resid_diff.abs().median()),
            "pct_same_space_group": float(same_sg.mean() * 100) if len(merged) else None,
        }
        log(f"  {key} {target}: n={len(merged)}, "
            f"mean|resid({a})-resid({b})|={resid_diff.abs().mean():.4f}, "
            f"mean signed={resid_diff.mean():.4f}, same-SG={same_sg.mean()*100:.1f}%")
    cross_func_paired[key] = sub
OUT["cross_functional_paired_residual_comparison"] = cross_func_paired
json.dump(OUT, open(DATA / "extended_robustness_and_sensitivity_checks.json", "w"), indent=2, default=str)

# ============================================================
# Normalized transfer-degradation metric per model (uses the existing extended
# model benchmark, which already has both internal and external R2/MAE per
# model per functional -- a derived statistic, no retraining needed)
# ============================================================
log("=== Normalized transfer-degradation per model ===")
emb = extended_analysis["extended_model_benchmark"]
degradation_by_model = {"targets": {}}
for target in TARGETS:
    degradation_by_model["targets"][target] = {}
    for model_name, d in emb[target].items():
        internal_r2 = d["internal"]["r2"]
        internal_mae = d["internal"]["mae"]
        per_functional = {}
        for f, ext in d["external"].items():
            ext_r2 = ext["r2"]
            ext_mae = ext["mae"]
            # >1 means transfer R2 exceeds internal (can happen, e.g. for a
            # narrower/easier external target distribution).
            retained_r2_frac = (ext_r2 / internal_r2) if internal_r2 != 0 else None
            mae_inflation_ratio = (ext_mae / internal_mae) if internal_mae != 0 else None
            per_functional[f] = {
                "internal_r2": internal_r2, "external_r2": ext_r2,
                "retained_r2_fraction": retained_r2_frac,
                "internal_mae": internal_mae, "external_mae": ext_mae,
                "mae_inflation_ratio": mae_inflation_ratio,
            }
        degradation_by_model["targets"][target][model_name] = per_functional
    log(f"  {target} (PBE): retained-R2-fraction by model: " + ", ".join(
        f"{m}={degradation_by_model['targets'][target][m]['pbe']['retained_r2_fraction']:.2f}"
        for m in emb[target] if "pbe" in degradation_by_model["targets"][target][m]))
OUT["normalized_transfer_degradation"] = degradation_by_model
json.dump(OUT, open(DATA / "extended_robustness_and_sensitivity_checks.json", "w"), indent=2, default=str)

log("ALL DONE")
