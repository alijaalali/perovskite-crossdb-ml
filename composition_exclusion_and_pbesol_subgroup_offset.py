# Excluded-composition retrain: removes every MP compound sharing a reduced
# composition with any Alexandria test set, retrains, and re-evaluates on the full
# Alexandria sets. Also compares a global PBEsol offset correction to a
# category-specific (Hubbard+U vs. non-+U) one. Appends to the connectivity-checks
# JSON written by connectivity_validation.py.
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor

ROOT = Path("/glade/work/ajalali/13 - Perovskite-CrossDB-ML-project")
DATA = ROOT / "data"
RNG = 42
N_BOOT = 2000

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

OUTFILE = DATA / "connectivity_and_robustness_checks.json"
OUT = json.load(open(OUTFILE)) if OUTFILE.exists() else {}

mp = pd.read_parquet(DATA / "mp_perovskites.parquet")
alex = {f: pd.read_parquet(DATA / f"alexandria_{f}.parquet") for f in ["pbe", "pbesol", "scan"]}

def get_feature_cols(df, mode="full"):
    magpie = [c for c in df.columns if c.startswith("MagpieData")]
    if mode == "composition_only":
        return magpie
    if mode == "lattice_independent":
        # composition + tolerance factor + one-hot space group (categorical: space
        # group numbers are identifiers, not a continuous physical ordering), no
        # raw lattice constants/volume.
        return magpie + ["tolerance_factor"]
    struct = ["space_group_number", "tolerance_factor", "a", "b", "c",
              "alpha", "beta", "gamma", "volume", "nsites"]
    return magpie + [c for c in struct if c in df.columns]


def onehot_sg(train_df, test_dfs, sg_col="space_group_number"):
    """One-hot encode space group using categories observed in MP (training)
    only; unseen categories in test sets map to an all-zero row."""
    cats = sorted(train_df[sg_col].dropna().unique())
    def enc(df):
        mat = np.zeros((len(df), len(cats)))
        for i, v in enumerate(df[sg_col].values):
            if v in cats:
                mat[i, cats.index(v)] = 1.0
        return mat
    return enc(train_df), {k: enc(v) for k, v in test_dfs.items()}, cats


def align(train_df, test_df, cols):
    Xtr = train_df[cols].to_numpy(dtype=float)
    Xte = test_df[cols].to_numpy(dtype=float)
    m = np.nanmean(Xtr, axis=0)
    Xtr = np.where(np.isnan(Xtr), m, Xtr)
    Xte = np.where(np.isnan(Xte), m, Xte)
    return Xtr, Xte


def fit_xgb(Xtr, ytr, **kw):
    model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
                          n_jobs=8, random_state=RNG)
    model.fit(Xtr, ytr)
    return model


def metrics(y, p):
    return {"n": int(len(y)), "mae": float(mean_absolute_error(y, p)),
            "rmse": float(np.sqrt(mean_squared_error(y, p))), "r2": float(r2_score(y, p))}


def bootstrap_ci(y, p, n=N_BOOT, seed=RNG):
    rng = np.random.RandomState(seed)
    y, p = np.asarray(y), np.asarray(p)
    maes = []
    for _ in range(n):
        idx = rng.randint(0, len(y), len(y))
        maes.append(mean_absolute_error(y[idx], p[idx]))
    return float(np.mean(maes)), float(np.percentile(maes, 2.5)), float(np.percentile(maes, 97.5))


def paired_bootstrap_mae_diff(y, p1, p2, n=N_BOOT, seed=RNG):
    """mean(|p1-y|) - mean(|p2-y|), bootstrap CI."""
    rng = np.random.RandomState(seed)
    y, p1, p2 = np.asarray(y), np.asarray(p1), np.asarray(p2)
    diffs = []
    for _ in range(n):
        idx = rng.randint(0, len(y), len(y))
        d = mean_absolute_error(y[idx], p1[idx]) - mean_absolute_error(y[idx], p2[idx])
        diffs.append(d)
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


TARGETS = ["formation_energy_per_atom", "band_gap"]

# ============================================================
# Excluded-composition retrain: remove Alexandria-overlapping compositions from
# MP training, evaluate on complete Alexandria sets
# ============================================================
log("=== Excluded-composition retrain ===")
mp["reduced_formula"] = mp["formula"].apply(lambda f: Composition(f).reduced_formula)
for f in alex:
    alex[f]["reduced_formula"] = alex[f]["formula"].apply(lambda x: Composition(x).reduced_formula)

alex_all_formulas = set()
for f in alex:
    alex_all_formulas |= set(alex[f]["reduced_formula"])

mp_excl = mp[~mp["reduced_formula"].isin(alex_all_formulas)].reset_index(drop=True)
log(f"MP: {len(mp)} total -> {len(mp_excl)} after excluding all Alexandria-overlapping compositions "
    f"({len(mp) - len(mp_excl)} removed)")

excl_retrain = {"n_mp_before": int(len(mp)), "n_mp_after_exclusion": int(len(mp_excl)), "targets": {}}
cols_full = get_feature_cols(mp, "full")
for target in TARGETS:
    excl_retrain["targets"][target] = {}
    ytr = mp_excl[target].to_numpy(dtype=float)
    Xtr, _ = align(mp_excl, mp_excl, cols_full)
    model = fit_xgb(Xtr, ytr)
    for f in alex:
        test_df = alex[f].dropna(subset=[target])
        _, Xte = align(mp_excl, test_df, cols_full)
        pred = model.predict(Xte)
        yte = test_df[target].to_numpy(dtype=float)
        m = metrics(yte, pred)
        excl_retrain["targets"][target][f] = m
        log(f"  {target} {f}: n={m['n']} MAE={m['mae']:.4f} R2={m['r2']:.3f}")
OUT["excluded_composition_retrain"] = excl_retrain
json.dump(OUT, open(OUTFILE, "w"), indent=2, default=str)

# Also report MAE+CI for the tiny novel-composition subsets from the original
# (non-excluded) final model.
log("Novel-composition-subset CIs from the full-MP-trained final model...")
novel_ci = {}
for target in TARGETS:
    novel_ci[target] = {}
    for f in alex:
        pred_file = DATA / f"final_model_predictions_{target}_{f}.csv"
        if not pred_file.exists():
            continue
        pdf = pd.read_csv(pred_file)
        test_df = alex[f].dropna(subset=[target]).reset_index(drop=True)
        if len(pdf) != len(test_df):
            continue
        pdf["reduced_formula"] = test_df["reduced_formula"].values
        mp_formulas = set(mp["reduced_formula"])
        novel_mask = ~pdf["reduced_formula"].isin(mp_formulas)
        sub = pdf[novel_mask]
        if len(sub) == 0:
            continue
        mae_mean, lo, hi = bootstrap_ci(sub["y_true"], sub["y_pred"])
        novel_ci[target][f] = {"n": int(len(sub)), "mae_bootstrap_mean": mae_mean,
                                "mae_ci95": [lo, hi]}
        log(f"  {target} {f} novel subset: n={len(sub)} MAE={mae_mean:.4f} CI=[{lo:.4f},{hi:.4f}]")
OUT["novel_subset_bootstrap_cis"] = novel_ci
json.dump(OUT, open(OUTFILE, "w"), indent=2, default=str)

# ============================================================
# Separate PBEsol offset correction for +U vs non-+U compounds, with the same
# calibration/held-out split protocol as the global offset correction
# ============================================================
log("=== Category-specific PBEsol offset correction ===")
pbesol_df = pd.read_csv(DATA / "pbesol_offset_detail.csv")

def calibrated_offset_mae(df, n_splits=30, seed=RNG):
    rng = np.random.RandomState(seed)
    maes = []
    offsets = []
    for i in range(n_splits):
        cal, held = train_test_split(df, test_size=0.5, random_state=rng.randint(0, 1_000_000))
        offset = cal["residual"].mean()
        corrected = held["y_pred"] - offset
        maes.append(mean_absolute_error(held["y_true"], corrected))
        offsets.append(offset)
    return float(np.mean(maes)), float(np.std(maes)), float(np.mean(offsets)), float(np.std(offsets))

mae_global, mae_global_sd, off_global, off_global_sd = calibrated_offset_mae(pbesol_df)

def calibrated_offset_mae_by_category(df, n_splits=30, seed=RNG):
    rng = np.random.RandomState(seed)
    maes = []
    for i in range(n_splits):
        seed_i = rng.randint(0, 1_000_000)
        corrected_vals, true_vals = [], []
        for cat_val in [0, 1]:
            sub = df[df["has_hubbard_u"] == cat_val]
            if len(sub) < 4:
                continue
            cal, held = train_test_split(sub, test_size=0.5, random_state=seed_i)
            offset = cal["residual"].mean()
            corrected_vals.extend((held["y_pred"] - offset).tolist())
            true_vals.extend(held["y_true"].tolist())
        maes.append(mean_absolute_error(true_vals, corrected_vals))
    return float(np.mean(maes)), float(np.std(maes))

mae_cat, mae_cat_sd = calibrated_offset_mae_by_category(pbesol_df)

offset_by_category = {
    "global_offset": {"corrected_mae_mean": mae_global, "corrected_mae_sd": mae_global_sd,
                       "offset_mean": off_global, "offset_sd": off_global_sd},
    "category_specific_offset": {"corrected_mae_mean": mae_cat, "corrected_mae_sd": mae_cat_sd},
    "raw_offsets_by_category": {
        "hubbard_u": float(pbesol_df[pbesol_df.has_hubbard_u == 1]["residual"].mean()),
        "non_hubbard_u": float(pbesol_df[pbesol_df.has_hubbard_u == 0]["residual"].mean()),
        "difference": float(pbesol_df[pbesol_df.has_hubbard_u == 1]["residual"].mean()
                             - pbesol_df[pbesol_df.has_hubbard_u == 0]["residual"].mean()),
    },
}
log(f"Global-offset corrected MAE: {mae_global:.4f}+-{mae_global_sd:.4f}")
log(f"Category-specific corrected MAE: {mae_cat:.4f}+-{mae_cat_sd:.4f}")
OUT["pbesol_offset_category_specific"] = offset_by_category
json.dump(OUT, open(OUTFILE, "w"), indent=2, default=str)

print("DONE: excluded-composition retrain + category-specific PBEsol offset")
