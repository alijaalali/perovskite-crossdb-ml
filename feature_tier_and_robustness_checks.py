# Space-group encoding comparison (raw numeric vs. one-hot) and a lattice-independent
# feature tier; paired-bootstrap composition-only vs. full-feature comparison with a
# PBEsol structural-degradation investigation; a controlled internal-vs-external
# comparison at matched training size; a metallic-vs-insulating band-gap split; and an
# ExtraTrees-vs-XGBoost paired comparison (ExtraTrees is XGBoost's closest competitor
# on PBE formation energy). Appends to the connectivity-checks JSON.
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
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

def paired_bootstrap_mae_diff(y, p1, p2, n=N_BOOT, seed=RNG):
    rng = np.random.RandomState(seed)
    y, p1, p2 = np.asarray(y), np.asarray(p1), np.asarray(p2)
    diffs = []
    for _ in range(n):
        idx = rng.randint(0, len(y), len(y))
        d = mean_absolute_error(y[idx], p1[idx]) - mean_absolute_error(y[idx], p2[idx])
        diffs.append(d)
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

TARGETS = ["formation_energy_per_atom", "band_gap"]

def onehot_sg(train_df, test_dfs, sg_col="space_group_number"):
    cats = sorted(train_df[sg_col].dropna().unique())
    def enc(df):
        mat = np.zeros((len(df), len(cats)))
        for i, v in enumerate(df[sg_col].values):
            if v in cats:
                mat[i, cats.index(v)] = 1.0
        return mat
    return enc(train_df), {k: enc(v) for k, v in test_dfs.items()}, cats

# ============================================================
# Space-group encoding (one-hot vs. raw numeric) and a lattice-independent
# feature tier (composition + tolerance + one-hot SG, no raw lattice constants/volume)
# ============================================================
log("=== Space-group encoding + lattice-independent feature tier ===")

magpie_cols = [c for c in mp.columns if c.startswith("MagpieData")]
sg_tier_results = {"targets": {}}
sg_onehot_train, sg_onehot_tests, sg_cats = onehot_sg(mp, alex)
log(f"{len(sg_cats)} distinct space groups one-hot encoded from MP training set")

for target in TARGETS:
    sg_tier_results["targets"][target] = {}
    ytr = mp[target].to_numpy(dtype=float)

    cols_full = get_feature_cols(mp, "full")
    Xtr_full, _ = align(mp, mp, cols_full)
    model_full = fit_xgb(Xtr_full, ytr)

    base_cols_noSG = magpie_cols + ["tolerance_factor", "a", "b", "c", "alpha", "beta", "gamma",
                                     "volume", "nsites"]
    Xtr_base, _ = align(mp, mp, base_cols_noSG)
    Xtr_onehot_full = np.hstack([Xtr_base, sg_onehot_train])

    cols_latind_base = magpie_cols + ["tolerance_factor"]
    Xtr_latind_base, _ = align(mp, mp, cols_latind_base)
    Xtr_latind = np.hstack([Xtr_latind_base, sg_onehot_train])

    model_onehot_full = fit_xgb(Xtr_onehot_full, ytr)
    model_latind = fit_xgb(Xtr_latind, ytr)

    sg_tier_results["targets"][target]["raw_numeric_SG_full"] = {}
    sg_tier_results["targets"][target]["onehot_SG_full"] = {}
    sg_tier_results["targets"][target]["lattice_independent_onehot_SG"] = {}

    for f in alex:
        test_df = alex[f].dropna(subset=[target])
        yte = test_df[target].to_numpy(dtype=float)

        _, Xte_full = align(mp, test_df, cols_full)
        pred_full = model_full.predict(Xte_full)
        sg_tier_results["targets"][target]["raw_numeric_SG_full"][f] = metrics(yte, pred_full)

        _, Xte_base = align(mp, test_df, base_cols_noSG)
        Xte_onehot_full = np.hstack([Xte_base, sg_onehot_tests[f]])
        pred_onehot = model_onehot_full.predict(Xte_onehot_full)
        sg_tier_results["targets"][target]["onehot_SG_full"][f] = metrics(yte, pred_onehot)

        _, Xte_latind_base = align(mp, test_df, cols_latind_base)
        Xte_latind = np.hstack([Xte_latind_base, sg_onehot_tests[f]])
        pred_latind = model_latind.predict(Xte_latind)
        sg_tier_results["targets"][target]["lattice_independent_onehot_SG"][f] = metrics(yte, pred_latind)

        log(f"  {target} {f}: raw-SG R2={metrics(yte,pred_full)['r2']:.3f}  "
            f"onehot-SG R2={metrics(yte,pred_onehot)['r2']:.3f}  "
            f"lattice-indep R2={metrics(yte,pred_latind)['r2']:.3f}")

OUT["space_group_encoding_and_lattice_independent_tier"] = sg_tier_results
json.dump(OUT, open(OUTFILE, "w"), indent=2, default=str)
print("Checkpoint: space-group encoding + lattice-independent tier done")

# ============================================================
# Paired bootstrap CIs, composition-only vs full-feature, every
# functional/target; also investigate PBEsol degradation
# ============================================================
log("=== Paired bootstrap, composition-only vs full-feature ===")
tier_comparison = {"targets": {}}
for target in TARGETS:
    tier_comparison["targets"][target] = {}
    ytr = mp[target].to_numpy(dtype=float)
    cols_comp = get_feature_cols(mp, "composition_only")
    cols_full = get_feature_cols(mp, "full")
    Xtr_comp, _ = align(mp, mp, cols_comp)
    Xtr_full, _ = align(mp, mp, cols_full)
    model_comp = fit_xgb(Xtr_comp, ytr)
    model_full = fit_xgb(Xtr_full, ytr)
    for f in alex:
        test_df = alex[f].dropna(subset=[target])
        yte = test_df[target].to_numpy(dtype=float)
        _, Xte_comp = align(mp, test_df, cols_comp)
        _, Xte_full = align(mp, test_df, cols_full)
        pred_comp = model_comp.predict(Xte_comp)
        pred_full = model_full.predict(Xte_full)
        diff_mean, lo, hi = paired_bootstrap_mae_diff(yte, pred_comp, pred_full)
        tier_comparison["targets"][target][f] = {
            "composition_only": metrics(yte, pred_comp),
            "full_feature": metrics(yte, pred_full),
            "paired_mae_diff_comp_minus_full": {"mean": diff_mean, "ci95": [lo, hi]},
        }
        log(f"  {target} {f}: comp-only MAE={metrics(yte,pred_comp)['mae']:.4f} "
            f"full MAE={metrics(yte,pred_full)['mae']:.4f} "
            f"paired diff={diff_mean:.4f} CI=[{lo:.4f},{hi:.4f}]")
OUT["paired_bootstrap_feature_tiers"] = tier_comparison
json.dump(OUT, open(OUTFILE, "w"), indent=2, default=str)

# PBEsol degradation investigation: does adding structural descriptors specifically
# worsen predictions for a particular subset? Check correlation of (full_error -
# comp_error) with tolerance factor and with the +U flag.
target = "formation_energy_per_atom"
f = "pbesol"
test_df = alex[f].dropna(subset=[target]).reset_index(drop=True)
yte = test_df[target].to_numpy(dtype=float)
cols_comp = get_feature_cols(mp, "composition_only")
cols_full = get_feature_cols(mp, "full")
ytr = mp[target].to_numpy(dtype=float)
Xtr_comp, Xte_comp = align(mp, test_df, cols_comp)
Xtr_full, Xte_full = align(mp, test_df, cols_full)
pred_comp = fit_xgb(Xtr_comp, ytr).predict(Xte_comp)
pred_full = fit_xgb(Xtr_full, ytr).predict(Xte_full)
err_comp = np.abs(pred_comp - yte)
err_full = np.abs(pred_full - yte)
degradation = err_full - err_comp  # positive = full-feature model is worse here
from scipy.stats import pearsonr
tf_vals = test_df["tolerance_factor"].to_numpy(dtype=float)
valid = ~np.isnan(tf_vals)
corr_tf, p_tf = pearsonr(degradation[valid], tf_vals[valid])
pbesol_degradation = {
    "mean_degradation_eV_per_atom": float(degradation.mean()),
    "frac_compounds_worse_with_structure": float((degradation > 0).mean()),
    "pearson_corr_degradation_vs_tolerance_factor": float(corr_tf),
    "pearson_p_value": float(p_tf),
}
log(f"PBEsol degradation investigation: mean degradation={degradation.mean():.4f} eV/atom, "
    f"{(degradation>0).mean()*100:.1f}% of compounds worse with structure, "
    f"corr with tolerance factor r={corr_tf:.3f} (p={p_tf:.3f})")
OUT["pbesol_structural_feature_degradation_investigation"] = pbesol_degradation
json.dump(OUT, open(OUTFILE, "w"), indent=2, default=str)

# ============================================================
# Controlled internal-vs-external: one model trained on the SAME 80% MP
# subset, evaluated on held-out 20% MP AND on full Alexandria
# ============================================================
log("=== Controlled internal-vs-external (matched training size) ===")
matched_training = {"targets": {}}
mp_train80, mp_test20 = train_test_split(mp, test_size=0.2, random_state=RNG)
for target in TARGETS:
    cols_full = get_feature_cols(mp, "full")
    ytr = mp_train80[target].to_numpy(dtype=float)
    Xtr, Xte_internal = align(mp_train80, mp_test20, cols_full)
    model = fit_xgb(Xtr, ytr)
    yte_internal = mp_test20[target].to_numpy(dtype=float)
    pred_internal = model.predict(Xte_internal)
    res = {"n_train": int(len(mp_train80)), "internal_held_out_20pct": metrics(yte_internal, pred_internal)}
    for f in alex:
        test_df = alex[f].dropna(subset=[target])
        _, Xte = align(mp_train80, test_df, cols_full)
        yte = test_df[target].to_numpy(dtype=float)
        pred = model.predict(Xte)
        res[f] = metrics(yte, pred)
    matched_training["targets"][target] = res
    log(f"  {target}: internal(20%)={res['internal_held_out_20pct']['r2']:.3f} "
        f"pbe={res['pbe']['r2']:.3f} pbesol={res['pbesol']['r2']:.3f} scan={res['scan']['r2']:.3f}")
OUT["controlled_internal_vs_external_matched_training_size"] = matched_training
json.dump(OUT, open(OUTFILE, "w"), indent=2, default=str)

# ============================================================
# Zero-gap (metallic) vs non-zero-gap (insulating) split
# ============================================================
log("=== Metallic vs insulating band-gap split ===")
metal_insulator = {}
target = "band_gap"
cols_full = get_feature_cols(mp, "full")
ytr = mp[target].to_numpy(dtype=float)
Xtr, _ = align(mp, mp, cols_full)
model = fit_xgb(Xtr, ytr)
mp_tr, mp_te = train_test_split(mp, test_size=0.2, random_state=RNG)
Xtr_i, Xte_i = align(mp_tr, mp_te, cols_full)
model_i = fit_xgb(Xtr_i, mp_tr[target].to_numpy(dtype=float))
pred_i = model_i.predict(Xte_i)
yte_i = mp_te[target].to_numpy(dtype=float)
zero_mask_i = yte_i < 1e-6
metal_insulator["mp_internal"] = {
    "zero_gap": metrics(yte_i[zero_mask_i], pred_i[zero_mask_i]) if zero_mask_i.sum() > 1 else None,
    "non_zero_gap": metrics(yte_i[~zero_mask_i], pred_i[~zero_mask_i]),
    "n_zero_gap": int(zero_mask_i.sum()), "pct_zero_gap": float(zero_mask_i.mean() * 100),
}
for f in alex:
    test_df = alex[f].dropna(subset=[target])
    yte = test_df[target].to_numpy(dtype=float)
    _, Xte = align(mp, test_df, cols_full)
    pred = model.predict(Xte)
    zero_mask = yte < 1e-6
    metal_insulator[f] = {
        "zero_gap": metrics(yte[zero_mask], pred[zero_mask]) if zero_mask.sum() > 1 else None,
        "non_zero_gap": metrics(yte[~zero_mask], pred[~zero_mask]),
        "n_zero_gap": int(zero_mask.sum()), "pct_zero_gap": float(zero_mask.mean() * 100),
    }
    log(f"  {f}: {zero_mask.sum()} zero-gap ({zero_mask.mean()*100:.1f}%), "
        f"non-zero-gap MAE={metrics(yte[~zero_mask], pred[~zero_mask])['mae']:.3f}")
OUT["metallic_vs_insulating_bandgap_split"] = metal_insulator
json.dump(OUT, open(OUTFILE, "w"), indent=2, default=str)

# ============================================================
# ExtraTrees vs XGBoost paired bootstrap on PBE formation energy (ExtraTrees
# has lower MAE than XGBoost there per the aggregate benchmark numbers --
# needs per-compound predictions for a real paired comparison)
# ============================================================
log("=== ExtraTrees vs XGBoost paired bootstrap ===")
target = "formation_energy_per_atom"
f = "pbe"
cols_full = get_feature_cols(mp, "full")
ytr = mp[target].to_numpy(dtype=float)
Xtr, _ = align(mp, mp, cols_full)
test_df = alex[f].dropna(subset=[target])
yte = test_df[target].to_numpy(dtype=float)
_, Xte = align(mp, test_df, cols_full)

xgb_model = fit_xgb(Xtr, ytr)
et_model = ExtraTreesRegressor(n_estimators=300, max_depth=None, n_jobs=8, random_state=42)
et_model.fit(Xtr, ytr)

pred_xgb = xgb_model.predict(Xte)
pred_et = et_model.predict(Xte)
diff_mean, lo, hi = paired_bootstrap_mae_diff(yte, pred_xgb, pred_et)
extratrees_comparison = {
    "xgboost": metrics(yte, pred_xgb),
    "extratrees": metrics(yte, pred_et),
    "paired_mae_diff_xgb_minus_et": {"mean": diff_mean, "ci95": [lo, hi]},
}
log(f"XGBoost MAE={metrics(yte,pred_xgb)['mae']:.4f}  ExtraTrees MAE={metrics(yte,pred_et)['mae']:.4f}  "
    f"paired diff={diff_mean:.4f} CI=[{lo:.4f},{hi:.4f}]")
OUT["extratrees_vs_xgboost_paired_comparison"] = extratrees_comparison
json.dump(OUT, open(OUTFILE, "w"), indent=2, default=str)

print("DONE: space-group/lattice-independent tier, feature-tier bootstrap, "
      "matched-training-size control, metal-insulator split, ExtraTrees comparison")
