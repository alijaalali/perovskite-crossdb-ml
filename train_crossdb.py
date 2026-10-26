"""
Phase 1 core test: train XGBoost on Materials Project perovskite oxides,
evaluate (a) an MP-internal random 80/20 split, (b) zero-shot transfer to
each of the three Alexandria functional variants (PBE/PBEsol/SCAN), held out
entirely -- no Alexandria data ever enters training.

Two targets modeled independently: formation_energy_per_atom (eV/atom) and
band_gap (eV).

Also checks the DFT-functional-mismatch confound flagged in HANDOFF.md:
does transfer error correlate with transition-metal (B-site) content, where
MP's Hubbard +U correction diverges most from Alexandria's plain-PBE run?
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

PROJ = Path("/glade/work/ajalali/Perovskite-CrossDB-ML-project")
DATA = PROJ / "data"
OUT = PROJ / "models" / "crossdb"
OUT.mkdir(parents=True, exist_ok=True)

TRANSITION_METALS = set(
    "Sc Ti V Cr Mn Fe Co Ni Cu Zn Y Zr Nb Mo Tc Ru Rh Pd Ag Cd "
    "Hf Ta W Re Os Ir Pt Au Hg".split()
)

TARGETS = ["formation_energy_per_atom", "band_gap"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_feature_cols(df):
    magpie = [c for c in df.columns if c.startswith("MagpieData")]
    struct = ["space_group_number", "tolerance_factor", "a", "b", "c",
              "alpha", "beta", "gamma", "volume", "nsites"]
    struct = [c for c in struct if c in df.columns]
    return magpie + struct


def load_mp():
    df = pd.read_parquet(DATA / "mp_perovskites.parquet")
    return df


def load_alexandria(functional):
    df = pd.read_parquet(DATA / f"alexandria_{functional}.parquet")
    return df


def align_features(train_df, test_df, feature_cols):
    """Ensure identical feature columns/order; fill any missing (e.g. a magpie
    stat that returned NaN for one but not the other) with the train mean."""
    X_train = train_df[feature_cols].to_numpy(dtype=float)
    X_test = test_df[feature_cols].to_numpy(dtype=float)
    col_mean = np.nanmean(X_train, axis=0)
    X_train = np.where(np.isnan(X_train), col_mean, X_train)
    X_test = np.where(np.isnan(X_test), col_mean, X_test)
    return X_train, X_test


def fit_and_eval(X_train, y_train, X_test, y_test, n_estimators=300):
    model = XGBRegressor(
        n_estimators=n_estimators, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
        n_jobs=8, random_state=42,
    )
    model.fit(X_train, y_train, verbose=False)
    pred = model.predict(X_test)
    metrics = {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "r2": float(r2_score(y_test, pred)) if len(y_test) > 1 else None,
    }
    return model, metrics, pred


def has_transition_metal(row):
    return int(row["a_site"] in TRANSITION_METALS or row["b_site"] in TRANSITION_METALS)


def main():
    t0 = time.time()
    mp_df = load_mp()
    feature_cols = get_feature_cols(mp_df)
    log(f"MP table: {mp_df.shape}, {len(feature_cols)} features")

    alex = {f: load_alexandria(f) for f in ["pbe", "pbesol", "scan"]}
    for f, df in alex.items():
        log(f"Alexandria {f}: {df.shape}")

    all_results = {}

    for target in TARGETS:
        log(f"=== Target: {target} ===")
        target_results = {}

        y_mp = mp_df[target].to_numpy(dtype=float)

        # --- (a) MP-internal random 80/20 split ---
        rng = np.random.RandomState(42)
        idx = rng.permutation(len(mp_df))
        n_test = int(0.2 * len(mp_df))
        test_idx, train_idx = idx[:n_test], idx[n_test:]
        X_all = mp_df[feature_cols].to_numpy(dtype=float)
        col_mean = np.nanmean(X_all[train_idx], axis=0)
        X_all = np.where(np.isnan(X_all), col_mean, X_all)
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_mp[train_idx], y_mp[test_idx]
        model_internal, m_internal, _ = fit_and_eval(X_train, y_train, X_test, y_test)
        target_results["mp_internal_split"] = m_internal
        log(f"  MP-internal split: {m_internal}")

        # --- (b) train on ALL of MP, transfer to each Alexandria functional ---
        X_train_full, _ = align_features(mp_df, mp_df, feature_cols)
        y_train_full = y_mp
        model_full = XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
            n_jobs=8, random_state=42,
        )
        model_full.fit(X_train_full, y_train_full, verbose=False)

        transfer_results = {}
        transfer_preds = {}
        for functional, adf in alex.items():
            adf_valid = adf.dropna(subset=[target]).reset_index(drop=True)
            if len(adf_valid) == 0:
                continue
            _, X_test_a = align_features(mp_df, adf_valid, feature_cols)
            y_test_a = adf_valid[target].to_numpy(dtype=float)
            pred_a = model_full.predict(X_test_a)
            m = {
                "n_test": int(len(y_test_a)),
                "mae": float(mean_absolute_error(y_test_a, pred_a)),
                "rmse": float(np.sqrt(mean_squared_error(y_test_a, pred_a))),
                "r2": float(r2_score(y_test_a, pred_a)) if len(y_test_a) > 1 else None,
            }
            transfer_results[functional] = m
            log(f"  MP -> Alexandria({functional}): {m}")

            # transition-metal confound check
            has_tm = adf_valid.apply(has_transition_metal, axis=1).to_numpy()
            err = np.abs(y_test_a - pred_a)
            if has_tm.sum() > 3 and (~has_tm.astype(bool)).sum() > 3:
                tm_mae = float(err[has_tm.astype(bool)].mean())
                non_tm_mae = float(err[~has_tm.astype(bool)].mean())
                transfer_results[functional]["tm_confound"] = {
                    "n_transition_metal": int(has_tm.sum()),
                    "n_non_transition_metal": int((~has_tm.astype(bool)).sum()),
                    "mae_transition_metal": tm_mae,
                    "mae_non_transition_metal": non_tm_mae,
                }
                log(f"    TM confound check: TM-containing MAE={tm_mae:.4f} "
                    f"(n={int(has_tm.sum())}), non-TM MAE={non_tm_mae:.4f} "
                    f"(n={int((~has_tm.astype(bool)).sum())})")

            transfer_preds[functional] = pd.DataFrame({
                "id": adf_valid["id"], "formula": adf_valid["formula"],
                "y_true": y_test_a, "y_pred": pred_a,
                "has_transition_metal": has_tm,
            })

        target_results["alexandria_transfer"] = transfer_results

        for functional, pdf in transfer_preds.items():
            pdf.to_csv(OUT / f"predictions_{target}_{functional}.csv", index=False)

        all_results[target] = target_results

    with open(OUT / "metrics.json", "w") as fh:
        json.dump(all_results, fh, indent=2, default=str)
    log(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
