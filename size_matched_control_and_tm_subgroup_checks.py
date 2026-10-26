# Size-matched random-removal control: removes a random 625 of the 1,684 MP
# compounds (the same count as the Alexandria-overlapping compounds removed for the
# excluded-composition retrain), repeated over 20 seeds, to separate the effect of
# losing training data from the effect of specifically removing Alexandria-overlapping
# compositions. Also compares the PBEsol offset between transition-metal-containing
# and non-transition-metal compounds, with a bootstrap CI on the difference.
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = Path("/glade/work/ajalali/13 - Perovskite-CrossDB-ML-project")
DATA = ROOT / "data"
RNG = 42
N_BOOT = 2000
TARGETS = ["formation_energy_per_atom", "band_gap"]

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

OUTFILE = DATA / "size_matched_control_and_tm_subgroup_checks.json"
OUT = json.load(open(OUTFILE)) if OUTFILE.exists() else {}

mp = pd.read_parquet(DATA / "mp_perovskites.parquet")
alex = {f: pd.read_parquet(DATA / f"alexandria_{f}.parquet") for f in ["pbe", "pbesol", "scan"]}

mp["reduced_formula"] = mp["formula"].apply(lambda f: Composition(f).reduced_formula)
for f in alex:
    alex[f]["reduced_formula"] = alex[f]["formula"].apply(lambda x: Composition(x).reduced_formula)

alex_all_formulas = set()
for f in alex:
    alex_all_formulas |= set(alex[f]["reduced_formula"])

mp_excl = mp[~mp["reduced_formula"].isin(alex_all_formulas)].reset_index(drop=True)
n_removed = len(mp) - len(mp_excl)
log(f"Composition-overlap exclusion removes {n_removed} of {len(mp)} MP compounds.")

def get_feature_cols(df):
    magpie = [c for c in df.columns if c.startswith("MagpieData")]
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

def fit_xgb(Xtr, ytr, seed):
    model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
                          n_jobs=8, random_state=seed)
    model.fit(Xtr, ytr)
    return model

def metrics(y, p):
    return {"n": int(len(y)), "mae": float(mean_absolute_error(y, p)),
            "rmse": float(np.sqrt(mean_squared_error(y, p))), "r2": float(r2_score(y, p))}

cols_full = get_feature_cols(mp)

# ============================================================
# Size-matched random-removal control
# ============================================================
log("=== Size-matched random-removal control ===")
N_SEEDS = 20
random_removal = {"n_mp_total": int(len(mp)), "n_removed": int(n_removed),
                   "n_seeds": N_SEEDS, "seed_results": {t: {f: [] for f in alex} for t in TARGETS}}

rng_master = np.random.RandomState(RNG)
for seed in range(N_SEEDS):
    rand_drop_idx = rng_master.choice(len(mp), size=n_removed, replace=False)
    mp_rand = mp.drop(index=mp.index[rand_drop_idx]).reset_index(drop=True)
    assert len(mp_rand) == len(mp_excl)
    for target in TARGETS:
        ytr = mp_rand[target].to_numpy(dtype=float)
        Xtr, _ = align(mp_rand, mp_rand, cols_full)
        model = fit_xgb(Xtr, ytr, seed=1000 + seed)
        for f in alex:
            test_df = alex[f].dropna(subset=[target])
            _, Xte = align(mp_rand, test_df, cols_full)
            pred = model.predict(Xte)
            yte = test_df[target].to_numpy(dtype=float)
            m = metrics(yte, pred)
            random_removal["seed_results"][target][f].append(m)
    log(f"  seed {seed} done")

random_removal["summary"] = {t: {} for t in TARGETS}
for target in TARGETS:
    for f in alex:
        r2s = np.array([r["r2"] for r in random_removal["seed_results"][target][f]])
        maes = np.array([r["mae"] for r in random_removal["seed_results"][target][f]])
        random_removal["summary"][target][f] = {
            "r2_mean": float(r2s.mean()), "r2_std": float(r2s.std()),
            "mae_mean": float(maes.mean()), "mae_std": float(maes.std()),
        }
        log(f"  random-removal control {target}/{f}: R2={r2s.mean():.3f}+-{r2s.std():.3f} "
            f"MAE={maes.mean():.4f}+-{maes.std():.4f}")

OUT["size_matched_random_removal_control"] = random_removal
json.dump(OUT, open(OUTFILE, "w"), indent=2, default=str)
log("Random-removal control written.")

# ============================================================
# PBEsol offset by transition-metal vs non-transition-metal subgroup, with
# bootstrap CI
# ============================================================
log("=== PBEsol offset by transition-metal subgroup ===")
pdf = pd.read_csv(DATA / "pbesol_offset_detail.csv")
tm = pdf.loc[pdf["has_tm_old_proxy"] == 1, "residual"].to_numpy()
non_tm = pdf.loc[pdf["has_tm_old_proxy"] == 0, "residual"].to_numpy()

rng = np.random.RandomState(RNG)
diffs = []
for _ in range(N_BOOT):
    tm_s = tm[rng.randint(0, len(tm), len(tm))]
    non_tm_s = non_tm[rng.randint(0, len(non_tm), len(non_tm))]
    diffs.append(tm_s.mean() - non_tm_s.mean())
diffs = np.array(diffs)

tm_subgroup = {
    "note": ("has_tm_old_proxy flags ANY transition metal on the A or B site "
             "(broader than the 8-element Hubbard-+U-applicable set reported "
             "elsewhere); residuals and flag are pre-computed in "
             "data/pbesol_offset_detail.csv."),
    "mean_residual_tm_containing": float(tm.mean()), "n_tm_containing": int(len(tm)),
    "mean_residual_non_tm": float(non_tm.mean()), "n_non_tm": int(len(non_tm)),
    "tm_minus_non_tm_diff_mean": float(diffs.mean()),
    "tm_minus_non_tm_diff_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
}
log(f"  TM-containing: mean residual={tm.mean():.4f} (n={len(tm)}); "
    f"non-TM: mean residual={non_tm.mean():.4f} (n={len(non_tm)}); "
    f"diff={diffs.mean():.4f}, 95% CI=[{np.percentile(diffs,2.5):.4f}, {np.percentile(diffs,97.5):.4f}]")

OUT["pbesol_offset_tm_subgroup"] = tm_subgroup
json.dump(OUT, open(OUTFILE, "w"), indent=2, default=str)
log("Done.")
