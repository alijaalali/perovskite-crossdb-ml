# Figure generation for the perovskite cross-database ML manuscript. Regenerates
# every figure from the single authoritative data source (extended_cross_database_
# analysis.json + final_model_predictions_*.csv) so no figure can drift out of sync
# with the tables/text.
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("PEROVSKITE_HOME", Path(__file__).resolve().parent))
DATA = ROOT / "data"
FIG = Path(os.environ.get("FIGURE_DIR", ROOT / "figures"))
FIG.mkdir(exist_ok=True, parents=True)

plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
                      "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                      "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})

BLUE, RED, GREEN, GRAY, ORANGE, PURPLE = (
    "#2166ac", "#b2182b", "#1a9850", "#666666", "#e08214", "#762a83")
FUNCTIONAL_COLORS = {"pbe": BLUE, "pbesol": RED, "scan": GREEN}
FUNCTIONAL_LABELS = {"pbe": "PBE", "pbesol": "PBEsol", "scan": "SCAN"}


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"saved {name}")


ANALYSIS = json.load(open(DATA / "extended_cross_database_analysis.json"))
BENCH = ANALYSIS["extended_model_benchmark"]


def fig1_overview():
    mp = pd.read_parquet(DATA / "mp_perovskites.parquet")
    alex = {f: pd.read_parquet(DATA / f"alexandria_{f}.parquet") for f in ["pbe", "pbesol", "scan"]}
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.9))
    ax = axes[0]
    names = ["Materials\nProject", "Alexandria\nPBE", "Alexandria\nPBEsol", "Alexandria\nSCAN"]
    counts = [len(mp), len(alex["pbe"]), len(alex["pbesol"]), len(alex["scan"])]
    colors = [GRAY, BLUE, RED, GREEN]
    bars = ax.bar(names, counts, color=colors, width=0.6)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 15, str(c), ha="center", fontsize=8)
    ax.set_ylabel("Number of ABO$_3$ perovskite oxides")
    ax.set_title("(a) Sample sizes")
    ax.set_ylim(0, max(counts) * 1.15)
    ax = axes[1]
    ax.hist(mp["tolerance_factor"], bins=30, density=True, color=GRAY, alpha=0.5,
            label=f"Materials Project (n={len(mp)})")
    for f, adf in alex.items():
        ax.hist(adf["tolerance_factor"], bins=20, density=True, histtype="step",
                linewidth=1.8, color=FUNCTIONAL_COLORS[f],
                label=f"Alexandria {FUNCTIONAL_LABELS[f]} (n={len(adf)})")
    ax.set_xlabel("Goldschmidt tolerance factor $t$")
    ax.set_ylabel("Probability density")
    ax.set_title("(b) Structural coverage overlap")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_ylim(0, 12)
    ax.set_xlim(0.6, 1.1)
    fig.tight_layout()
    save(fig, "fig1_dataset_overview")


def figs1_training_distributions():
    mp = pd.read_parquet(DATA / "mp_perovskites.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))
    axes[0].hist(mp["formation_energy_per_atom"], bins=40, color=GRAY)
    axes[0].set_xlabel("Formation energy (eV/atom)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("(a) MP training distribution: formation energy")
    axes[1].hist(mp["band_gap"], bins=40, color=GRAY)
    axes[1].set_xlabel("Band gap (eV)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("(b) MP training distribution: band gap")
    fig.tight_layout()
    save(fig, "figs1_mp_training_distributions")


def _parity(target, fname, xlabel_prefix, fig_name, xlim_override=None):
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.9), sharex=True, sharey=True)
    for ax, f in zip(axes, ["pbe", "pbesol", "scan"]):
        df = pd.read_csv(DATA / f"final_model_predictions_{target}_{f}.csv")
        ax.scatter(df["y_true"], df["y_pred"], s=14, alpha=0.6,
                   color=FUNCTIONAL_COLORS[f], edgecolors="none")
        if xlim_override:
            lims = xlim_override(df)
        else:
            lims = [min(df["y_true"].min(), df["y_pred"].min()) - 0.2,
                    max(df["y_true"].max(), df["y_pred"].max()) + 0.2]
        ax.plot(lims, lims, "k--", linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        r2 = BENCH[target]["XGBoost"]["external"][f]["r2"]
        ax.set_title(f"Alexandria {FUNCTIONAL_LABELS[f]}\n$R^2$={r2:.3f}")
        ax.set_xlabel(f"DFT {xlabel_prefix}", fontsize=8)
    axes[0].set_ylabel(f"XGBoost-predicted {xlabel_prefix}\n(trained on Materials Project only)", fontsize=8)
    fig.tight_layout()
    save(fig, fig_name)


def fig2_transfer_parity_formation_energy():
    _parity("formation_energy_per_atom", None, "formation energy (eV/atom)",
            "fig2_transfer_parity_formation_energy")


def figs3_transfer_parity_band_gap():
    _parity("band_gap", None, "band gap (eV)", "figs3_transfer_parity_band_gap",
            xlim_override=lambda df: [-0.3, max(df["y_true"].max(), df["y_pred"].max()) + 0.3])


def fig3_crossdb_summary():
    funcs = ["pbe", "pbesol", "scan"]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.3))
    for ax, target, label in zip(axes, ["formation_energy_per_atom", "band_gap"],
                                  ["Formation energy (eV/atom)", "Band gap (eV)"]):
        xgb = BENCH[target]["XGBoost"]
        r2 = [xgb["external"][f]["r2"] for f in funcs]
        mae = [xgb["external"][f]["mae"] for f in funcs]
        ref_r2 = xgb["internal"]["r2"]
        ref_mae = xgb["internal"]["mae"]
        x = np.arange(len(funcs))
        width = 0.35
        ax2 = ax.twinx()
        ax.bar(x - width / 2, r2, width, color=BLUE, label="$R^2$ (transfer)")
        ax2.bar(x + width / 2, mae, width, color=ORANGE, label="MAE (transfer)")
        ax.axhline(ref_r2, color=BLUE, linestyle="--", linewidth=1.3, label="$R^2$ (MP-internal)")
        ax2.axhline(ref_mae, color=ORANGE, linestyle="--", linewidth=1.3, label="MAE (MP-internal)")
        ax.set_xticks(x)
        ax.set_xticklabels([FUNCTIONAL_LABELS[f] for f in funcs])
        ax.set_ylabel("$R^2$", color=BLUE)
        ax2.set_ylabel(f"MAE, {label.split('(')[1][:-1]}", color=ORANGE)
        ax.tick_params(axis="y", labelcolor=BLUE)
        ax2.tick_params(axis="y", labelcolor=ORANGE)
        ax.set_ylim(0, 1.05)
        ax.set_title(label)
    fig.suptitle("Cross-database transfer: XGBoost trained on Materials Project,\n"
                 "evaluated zero-shot on three Alexandria functional variants", fontsize=9)
    fig.tight_layout()
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.1, -0.1), fontsize=8, frameon=False, ncol=2)
    save(fig, "fig3_crossdb_summary")


def _model_benchmark(target, fig_name, title):
    models = list(BENCH[target].keys())
    funcs = ["pbe", "pbesol", "scan"]
    data = np.array([[BENCH[target][m]["external"][f]["r2"] for f in funcs] for m in models])
    # sorted by internal R^2, the order used in the Supplementary Material tables
    internal_r2 = np.array([BENCH[target][m]["internal"]["r2"] for m in models])
    order = np.argsort(-internal_r2)
    models_sorted = [models[i] for i in order]
    data = data[order]
    fig, ax = plt.subplots(figsize=(4.9, 3.6))
    y = np.arange(len(models_sorted))
    width = 0.25
    for i, f in enumerate(funcs):
        ax.barh(y + (i - 1) * width, data[:, i], height=width, color=FUNCTIONAL_COLORS[f],
                label=FUNCTIONAL_LABELS[f])
    ax.set_yticks(y)
    ax.set_yticklabels(models_sorted)
    ax.invert_yaxis()
    ax.set_xlabel("$R^2$, MP $\\to$ Alexandria transfer")
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.16), ncol=3)
    if target == "band_gap":
        ax.axvline(0, color="black", linewidth=0.8)
    else:
        ax.set_xlim(0, 1.05)
    fig.tight_layout()
    save(fig, fig_name)


def figs2_model_benchmark_formation_energy():
    _model_benchmark("formation_energy_per_atom", "figs2_model_benchmark_formation_energy",
                      "Formation-energy transfer across 11 model families")


def figs4_model_benchmark_band_gap():
    _model_benchmark("band_gap", "figs4_model_benchmark_band_gap",
                      "Band-gap transfer across 11 model families")


def fig6_overlap():
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2))
    funcs = ["pbe", "pbesol", "scan"]
    ax = axes[0]
    ov = ANALYSIS["composition_overlap"]
    x = np.arange(3)
    overlap = [ov[f]["n_overlap"] for f in funcs]
    novel = [ov[f]["n_novel"] for f in funcs]
    ax.bar(x, overlap, color=GRAY, label="Shares a composition with an MP training compound")
    ax.bar(x, novel, bottom=overlap, color=ORANGE, label="Composition-novel (not in MP)")
    for i, f in enumerate(funcs):
        ax.text(i, ov[f]["n_total"] + 18, f"{ov[f]['pct_overlap']:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([FUNCTIONAL_LABELS[f] for f in funcs])
    ax.set_ylim(0, 340)
    ax.set_ylabel("Number of Alexandria test compounds")
    ax.set_title("(a) Composition overlap, Alexandria vs.\nMaterials Project training set")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1)
    ax = axes[1]
    acc = ANALYSIS["overlap_split_accuracy"]["formation_energy_per_atom"]
    w = 0.35
    r2_ov = [acc[f]["overlapping"]["r2"] for f in funcs]
    r2_nov = [acc[f]["novel"]["r2"] if acc[f]["novel"] else np.nan for f in funcs]
    ax.bar(x - w / 2, r2_ov, w, color=BLUE, label="Composition-overlapping")
    ax.bar(x + w / 2, r2_nov, w, color=RED, label="Composition-novel")
    ax.set_xticks(x)
    ax.set_xticklabels([FUNCTIONAL_LABELS[f] for f in funcs])
    ax.set_ylabel("$R^2$, formation energy")
    ax.set_title("(b) Transfer accuracy: overlapping vs.\nnovel compositions")
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=1)
    fig.tight_layout()
    save(fig, "fig6_composition_overlap")


def fig7_feature_tiers():
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2))
    funcs = ["pbe", "pbesol", "scan"]
    tiers = ["composition_only", "plus_tolerance", "full"]
    tier_labels = ["Composition\nonly (132)", "+ tolerance\nfactor (133)", "+ full\ncrystallographic (142)"]
    colors = [GRAY, ORANGE, BLUE]
    for ax, target, title in zip(
        axes, ["formation_energy_per_atom", "band_gap"],
        ["(a) Formation energy", "(b) Band gap"]
    ):
        x = np.arange(3)
        w = 0.25
        for i, tier in enumerate(tiers):
            vals = [ANALYSIS["feature_tier_comparison"][tier]["targets"][target][f]["r2"] for f in funcs]
            ax.bar(x + (i - 1) * w, vals, w, color=colors[i], label=tier_labels[i])
        ax.set_xticks(x)
        ax.set_xticklabels([FUNCTIONAL_LABELS[f] for f in funcs])
        ax.set_ylabel("$R^2$, MP $\\to$ Alexandria transfer")
        ax.set_title(title)
    fig.tight_layout()
    axes[0].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(1.1, -0.12), ncol=3)
    save(fig, "fig7_feature_tier_comparison")


def fig8_hubbard_corrected():
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2))
    funcs = ["pbe", "pbesol", "scan"]
    for ax, target, title in zip(
        axes, ["formation_energy_per_atom", "band_gap"],
        ["(a) Formation energy (eV/atom)", "(b) Band gap (eV)"]
    ):
        hc = ANALYSIS["hubbard_u_corrected"][target]
        x = np.arange(3)
        w = 0.35
        u_mae = [hc[f]["mae_hubbard_u"] for f in funcs]
        nonu_mae = [hc[f]["mae_non_hubbard_u"] for f in funcs]
        ax.bar(x - w / 2, u_mae, w, color=PURPLE, label="$+U$-applicable (Co,Cr,Fe,Mn,Mo,Ni,V,W)")
        ax.bar(x + w / 2, nonu_mae, w, color=GREEN, label="Not $+U$-applicable")
        ax.set_xticks(x)
        ax.set_xticklabels([FUNCTIONAL_LABELS[f] for f in funcs])
        ax.set_ylabel("MAE (eV/atom)" if target == "formation_energy_per_atom" else "MAE (eV)")
        ax.set_title(title)
        if target == "formation_energy_per_atom":
            ax.set_ylim(0, 0.43)
            ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    save(fig, "fig8_hubbard_u_corrected")


def fig9_pbesol_calibration():
    df = pd.read_csv(DATA / "pbesol_offset_detail.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.1))
    pc = ANALYSIS["pbesol_offset_calibrated"]
    ax = axes[0]
    ax.scatter(df["y_true"], df["residual"], s=14, alpha=0.5, c=np.where(df["has_hubbard_u"] == 1, PURPLE, GREEN))
    ax.axhline(pc["raw_mean_residual_full_set_in_sample"], color="k", ls="--", lw=1,
               label=f"Mean residual = {pc['raw_mean_residual_full_set_in_sample']:.3f}")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("True formation energy (eV/atom)")
    ax.set_ylabel("Residual, predicted $-$ true (eV/atom)")
    ax.set_title("(a) PBEsol residual vs. true value\n(purple = $+U$-applicable, green = not)")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    cats = ["Raw MAE\n(held-out half)", "Corrected MAE\n(calibration/test split)"]
    means = [pc["mae_raw_on_held_out_half_mean"], pc["mae_corrected_mean"]]
    ax.bar(cats, means, color=[RED, BLUE])
    ax.errorbar([1], [pc["mae_corrected_mean"]], yerr=[pc["mae_corrected_std"]], fmt="none",
                color="black", capsize=4)
    ax.set_ylabel("MAE (eV/atom)")
    ax.set_title("(b) Offset correction, 30 random\ncalibration/test splits (mean $\\pm$ s.d.)")
    fig.tight_layout()
    save(fig, "fig9_pbesol_calibration_protocol")


if __name__ == "__main__":
    fig1_overview()
    figs1_training_distributions()
    fig2_transfer_parity_formation_energy()
    figs3_transfer_parity_band_gap()
    fig3_crossdb_summary()
    figs2_model_benchmark_formation_energy()
    figs4_model_benchmark_band_gap()
    fig6_overlap()
    fig7_feature_tiers()
    fig8_hubbard_corrected()
    fig9_pbesol_calibration()
    print("ALL FIGURES REGENERATED")
