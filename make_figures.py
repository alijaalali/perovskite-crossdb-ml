# Figure generation for the perovskite cross-database ML manuscript. Regenerates
# every figure from the single authoritative data source (extended_cross_database_
# analysis.json + final_model_predictions_*.csv) so no figure can drift out of sync
# with the tables/text.
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/glade/work/ajalali/13 - Perovskite-CrossDB-ML-project")
MDIR = ROOT / "13th - Perovskite-CrossDB-ML-project"
DATA = ROOT / "data"
FIG = MDIR / "figures"
FIG.mkdir(exist_ok=True, parents=True)

plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                      "axes.spines.right": False, "figure.dpi": 150})

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
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    names = ["Materials\nProject", "Alexandria\nPBE", "Alexandria\nPBEsol", "Alexandria\nSCAN"]
    counts = [len(mp), len(alex["pbe"]), len(alex["pbesol"]), len(alex["scan"])]
    colors = [GRAY, BLUE, RED, GREEN]
    bars = ax.bar(names, counts, color=colors, width=0.6)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 15, str(c), ha="center", fontsize=10)
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
    ax.legend(fontsize=8, frameon=False)
    ax.set_xlim(0.6, 1.1)
    fig.tight_layout()
    save(fig, "fig1_dataset_overview")


def figs1_training_distributions():
    mp = pd.read_parquet(DATA / "mp_perovskites.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
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
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True, sharey=True)
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
        ax.set_xlabel(f"DFT {xlabel_prefix}")
    axes[0].set_ylabel(f"XGBoost-predicted {xlabel_prefix}\n(trained on Materials Project only)")
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
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
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
    axes[0].legend(loc="lower left", fontsize=8, frameon=False)
    fig.suptitle("Cross-database transfer: XGBoost trained on Materials Project,\n"
                 "evaluated zero-shot on three Alexandria functional variants", fontsize=11)
    fig.tight_layout()
    save(fig, "fig3_crossdb_summary")


def _model_benchmark(target, fig_name, title):
    models = list(BENCH[target].keys())
    funcs = ["pbe", "pbesol", "scan"]
    data = np.array([[BENCH[target][m]["external"][f]["r2"] for f in funcs] for m in models])
    # Sort by each model's INTERNAL (MP 80/20) R^2, matching the order used in the
    # companion SI tables, rather than by mean external R^2 -- the two sort keys
    # disagree for several models, so this keeps the figure's bar order consistent
    # with its companion table.
    internal_r2 = np.array([BENCH[target][m]["internal"]["r2"] for m in models])
    order = np.argsort(-internal_r2)
    models_sorted = [models[i] for i in order]
    data = data[order]
    fig, ax = plt.subplots(figsize=(8, 5.5))
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
    ax.legend(fontsize=9, frameon=False, loc="lower right")
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


if __name__ == "__main__":
    fig1_overview()
    figs1_training_distributions()
    fig2_transfer_parity_formation_energy()
    figs3_transfer_parity_band_gap()
    fig3_crossdb_summary()
    figs2_model_benchmark_formation_energy()
    figs4_model_benchmark_band_gap()
    print("ALL FIGURES REGENERATED")
