# Cross-database ML for perovskite-oxide formation energy and band gap (Materials Project vs. Alexandria)

Scripts used to build features, train models, run robustness/sensitivity analyses, and generate figures/tables for the manuscript.

## Pipeline order

Set `PEROVSKITE_HOME` to the directory that holds `data/` (default: the directory of the scripts). All outputs are written to `data/`.

1. `build_mp_table.py` - Materials Project perovskite-oxide feature table (needs an `MP_API_KEY` environment variable); the resulting `data/mp_perovskites.parquet` is included
2. `build_alexandria_tables.py` - Alexandria perovskite-oxide feature tables (PBE/PBEsol/SCAN); the resulting `data/alexandria_*.parquet` are included
3. `train_crossdb.py` - first-pass transfer test (outputs to `data/initial_run/`)
4. `benchmark_models.py` - first-pass external-only 11-model benchmark (`data/initial_run/`)
5. `extended_cross_database_analysis.py` - final model, composition overlap, PBEsol offset calibration, Hubbard+U classification, feature tiers, distribution matching, bootstrap CIs; writes `final_model_predictions_*.csv` and `extended_cross_database_analysis.json` (source of Table 1)
6. `connectivity_validation.py` - space-group filter validated against atomic structures (needs the raw Alexandria `.json.bz2` files, path in `ALEXANDRIA_RAW`)
7. `composition_exclusion_and_pbesol_subgroup_offset.py` - excluded-composition retrain and category-specific PBEsol offset
8. `feature_tier_and_robustness_checks.py` - space-group encoding, lattice-independent tier, matched-training-size control, metallic/insulating split, ExtraTrees comparison
9. `canonical_space_group_and_structural_shift_analysis.py` - canonical-space-group retrain, composition-grouped CV, structural shift, paired cross-functional comparison, normalized degradation
10. `size_matched_control_and_tm_subgroup_checks.py` - random-removal control and PBEsol offset by transition-metal subgroup
11. `benchmark_models_internal_and_external.py` - single-pass internal and external 11-model benchmark (Supplementary Tables S3 and S4)
12. `summarize_main_results.py` - writes `data/metrics.json` (Table 1 values with bootstrap intervals)
13. `make_figures.py` - manuscript and Supplementary Material figures (`FIGURE_DIR` sets the output folder)

Steps 3 to 13 were re-run in this order from the deposited inputs and reproduce every number in the manuscript.

## Requirements

See `requirements.txt` (Python 3.11; numpy 2.4.6, pandas 2.3.3, scikit-learn 1.9.1, xgboost 3.2.0, pymatgen 2026.5.4, matminer 0.10.1).

## Data

`data/` contains the inputs and all results:

- `mp_training_records.csv` - the 1,684 Materials Project records (public `material_id`, formula, space group, formation energy, band gap, `theoretical` flag)
- `mp_perovskites.parquet`, `alexandria_{pbe,pbesol,scan}.parquet` - filtered ABO3 perovskite tables with features, as used by the scripts (Materials Project and Alexandria data, both CC BY 4.0)
- `final_model_predictions_*.csv` - per-compound true and predicted values of the final model for every transfer test (and the MP-internal split)
- `metrics.json`, `extended_cross_database_analysis.json`, `connectivity_and_robustness_checks.json`, `extended_robustness_and_sensitivity_checks.json`, `size_matched_control_and_tm_subgroup_checks.json`, `benchmark_models_internal_and_external.json` - all reported metrics
- `pbesol_offset_detail.csv`, `connectivity_validation_detail.csv` - per-compound details
- `initial_run/` - output of the first-pass scripts (steps 3 and 4)

The raw Alexandria convex-hull archives (about 240 MB) are not included; they are public at https://alexandria.icams.rub.de and are needed only for step 6.
