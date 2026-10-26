# Cross-database ML for perovskite-oxide formation energy and band gap (Materials Project vs. Alexandria)

Scripts used to build features, train models, run robustness/sensitivity analyses, and generate figures/tables for the manuscript.

## Pipeline order

1. `build_mp_table.py` — Materials Project perovskite-oxide feature table (requires an `MP_API_KEY` environment variable)
2. `build_alexandria_tables.py` — Alexandria perovskite-oxide feature tables (PBE/PBEsol/SCAN)
3. `train_crossdb.py` — core cross-database transfer test (MP-internal baseline + zero-shot transfer to each Alexandria functional)
4. `benchmark_models.py` — 11-model external-only benchmark
5. `extended_cross_database_analysis.py` — composition overlap, PBEsol offset calibration, Hubbard+U classification, feature-tier comparison, distribution matching, extended internal+external 11-model benchmark, bootstrap CIs
6. `connectivity_validation.py` — validates the perovskite space-group filter against real atomic structures
7. `composition_exclusion_and_pbesol_subgroup_offset.py` — excluded-composition retrain + category-specific PBEsol offset
8. `feature_tier_and_robustness_checks.py` — space-group encoding, lattice-independent feature tier, matched-training-size control, metallic-vs-insulating split, ExtraTrees comparison
9. `canonical_space_group_and_structural_shift_analysis.py` — canonical-space-group-restricted retrain, composition-grouped CV, structural feature-shift quantification, cross-functional paired comparison, normalized transfer-degradation metric
10. `size_matched_control_and_tm_subgroup_checks.py` — size-matched random-removal control, PBEsol offset by transition-metal subgroup
11. `benchmark_models_internal_and_external.py` — single-pass internal+external 11-model benchmark regeneration
12. `make_figures.py` — manuscript figures

## Requirements

Python 3, pandas, numpy, scikit-learn, scipy, xgboost, matminer, pymatgen, mp-api, ijson, matplotlib.

## Data

Raw data (Materials Project API query results, Alexandria convex-hull archives) are not included here; scripts expect them in a local `data/` directory. See the manuscript's Data Availability statement. `build_mp_table.py` requires a Materials Project API key set as the `MP_API_KEY` environment variable.
