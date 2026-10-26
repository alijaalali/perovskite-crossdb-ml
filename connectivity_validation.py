# Validates the perovskite space-group filter against real atomic structures: MP's
# table only stores lattice parameters, not full atomic positions, but Alexandria's
# raw archives include full structures for every entry, and Alexandria shares most of
# its compositions with MP under the identical ABO3+perovskite-space-group filter, so
# its structures are a legitimate empirical check of whether that filter actually
# selects corner-sharing perovskite connectivity, applied directly to real atomic
# positions. This validates a canonical/ambiguous space-group split for the MP table.
import json
import time
import bz2
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Structure, Composition, Element, Lattice
from pymatgen.analysis.local_env import CrystalNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

ROOT = Path("/glade/work/ajalali/13 - Perovskite-CrossDB-ML-project")
DATA = ROOT / "data"
OUT = {}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("=== Perovskite connectivity validation ===")

PEROVSKITE_SG = {221, 225, 62, 167, 140, 129, 71, 194, 12, 15, 55, 74, 148}
# Canonical corner-sharing-perovskite space groups (standard Glazer tilt-system /
# simple & double perovskite settings): cubic Pm-3m (221), double-perovskite Fm-3m
# (225), orthorhombic GdFeO3-type Pnma (62), rhombohedral LaAlO3-type R-3c (167),
# tetragonal I4/mcm (140). The remaining space groups in the original filter are
# crystallographically ambiguous for ABO3 -- some host genuinely distorted
# corner-sharing perovskites, but several (above all P6_3/mmc, 194) are also the
# standard settings for face-sharing hexagonal perovskite POLYTYPES (2H/4H/6H/9R-type,
# e.g. BaNiO3, hexagonal BaTiO3), which are not corner-sharing 3D perovskite networks.
CANONICAL_SG = {221, 225, 62, 167, 140}
AMBIGUOUS_SG = PEROVSKITE_SG - CANONICAL_SG

mp = pd.read_parquet(DATA / "mp_perovskites.parquet")
mp_sg_counts = mp["space_group_number"].value_counts().sort_index().to_dict()
n_canonical = int(mp["space_group_number"].isin(CANONICAL_SG).sum())
n_ambiguous = int(mp["space_group_number"].isin(AMBIGUOUS_SG).sum())
n_sg194 = int((mp["space_group_number"] == 194).sum())
log(f"MP: {n_canonical} canonical-SG, {n_ambiguous} ambiguous-SG (of which {n_sg194} are SG194)")

def assign_ab_sites(formula_pretty):
    comp = Composition(formula_pretty)
    non_o = [str(el) for el in comp.elements if str(el) != "O"]
    if len(non_o) != 2:
        return None, None
    radii = {}
    for el in non_o:
        try:
            r = Element(el).average_ionic_radius
            radii[el] = float(r) if r is not None else -1
        except Exception:
            radii[el] = -1
    ordered = sorted(non_o, key=lambda e: -radii[e])
    return ordered[0], ordered[1]


def classify_connectivity(structure, b_site_symbol):
    """For each pair of B-site-cation nearest-structural-neighbors, count
    shared (bridging) oxygen atoms. 1 shared O = corner-sharing, 2 = edge-
    sharing, 3 = face-sharing. Returns the majority linkage type and the
    fraction of B-B links that are corner-sharing.

    Two correctness requirements: (1) many Alexandria/MP relaxations are
    stored in a small (e.g. 5-atom) primitive cell with only one B site, so
    structures are expanded to a 2x2x2 supercell first to get real B-B
    neighbor pairs; (2) a bridging oxygen must be identified by (site_index,
    periodic image), not by site_index alone -- using site_index alone
    conflates different periodic images of the same crystallographic oxygen
    and produces spurious edge-sharing counts even for textbook corner-sharing
    cubic perovskites, which is why a sanity check against an ideal cubic
    SrTiO3-type structure runs before trusting this on real data."""
    cnn = CrystalNN()
    sc = structure * (2, 2, 2)
    b_indices = [i for i, s in enumerate(sc) if s.specie.symbol == b_site_symbol]
    if len(b_indices) < 2:
        return None, None
    b_o_neighbors = {}
    for bi in b_indices:
        try:
            nn_info = cnn.get_nn_info(sc, bi)
        except Exception:
            continue
        o_sites = {(n["site_index"], tuple(np.round(n["image"], 3)))
                   for n in nn_info if n["site"].specie.symbol == "O"}
        b_o_neighbors[bi] = o_sites
    if len(b_o_neighbors) < 2:
        return None, None
    coords = {bi: sc[bi].coords for bi in b_o_neighbors}
    b_list = list(b_o_neighbors.keys())
    dists = [np.linalg.norm(coords[b_list[ii]] - coords[b_list[jj]])
             for ii in range(len(b_list)) for jj in range(ii + 1, len(b_list))]
    if not dists:
        return None, None
    d_min = min(dists)
    link_counts = []
    for ii in range(len(b_list)):
        for jj in range(ii + 1, len(b_list)):
            d = np.linalg.norm(coords[b_list[ii]] - coords[b_list[jj]])
            if d > d_min * 1.15:
                continue
            shared = b_o_neighbors[b_list[ii]] & b_o_neighbors[b_list[jj]]
            if len(shared) > 0:
                link_counts.append(len(shared))
    if not link_counts:
        return None, None
    link_counts = np.array(link_counts)
    frac_corner = float((link_counts == 1).mean())
    frac_edge = float((link_counts == 2).mean())
    frac_face = float((link_counts >= 3).mean())
    majority = max([("corner", frac_corner), ("edge", frac_edge), ("face", frac_face)],
                    key=lambda x: x[1])[0]
    return majority, frac_corner


def _sanity_check_classify_connectivity():
    """Ideal cubic SrTiO3-type structure: B-site octahedra MUST be
    classified as corner-sharing. Run before trusting this on real data."""
    lat = Lattice.cubic(3.9)
    s = Structure(lat, ["Sr", "Ti", "O", "O", "O"],
                  [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    majority, frac_corner = classify_connectivity(s, "Ti")
    assert majority == "corner" and frac_corner == 1.0, (
        f"classify_connectivity sanity check FAILED on ideal cubic perovskite: "
        f"got majority={majority}, frac_corner={frac_corner} (expected corner, 1.0)")
    log("classify_connectivity sanity check passed: ideal cubic perovskite "
        "correctly classified as 100% corner-sharing.")


_sanity_check_classify_connectivity()


import ijson

results_by_entry = []
for functional, fname in [("PBE", "alex_pbe.json.bz2"), ("PBEsol", "alex_pbesol.json.bz2"),
                           ("SCAN", "alex_scan.json.bz2")]:
    log(f"Streaming {functional} raw structures (ijson, low-memory)...")
    n_checked, n_canon_sg, n_ambig_sg, n_parse_fail = 0, 0, 0, 0
    f_handle = bz2.open(DATA / "alexandria_raw" / fname, "rb")
    # use_float=True: without it ijson parses JSON numbers as Decimal, which
    # pymatgen's Structure.from_dict cannot handle.
    entries_iter = ijson.items(f_handle, "entries.item", use_float=True)
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    for ent in entries_iter:
        try:
            struct = Structure.from_dict(ent["structure"])
            comp = struct.composition
            red = comp.reduced_composition.get_el_amt_dict()
            if red.get("O") != 3:
                continue
            non_o = sorted(v for k, v in red.items() if k != "O")
            if non_o != [1, 1]:
                continue
        except Exception:
            n_parse_fail += 1
            continue
        try:
            sga = SpacegroupAnalyzer(struct, symprec=0.1)
            sg_num = sga.get_space_group_number()
        except Exception:
            continue
        if sg_num not in PEROVSKITE_SG:
            continue
        formula_pretty = comp.reduced_formula
        a_site, b_site = assign_ab_sites(formula_pretty)
        if b_site is None:
            continue
        majority, frac_corner = classify_connectivity(struct, b_site)
        if majority is None:
            continue
        n_checked += 1
        cat = "canonical" if sg_num in CANONICAL_SG else "ambiguous"
        if cat == "canonical":
            n_canon_sg += 1
        else:
            n_ambig_sg += 1
        results_by_entry.append({
            "functional": functional, "formula": formula_pretty, "sg": sg_num,
            "category": cat, "majority_linkage": majority, "frac_corner": frac_corner,
        })
    f_handle.close()
    log(f"{functional}: checked {n_checked} perovskite-filtered entries "
        f"({n_canon_sg} canonical-SG, {n_ambig_sg} ambiguous-SG), "
        f"{n_parse_fail} structures failed to parse/filter")

conn_df = pd.DataFrame(results_by_entry)
if len(conn_df) == 0:
    raise RuntimeError("No entries survived the connectivity-validation filters -- "
                        "this almost certainly means the parsing step is broken, "
                        "not that zero real perovskites exist in these files.")
conn_df.to_csv(DATA / "connectivity_validation_detail.csv", index=False)

summary = {}
for cat in ["canonical", "ambiguous"]:
    sub = conn_df[conn_df["category"] == cat]
    if len(sub) == 0:
        continue
    corner_frac = float((sub["majority_linkage"] == "corner").mean())
    summary[cat] = {
        "n": int(len(sub)),
        "frac_majority_corner_sharing": corner_frac,
        "linkage_counts": sub["majority_linkage"].value_counts().to_dict(),
    }
sg194_sub = conn_df[conn_df["sg"] == 194]
if len(sg194_sub) > 0:
    summary["sg194_specifically"] = {
        "n": int(len(sg194_sub)),
        "linkage_counts": sg194_sub["majority_linkage"].value_counts().to_dict(),
    }
log(f"Connectivity validation summary: {json.dumps(summary, indent=2)}")

OUT["perovskite_connectivity_validation"] = {
    "mp_space_group_counts": {str(k): int(v) for k, v in mp_sg_counts.items()},
    "mp_n_canonical_sg": n_canonical,
    "mp_n_ambiguous_sg": n_ambiguous,
    "mp_n_sg194": n_sg194,
    "canonical_sg_list": sorted(CANONICAL_SG),
    "ambiguous_sg_list": sorted(AMBIGUOUS_SG),
    "alexandria_connectivity_check": summary,
    "n_alexandria_entries_checked": int(len(conn_df)),
}

print("\n--- Connectivity validation done, saving checkpoint ---")
with open(DATA / "connectivity_and_robustness_checks.json", "w") as f:
    json.dump(OUT, f, indent=2, default=str)
