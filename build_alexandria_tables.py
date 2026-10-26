"""
Phase 1: Alexandria perovskite-oxide tables, one per functional (PBE/PBEsol/
SCAN). Streams the large convex_hull JSON files with ijson (Decimal->float
fix required: composition dict values come back as Decimal objects and
silently fail pymatgen.Composition construction unless cast to float first).

Same feature set as the MP table: magpie composition statistics + simple
structural descriptors + Goldschmidt tolerance factor, so the two tables
are directly comparable feature-for-feature.
"""
import bz2
import time
from pathlib import Path

import ijson
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from pymatgen.core.periodic_table import Element
from matminer.featurizers.composition import ElementProperty

PROJ = Path("/glade/work/ajalali/13 - Perovskite-CrossDB-ML-project")
RAW = PROJ / "data" / "alexandria_raw"
DATA = PROJ / "data"

PEROVSKITE_SG = {221, 225, 62, 167, 140, 129, 71, 194, 12, 15, 55, 74, 148}
O_RADIUS_6COORD = 1.40

FILES = {
    "pbe": RAW / "alex_pbe.json.bz2",
    "pbesol": RAW / "alex_pbesol.json.bz2",
    "scan": RAW / "alex_scan.json.bz2",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tolerance_factor(a_site, b_site):
    try:
        r_a = Element(a_site).average_ionic_radius
        r_b = Element(b_site).average_ionic_radius
        if r_a is None or r_b is None:
            return np.nan
        return (float(r_a) + O_RADIUS_6COORD) / (np.sqrt(2) * (float(r_b) + O_RADIUS_6COORD))
    except Exception:
        return np.nan


def assign_ab_sites(non_o_elements):
    radii = {}
    for el in non_o_elements:
        try:
            r = Element(el).average_ionic_radius
            radii[el] = float(r) if r is not None else -1
        except Exception:
            radii[el] = -1
    ordered = sorted(non_o_elements, key=lambda e: -radii[e])
    return ordered[0], ordered[1]


def process_file(path, functional):
    rows = []
    total = 0
    with bz2.open(path, "rb") if path.suffix == ".bz2" else open(path, "rb") as f:
        for item in ijson.items(f, "entries.item"):
            total += 1
            comp_dict = item.get("composition", {})
            els = set(comp_dict.keys())
            if "O" not in els or len(els) != 3:
                continue
            try:
                comp = Composition({k: float(v) for k, v in comp_dict.items()})
                red = comp.reduced_composition.get_el_amt_dict()
            except Exception:
                continue
            o_r = red.get("O")
            non_o = sorted([k for k in red if k != "O"])
            non_o_vals = sorted(v for k, v in red.items() if k != "O")
            if o_r != 3 or non_o_vals != [1, 1]:
                continue
            data = item.get("data", {})
            sg = data.get("spg")
            if sg not in PEROVSKITE_SG:
                continue

            a_site, b_site = assign_ab_sites(non_o)
            tf = tolerance_factor(a_site, b_site)

            struct = item.get("structure", {})
            lat = struct.get("lattice", {})

            e_form = data.get("e_form")
            band_gap = data.get("band_gap_ind")
            if band_gap is None:
                band_gap = data.get("band_gap_dir")

            rows.append({
                "id": data.get("mat_id"),
                "formula": data.get("formula", comp.reduced_formula),
                "a_site": a_site,
                "b_site": b_site,
                "space_group_number": sg,
                "energy_above_hull": data.get("e_above_hull"),
                "formation_energy_per_atom": e_form,
                "band_gap": band_gap,
                "is_gap_direct": None,
                "nsites": data.get("nsites"),
                "volume": lat.get("volume"),
                "density": None,
                "tolerance_factor": tf,
                "theoretical": True,
                "a": lat.get("a"), "b": lat.get("b"), "c": lat.get("c"),
                "alpha": lat.get("alpha"), "beta": lat.get("beta"), "gamma": lat.get("gamma"),
                "functional": functional,
            })
    log(f"[{functional}] processed {total} total entries, {len(rows)} ABO3 perovskite-oxide rows")
    return rows


def add_magpie_features(df):
    ep = ElementProperty.from_preset("magpie")
    comps = df["formula"].apply(Composition)
    feat_df = ep.featurize_dataframe(pd.DataFrame({"composition": comps}), "composition",
                                      ignore_errors=True)
    feat_df = feat_df.drop(columns=["composition"])
    feat_df.columns = [f"MagpieData_{c}" if not c.startswith("MagpieData") else c
                        for c in feat_df.columns]
    return pd.concat([df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)


def main():
    t0 = time.time()
    for functional, path in FILES.items():
        log(f"Processing {functional} from {path}...")
        rows = process_file(path, functional)
        df = pd.DataFrame(rows)
        df = df.dropna(subset=["formation_energy_per_atom", "band_gap"])
        log(f"[{functional}] after dropping missing targets: {len(df)}")
        df = add_magpie_features(df)
        out = DATA / f"alexandria_{functional}.parquet"
        df.to_parquet(out)
        log(f"[{functional}] saved {df.shape} to {out}")
    log(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
