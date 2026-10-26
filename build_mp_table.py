"""
Phase 1: Materials Project (MP) perovskite-oxide master table.

Queries all ABO3-stoichiometry ternary oxides in recognized perovskite space
groups, pulls structure + thermodynamic + electronic data, and builds a
feature table: magpie composition statistics (matminer) + simple structural
descriptors (lattice params, space group, density) + a tolerance-factor-style
ionic-radius descriptor (Goldschmidt tolerance factor, using Shannon ionic
radii via pymatgen where available, NaN otherwise -- documented, not
silently dropped).

Targets: formation_energy_per_atom, band_gap.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from mp_api.client import MPRester
from pymatgen.core import Composition
from pymatgen.core.periodic_table import Species
from matminer.featurizers.composition import ElementProperty

API_KEY = os.environ["MP_API_KEY"]
PEROVSKITE_SG = {221, 225, 62, 167, 140, 129, 71, 194, 12, 15, 55, 74, 148}

PROJ = Path("/glade/work/ajalali/13 - Perovskite-CrossDB-ML-project")
DATA = PROJ / "data"
DATA.mkdir(exist_ok=True)

# Shannon ionic radii (12-coordinate A site, 6-coordinate B site, most common
# oxidation states for perovskite-forming cations), angstrom. Used only for
# the tolerance-factor descriptor; NaN when unavailable rather than guessed.
O_RADIUS_6COORD = 1.40  # Shannon radius of O2- (CN6)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def query_mp():
    with MPRester(API_KEY) as mpr:
        docs = mpr.materials.summary.search(
            elements=["O"],
            num_elements=3,
            fields=[
                "material_id", "formula_pretty", "structure", "symmetry",
                "energy_above_hull", "formation_energy_per_atom", "band_gap",
                "is_gap_direct", "nsites", "volume", "density", "theoretical",
            ],
            chunk_size=1000,
        )
    return docs


def is_abo3(formula_pretty):
    comp = Composition(formula_pretty)
    red = comp.reduced_composition.get_el_amt_dict()
    o_r = red.get("O")
    non_o = sorted(v for k, v in red.items() if k != "O")
    return o_r == 3 and non_o == [1, 1]


def tolerance_factor(comp, a_site, b_site):
    """Goldschmidt tolerance factor t = (r_A + r_O) / (sqrt(2) * (r_B + r_O)).
    Uses pymatgen's average ionic radius estimate per element (guesses the
    most common oxidation state); returns NaN if radii can't be estimated."""
    try:
        from pymatgen.core.periodic_table import Element
        r_a = Element(a_site).average_ionic_radius
        r_b = Element(b_site).average_ionic_radius
        if r_a is None or r_b is None:
            return np.nan
        r_a, r_b = float(r_a), float(r_b)
        return (r_a + O_RADIUS_6COORD) / (np.sqrt(2) * (r_b + O_RADIUS_6COORD))
    except Exception:
        return np.nan


def assign_ab_sites(formula_pretty):
    """Heuristic: in ABO3, the site with the larger average ionic radius is
    A (12-coordinate), the smaller is B (6-coordinate)."""
    comp = Composition(formula_pretty)
    non_o = [str(el) for el in comp.elements if str(el) != "O"]
    if len(non_o) != 2:
        return None, None
    from pymatgen.core.periodic_table import Element
    radii = {}
    for el in non_o:
        try:
            r = Element(el).average_ionic_radius
            radii[el] = float(r) if r is not None else -1
        except Exception:
            radii[el] = -1
    ordered = sorted(non_o, key=lambda e: -radii[e])
    return ordered[0], ordered[1]  # A, B


def build_feature_row(formula_pretty, struct_dict, sg_number, e_hull, e_form,
                       band_gap, is_gap_direct, nsites, volume, density,
                       material_id, theoretical):
    a_site, b_site = assign_ab_sites(formula_pretty)
    tf = tolerance_factor(Composition(formula_pretty), a_site, b_site) if a_site else np.nan
    row = {
        "id": material_id,
        "formula": formula_pretty,
        "a_site": a_site,
        "b_site": b_site,
        "space_group_number": sg_number,
        "energy_above_hull": e_hull,
        "formation_energy_per_atom": e_form,
        "band_gap": band_gap,
        "is_gap_direct": is_gap_direct,
        "nsites": nsites,
        "volume": volume,
        "density": density,
        "tolerance_factor": tf,
        "theoretical": theoretical,
    }
    if struct_dict is not None:
        lat = struct_dict["lattice"]
        row.update({
            "a": lat["a"], "b": lat["b"], "c": lat["c"],
            "alpha": lat["alpha"], "beta": lat["beta"], "gamma": lat["gamma"],
        })
    return row


def add_magpie_features(df):
    ep = ElementProperty.from_preset("magpie")
    from pymatgen.core import Composition as C
    comps = df["formula"].apply(C)
    feat_df = ep.featurize_dataframe(pd.DataFrame({"composition": comps}), "composition",
                                      ignore_errors=True)
    feat_df = feat_df.drop(columns=["composition"])
    feat_df.columns = [f"MagpieData_{c}" if not c.startswith("MagpieData") else c
                        for c in feat_df.columns]
    return pd.concat([df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)


def main():
    t0 = time.time()
    log("Querying Materials Project...")
    docs = query_mp()
    log(f"Total ternary oxides returned: {len(docs)}")

    rows = []
    for d in docs:
        try:
            if not is_abo3(d.formula_pretty):
                continue
        except Exception:
            continue
        sg = d.symmetry.number
        if sg not in PEROVSKITE_SG:
            continue
        struct_dict = d.structure.as_dict() if d.structure is not None else None
        row = build_feature_row(
            d.formula_pretty, struct_dict, sg, d.energy_above_hull,
            d.formation_energy_per_atom, d.band_gap, d.is_gap_direct,
            d.nsites, d.volume, d.density, str(d.material_id), d.theoretical,
        )
        rows.append(row)
    log(f"ABO3 perovskite-space-group entries: {len(rows)}")

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["formation_energy_per_atom", "band_gap"])
    log(f"After dropping missing targets: {len(df)}")

    log("Computing magpie composition features...")
    df = add_magpie_features(df)
    log(f"Final table: {df.shape}")

    df.to_parquet(DATA / "mp_perovskites.parquet")
    log(f"Saved to {DATA / 'mp_perovskites.parquet'}")
    log(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
