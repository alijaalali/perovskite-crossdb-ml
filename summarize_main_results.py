# Writes data/metrics.json: the headline table (MP-internal reference and the three
# Alexandria transfer tests, with bootstrap 95% intervals) from the saved analysis output.
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("PEROVSKITE_HOME", Path(__file__).resolve().parent))
DATA = ROOT / "data"

analysis = json.load(open(DATA / "extended_cross_database_analysis.json"))
cis = analysis["bootstrap_cis"]
out = {}
for target, d in cis.items():
    out[target] = {"mp_internal": d["mp_internal"], "alexandria_transfer": d["transfer"]}
json.dump(out, open(DATA / "metrics.json", "w"), indent=2)
print("wrote", DATA / "metrics.json")
