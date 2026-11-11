"""Collect the fewer-cells test results (scripts/size_tests.py, scripts/fewer_cells.py) into
one row per model copy. Writes runs/size_summary.json and prints a table.
Usage: python scripts/size_summary.py"""
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "runs"; X = ROOT / "experiments"
SIZES = {50: ["fit_oi151_c50"], 150: ["fit_oi151_c150"], 300: ["fit_oi151_c300", "fit_oi151_c300b"],
         450: ["fit_oi151_c450", "fit_oi151_c450b"], 576: ["fit_oi151"]}
FREQ = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 15.0]
L = lambda p: json.load(open(p)) if Path(p).exists() else None

def threshold(p):
    """Lowest probed frequency where the light is reported on at least half the trials."""
    return next((f for f in FREQ[1:] if p[str(f)] >= 0.5), None)

def e5(path, seed):
    r = L(path / "results.json")
    if not r: return None
    v = r["seeds"][str(seed)]
    cr = [p["perturbed"]["p_report_stim"]["15.0"] for p in v["perturbations"].values()]
    hit = [p["perturbed"]["p_lick_no_stim"] for p in v["perturbations"].values()]
    return {"cr_min": round(min(cr), 3), "hit_min": round(min(hit), 3)}

rows, data = [], {}
fc = L(R / "fewer_cells.json")
for n, runs in SIZES.items():
    for run in runs:
        pt = L(R / f"paper_tests_{run}.json")
        e4 = L(R / ("easy_e4.json" if run == "fit_oi151" else f"easy_e4_{run}.json"))
        e7 = L(R / ("easy_e7.json" if run == "fit_oi151" else f"easy_e7_{run}.json"))
        e8 = L(R / f"easy_e8_{run}.json")
        noisy = "fit_oi151_noise15" if run == "fit_oi151" else f"{run}_noise15"
        data[run] = {"E2": pt["data (576 cells)"], "E3": pt["E3 data"], "E4_FA": {k: e4["data"][k]["FA"] for k in ("light_cluster_in", "reward_cluster_in")}}
        for s in json.load(open(R / run / "summary.json")):
            k = s["seed"]; sd = f"seed_{k}"
            t = fc[run]["seeds"][str(k)]
            n2 = pt[f"model {sd}, plus the cells' own trial noise (5 draws)"]
            n3 = pt[f"E3 model {sd}, plus the cells' own trial noise (draw 1)"]
            m4 = e4[f"model {sd} + cells' trial noise"]
            row = {"cells": n, "run": run, "seed": k, "R2": round(s["r2_psth"], 3), "E1_lick_acc": round(s["lick_accuracy"], 3),
                   "E2_up": n2["excited"], "E2_down": n2["suppressed"],
                   "E3_cp": n3["cp_significant_matched_null"], "E3_dec_crfa": n3["decoder_cr_vs_fa"], "E3_dec_light": n3["decoder_light_vs_none"],
                   "E4_FA_light": m4["light_cluster_in"]["FA"], "E4_FA_reward": m4["reward_cluster_in"]["FA"],
                   "E5_clean": e5(X / ("easy_e5" if run == "fit_oi151" else f"easy_e5_{run}"), k),
                   "E5_noisy": e5(X / ("easy_e5_noise15" if run == "fit_oi151" else f"easy_e5_{noisy}"), k),
                   "sig_constrained": t["constrained E"]["sigmoid"], "gauss_constrained": t["constrained E"]["gaussian"],
                   "gauss_freeE": t["free E"]["gaussian"], "threshold_hz": threshold(t["p_report_stim"])}
            if e7 and f"seed {k}" in e7:
                b, a = e7[f"seed {k}"]["before (light = no water)"], e7[f"seed {k}"]["after (light = water)"]
                row |= {"E7_light_before": b["light_excited"], "E7_light_after": a["light_excited"], "E7_acc_after": a["accuracy"]}
            if e8:
                g = lambda c: e8[c][str(k)]["fa_rate"]
                row |= {"E8_FA_4s": g("4 s"), "E8_FA_1s": g("1 s"), "E8_FA_4p": g("267 ms (4 pulses)"), "E8_FA_1p": g("20 ms (1 pulse)"),
                        "E8_FA_gap2": g("4 s + 2 s gap"), "E8_FA_gap4": g("4 s + 4 s gap"), "E8_FA_gap8": g("4 s + 8 s gap")}
            rows.append(row)
(R / "size_summary.json").write_text(json.dumps({"rows": rows, "data": data}, indent=1))
cols = list(dict.fromkeys(c for r in rows for c in r if c not in ("run",)))
print("\t".join(cols))
for r in rows:
    print("\t".join(str(r.get(c, "")) if not isinstance(r.get(c), dict) else f"{r[c]['cr_min']}/{r[c]['hit_min']}" for c in cols))
print(json.dumps(data, indent=0)[:3000])
