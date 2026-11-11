"""Compare fits that constrain fewer OI151 cells (fit_oi151_c50/c150/c300) with the full fit.

For each run and seed: held-out R2 (from summary.json), lick accuracy, the detection
curve at 0-15 Hz, and sigmoid / Gaussian shares by unit group (constrained E, free E, I),
using the same quick rule as the tuning slides. Writes runs/fewer_cells.json.
Usage: python scripts/fewer_cells.py [run ...]"""
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from rnn_workbench.tasks import get_task, task_config_from_dict
from rnn_workbench.train import load_model
from rnn_workbench.analysis import unit_responses
from rnn_workbench.vagal_task import behavior_summary

F = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 15.0]

def classify(mean):
    """mean: (n_freq, n_units). Same quick rule as the tuning slides."""
    rise = mean[1:].max(0) - mean[0]; drop = mean[0] - mean[1:].min(0)
    ref = np.percentile(np.maximum(rise, 0), 95)
    y = mean[1:]; pk = np.argmax(y, 0); fall = y.max(0) - y[-1]
    strong = (rise > drop) & (rise >= 0.1 * ref)
    gauss = strong & (pk < len(F) - 2) & (fall >= 0.3 * rise)
    lab = np.where(drop > rise, "suppressed", np.where(~strong, "weak", np.where(gauss, "gaussian", "sigmoid")))
    return lab, pk

def one(run, seed):
    tc = json.load(open(ROOT / "runs" / run / "train_config.json"))["task"]
    t = get_task(task_config_from_dict({**tc, "reward_delay": 200}))
    b = t.eval_batch(40, seed=4321)
    m = load_model(str(ROOT / "runs" / run / f"seed_{seed}"))
    torch.manual_seed(0)
    out, rates = t.run(m, b)
    beh = behavior_summary(t.choices(out, b), b.conditions, "no_reward")
    freqs = np.array([c["freq"] for c in b.conditions])
    resp = unit_responses(rates, b.periods, "stimulus", "baseline")
    mean = np.array([resp[freqs == f].mean(0) for f in F])
    ck = torch.load(ROOT / "runs" / run / f"seed_{seed}" / "model.pt", weights_only=False)
    rec = np.array(ck["unit_of_cell"]); n_exc = m.cfg.n_exc if hasattr(m.cfg, "n_exc") else int(round(m.cfg.n_units * m.cfg.frac_exc))
    lab, pk = classify(mean)
    groups = {"constrained E": rec, "free E": np.setdiff1d(np.arange(n_exc), rec), "I": np.arange(n_exc, m.cfg.n_units)}
    res = {"p_report_stim": {str(k): v for k, v in beh["p_report_stim"].items()}}
    for g, ids in groups.items():
        l = lab[ids]
        res[g] = {"n": int(len(ids)), **{k: round(float((l == k).mean()), 3) for k in ["sigmoid", "gaussian", "suppressed", "weak"]}}
    allE = np.arange(n_exc); g = allE[lab[allE] == "gaussian"]
    res["gaussian_peak_hz_E"] = {str(F[1:][i]): int((pk[g] == i).sum()) for i in range(len(F) - 1)}
    return res

runs = sys.argv[1:] or ["fit_oi151_c50", "fit_oi151_c150", "fit_oi151_c300", "fit_oi151"]
outp = ROOT / "runs" / "fewer_cells.json"
allres = json.load(open(outp)) if outp.exists() else {}
for run in runs:
    summ = json.load(open(ROOT / "runs" / run / "summary.json"))
    allres[run] = {"summary": summ, "seeds": {}}
    for s in summ:
        allres[run]["seeds"][str(s["seed"])] = one(run, s["seed"])
        print(run, s["seed"], json.dumps(allres[run]["seeds"][str(s["seed"])]))
    outp.write_text(json.dumps(allres, indent=1))
