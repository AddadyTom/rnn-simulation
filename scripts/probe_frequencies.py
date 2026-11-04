"""Q2 prediction for OI151: test the fitted network (and the task-only networks) at
0.5-15 Hz, although the mouse and the fit only saw 15 Hz, and count units with sigmoid
or Gaussian frequency tuning, as in the paper.

Needs runs/fit_oi151 and runs/vagal_oi151. Writes runs/probe_frequencies.json."""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np, torch
from scipy.stats import f_oneway
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rnn_workbench.analysis import tuning_classes, unit_responses
from rnn_workbench.tasks import get_task, task_config_from_dict
from rnn_workbench.train import load_model

R = ROOT / "runs"
N_SHUFFLE = 20


def probe(run_dir: Path, seed_dir: Path, groups):
    tc = json.loads((run_dir / "train_config.json").read_text())["task"]
    task = get_task(task_config_from_dict({**tc, "reward_delay": 200}))
    batch = task.eval_batch(40, seed=4321)
    model = load_model(seed_dir)
    torch.manual_seed(0)
    out, rates = task.run(model, batch)
    beh = task.behavior(out, batch)
    freqs = np.array([c["freq"] for c in batch.conditions])
    resp = unit_responses(rates, batch.periods, "stimulus", "baseline")
    res = {"p_report_stim": beh.get("p_report_stim")}
    for name, units in groups(model).items():
        # Only units whose response changes with frequency (one-way ANOVA, p < 0.01) are
        # fitted; the others are counted as "none". This keeps 900-unit networks fast.
        r = resp[:, units]
        groups_f = [r[freqs == f] for f in np.unique(freqs)]
        p = np.array([f_oneway(*[g[:, j] for g in groups_f]).pvalue if r[:, j].std() > 1e-9 else 1.0
                      for j in range(r.shape[1])])
        tuned = np.flatnonzero(p < 0.01)
        kinds = tuning_classes(r[:, tuned], freqs, n_shuffle=N_SHUFFLE) if len(tuned) else []
        c = Counter(k["kind"] for k in kinds)
        c["none"] += len(units) - len(tuned)
        res[name] = {"n": len(units), **{k: c.get(k, 0) for k in ("sigmoid", "gaussian", "none")},
                     "sigmoid_threshold_hz_median": float(np.median([k["threshold_hz"] for k in kinds if k["kind"] == "sigmoid"]))
                     if c.get("sigmoid") else None}
    return res


ck = torch.load(R / "fit_oi151/seed_0/model.pt", weights_only=False)
rec = np.array(ck["unit_of_cell"])
def fitted_groups(m):
    ne = m.n_exc
    return {"recorded E": rec, "hidden E": np.setdiff1d(np.arange(ne), rec), "I": np.arange(ne, m.cfg.n_units)}
def task_groups(m):
    return {"E": np.arange(m.n_exc), "I": np.arange(m.n_exc, m.cfg.n_units)}

result = {"fitted": probe(R / "fit_oi151", R / "fit_oi151/seed_0", fitted_groups), "task_only": {}}
print("fitted", result["fitted"], flush=True)
for sd in sorted((R / "vagal_oi151").glob("seed_*")):
    result["task_only"][sd.name] = probe(R / "vagal_oi151", sd, task_groups)
    print(sd.name, result["task_only"][sd.name], flush=True)
(R / "probe_frequencies.json").write_text(json.dumps(result, indent=1))
