"""Q2 prediction for OI151: test the fitted network (and the task-only networks) at
0.5-15 Hz, although the mouse and the fit only saw 15 Hz, and count units with sigmoid
or Gaussian frequency tuning, as in the paper.

Needs runs/fit_oi151 and runs/vagal_oi151. Writes runs/probe_frequencies.json."""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rnn_workbench.analysis import tuning_classes_checked, unit_responses
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
        kinds = tuning_classes_checked(resp[:, units], freqs, n_shuffle=N_SHUFFLE)
        c = Counter(k["kind"] for k in kinds)
        res[name] = {"n": len(units), **{k: c.get(k, 0) for k in ("sigmoid", "gaussian", "rising", "suppressed", "weak", "none")},
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
ck2 = torch.load(R / "fit_oi151_2area/seed_0/model.pt", weights_only=False)
def two_area_groups(m):
    area = np.array(m.cfg.area_of_unit); exc = m.is_exc.numpy(); rec2 = np.array(ck2["unit_of_cell"])
    return {"recorded E": rec2, "hidden insula E": np.setdiff1d(np.flatnonzero(exc & (area == 0)), rec2),
            "insula I": np.flatnonzero(~exc & (area == 0)), "decision E": np.flatnonzero(exc & (area == 1))}
result["fitted_2area"] = probe(R / "fit_oi151_2area", R / "fit_oi151_2area/seed_0", two_area_groups)
print("fitted_2area", result["fitted_2area"], flush=True)
for run in ("vagal_oi151", "vagal_oi151_n908"):
    result[run] = {}
    for sd in sorted((R / run).glob("seed_*")):
        result[run][sd.name] = probe(R / run, sd, task_groups)
        print(run, sd.name, result[run][sd.name], flush=True)
(R / "probe_frequencies.json").write_text(json.dumps(result, indent=1))
