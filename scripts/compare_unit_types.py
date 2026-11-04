"""Step 7 of the OI151 plan: compare unit response types in the data, the fitted network and task-only networks.
All networks get the mouse's real trial inputs; all signals are sampled at the 15 Hz frames.

Needs runs/fit_oi151 (python -m rnn_workbench fit configs/fit_oi151.yaml)
and runs/vagal_oi151 (train with configs/train_vagal_oi151.yaml).
Writes runs/compare_unit_types.json."""
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import rnn_workbench.fit as F
from rnn_workbench.data import load_session
from rnn_workbench.train import load_model
from rnn_workbench.vagal_task import VagalOptoTask

R = ROOT / "runs"
cfg = F.load_fit_config(json.load(open(R / "fit_oi151/train_config.json"))["fit"])
cfg.session_dir = str(ROOT / cfg.session_dir)
sess = load_session(Path(cfg.session_dir), cfg.events_file, cfg.iscell_file, cfg.signal, cfg.fs, cfg.smooth_frames)
task = VagalOptoTask(cfg.task)
torch.manual_seed(0)
data = F.build_trials(sess, task, np.random.default_rng(0), False)
x, steps = data["x"], data["frame_steps"]
fs = cfg.fs
nogo = np.array([e["freq_hz"] > 0 for e in data["events"]])
licked = data["licked"].numpy().astype(bool)

def classify(A, thr=0.2):
    """A: (trials, frames, units). Response types from the trial averages only, so that
    networks with less trial-to-trial noise than the data are not counted as more
    responsive. Each effect is divided by the range of the unit's go and no-go averages."""
    w = lambda a, b: slice(int(a * fs), int(b * fs))
    G, N = A[~nogo, :int(9 * fs)].mean(0), A[nogo, :int(9 * fs)].mean(0)
    rng_ = np.maximum(np.r_[G, N].max(0) - np.r_[G, N].min(0), 1e-9)
    live = np.r_[G, N].max(0) > 1e-6
    out = {}
    d = (N[w(1, 5)].mean(0) - N[w(0, 1)].mean(0)) / rng_
    out["opto_up"], out["opto_down"] = ((d > thr) & live).mean(), ((d < -thr) & live).mean()
    for nm, (a, b) in {"odor": (5, 7), "reward": (7, 9)}.items():
        d = ((G[w(a, b)].mean(0) - G[w(0, 1)].mean(0)) - (N[w(a, b)].mean(0) - N[w(0, 1)].mean(0))) / rng_
        out[f"{nm}_go_higher"], out[f"{nm}_nogo_higher"] = ((d > thr) & live).mean(), ((d < -thr) & live).mean()
    tr = N[w(1, 5)]
    t = np.arange(tr.shape[0])
    r = np.array([np.corrcoef(t, tr[:, n])[0, 1] if tr[:, n].std() > 1e-9 else 0 for n in range(tr.shape[1])])
    amp = (tr[-15:].mean(0) - tr[:15].mean(0)) / rng_
    out["ramp_up"], out["ramp_down"] = ((r > 0.8) & (amp > thr)).mean(), ((r < -0.8) & (amp < -thr)).mean()
    out["silent"] = (~live).mean()
    P = np.r_[G, N][:, live]
    P = (P - P.mean(0)) / rng_[live]
    ev = np.clip(np.linalg.eigvalsh(np.cov(P.T)), 0, None)
    out["dim_pr"] = float(ev.sum() ** 2 / (ev ** 2).sum())
    out["n_units"] = A.shape[2]
    return {k: round(float(v), 3) for k, v in out.items()}

def sign_match(A, B):
    """Per cell: does the model's opto and odor effect have the same sign and size as the data's?"""
    w = lambda a, b: slice(int(a * fs), int(b * fs))
    def eff(X):
        G, N = X[~nogo].mean(0), X[nogo].mean(0)
        return {"opto": N[w(1, 5)].mean(0) - N[w(0, 1)].mean(0),
                "odor": (G[w(5, 7)].mean(0) - G[w(0, 1)].mean(0)) - (N[w(5, 7)].mean(0) - N[w(0, 1)].mean(0))}
    ea, eb = eff(A), eff(B)
    return {k: {"same_sign": round(float((np.sign(ea[k]) == np.sign(eb[k])).mean()), 3),
                "corr": round(float(np.corrcoef(ea[k], eb[k])[0, 1]), 3)} for k in ea}

def run(model, xin):
    with torch.no_grad():
        out, rates = model(xin, noise=True)
    return out, rates

res = {}
act = data["act"].numpy()
res["data (576 cells)"] = classify(act)

ck = torch.load(R / "fit_oi151/seed_0/model.pt", weights_only=False)
fm = load_model(R / "fit_oi151/seed_0")
ro = F.Readout(act.shape[2], cfg.calcium_tau_ms or 1000, cfg.task.dt, False)
ro.load_state_dict(ck["readout"], strict=False)
units = torch.tensor(ck["unit_of_cell"])
out, rates = run(fm, x)
with torch.no_grad():
    pred = ro(rates[:, :, units], steps).transpose(0, 1).numpy()
rr = rates[steps].transpose(0, 1).numpy()
ne = fm.n_exc
hidden_e = np.setdiff1d(np.arange(ne), ck["unit_of_cell"])
res["fitted: recorded units (readout)"] = classify(pred)
res["fitted vs data, per cell"] = sign_match(act, pred)
res["fitted: recorded units (rates)"] = classify(rr[:, :, ck["unit_of_cell"]])
res["fitted: hidden E"] = classify(rr[:, :, hidden_e])
res["fitted: I"] = classify(rr[:, :, ne:])
s, e = task.periods["response"]
res["fitted: lick acc on all trials"] = float(((torch.softmax(out[s:e], -1)[..., 1].mean(0) > 0.5).numpy() == licked).mean())

agg = {"E": [], "I": []}
accs = []
for sd in sorted((R / "vagal_oi151").glob("seed_*")):
    m = load_model(sd)
    out, rates = run(m, x)
    rr = rates[steps].transpose(0, 1).numpy()
    agg["E"].append(classify(rr[:, :, :m.n_exc])); agg["I"].append(classify(rr[:, :, m.n_exc:]))
    accs.append(float(((torch.softmax(out[s:e], -1)[..., 1].mean(0) > 0.5).numpy() == licked).mean()))
for k, L in agg.items():
    res[f"task-only ({len(L)} nets): {k} mean"] = {kk: round(float(np.mean([d[kk] for d in L])), 3) for kk in L[0]}
res["task-only: lick acc vs mouse"] = accs
print(json.dumps(res, indent=1))
R / "compare_unit_types.json".write_text(json.dumps(res, indent=1))
