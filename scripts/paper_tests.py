"""Step 1 of the easy-task plan: run the paper's single-cell response test (E2) on the real
OI151 cells and on the fitted one-area model's copies of them.

Test (paper Methods, "Single-Neuron Response Classification"): on light trials, the mean
activity 1 s after light onset until light offset vs 0-1 s after trial onset, auROC with
500 within-trial shuffles, 10th/90th percentile thresholds; then the same window on hit
trials as a control. Needs runs/fit_oi151. Writes runs/paper_tests.json."""
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import rnn_workbench.fit as F
from rnn_workbench.analysis import choice_probability_paper, decode_accuracy, event_responsive
from rnn_workbench.data import load_session
from rnn_workbench.train import load_model
from rnn_workbench.vagal_task import VagalOptoTask

run_dir = ROOT / "runs" / (sys.argv[1] if len(sys.argv) > 1 else "fit_oi151")
cfg = F.load_fit_config(json.load(open(run_dir / "train_config.json"))["fit"])
cfg.session_dir = str(ROOT / cfg.session_dir)
sess = load_session(Path(cfg.session_dir), cfg.events_file, cfg.iscell_file, cfg.signal, cfg.fs, cfg.smooth_frames)
sess = F.select_cells(sess, cfg.n_fit_cells)  # the subset the run was fitted to (no-op for all cells)
task = VagalOptoTask(cfg.task)
torch.manual_seed(0)
data = F.build_trials(sess, task, np.random.default_rng(0), False)
x, steps, fs = data["x"], data["frame_steps"], cfg.fs
light = np.array([e["freq_hz"] > 0 for e in data["events"]])
licked = data["licked"].numpy().astype(bool)
hit = ~light & licked
dt = task.cfg.dt
on, off = (s * dt / 1000 for s in task.periods["stimulus"])
win = lambda a, b: slice(int(round(a * fs)), int(round(b * fs)))
W_BASE, W_EVENT = win(0, 1), win(on + 1, off)


def test(A):
    """A: (trials, frames, units)."""
    r = event_responsive(A[light][:, W_EVENT].mean(1), A[light][:, W_BASE].mean(1),
                         control=A[hit][:, W_EVENT].mean(1))
    return r


W_LIGHT = win(on, off)
cr, fa = light & ~licked, light & licked


def e3(A):
    """E3: choice probability (both nulls) and two decoders on the 4 s light window."""
    L = A[:, W_LIGHT].mean(1)
    out = {}
    for ns in ("half", "matched"):
        r = choice_probability_paper(L[cr], L[fa], null_size=ns)
        out[f"cp_significant_{ns}_null"] = round(r["fraction_significant"], 3)
    m = cr | fa
    out["decoder_cr_vs_fa"] = round(np.mean([decode_accuracy(L[m], cr[m].astype(int), balance_groups=np.zeros(m.sum()), seed=s)
                                             for s in range(20)]), 3)
    out["decoder_light_vs_none"] = round(decode_accuracy(L, light.astype(int)), 3)
    return out


def summary(r):
    return {"excited": round(r["excited"], 3), "suppressed": round(r["suppressed"], 3)}


# Trial noise borrowed from the real cells: each cell's residual (trial minus its
# condition's mean) on a random real trial.
cond = light * 2 + licked
act = data["act"].numpy()
resid = act.copy()
for c in np.unique(cond):
    resid[cond == c] -= act[cond == c].mean(0)


def borrowed_noise(rng):
    """Each trial gets the residual of a random real trial (any condition, with
    replacement), so the noise added to a small group such as the 7 FA trials does not
    sum to zero, which would bias the decoders below chance."""
    return resid[rng.integers(len(resid), size=len(resid))]


res = {"windows_s": {"baseline": [0, 1], "event": [on + 1, off]},
       "n_light_trials": int(light.sum()), "n_hit_trials": int(hit.sum())}
r_data = test(act)
res["data (576 cells)"] = summary(r_data)
res["n_cr_fa"] = [int(cr.sum()), int(fa.sum())]
res["E3 data"] = e3(act)

for sd in sorted(run_dir.glob("seed_*")):
    ck = torch.load(sd / "model.pt", weights_only=False)
    m = load_model(sd)
    ro = F.Readout(act.shape[2], cfg.calcium_tau_ms or 1000, dt, False)
    ro.load_state_dict(ck["readout"], strict=False)
    units = torch.tensor(ck["unit_of_cell"])
    for noise in (True, False):
        torch.manual_seed(1)
        with torch.no_grad():
            _, rates = m(x, noise=noise)
            pred = ro(rates[:, :, units], steps).transpose(0, 1).numpy()
        r = test(pred)
        agree = (r["label"] == r_data["label"])
        key = f"model {sd.name}, {'with' if noise else 'without'} training noise"
        res[key] = summary(r) | {
            "same_label_as_its_cell": round(float(agree.mean()), 3),
            "of_suppressed_cells_also_suppressed_in_model":
                round(float((r["label"][r_data["label"] == -1] == -1).mean()), 3)}
    rng = np.random.default_rng(0)
    noisy = [pred + borrowed_noise(rng) for _ in range(5)]
    draws = [test(n) for n in noisy]
    res[f"E3 model {sd.name}, plus the cells' own trial noise (draw 1)"] = e3(noisy[0])
    res[f"model {sd.name}, plus the cells' own trial noise (5 draws)"] = {
        "excited": round(float(np.mean([d["excited"] for d in draws])), 3),
        "suppressed": round(float(np.mean([d["suppressed"] for d in draws])), 3),
        "suppressed_range": [round(min(d["suppressed"] for d in draws), 3), round(max(d["suppressed"] for d in draws), 3)],
        "same_label_as_its_cell": round(float(np.mean([(d["label"] == r_data["label"]).mean() for d in draws])), 3)}
    # trial-to-trial spread, model vs data: std across light trials of the event-window mean
    sd_data = act[light][:, W_EVENT].mean(1).std(0)
    sd_model = pred[light][:, W_EVENT].mean(1).std(0)
    res[f"model {sd.name}, trial spread / data spread (median over cells, no noise)"] = round(
        float(np.median(sd_model / np.maximum(sd_data, 1e-9))), 3)

print(json.dumps(res, indent=1))
(ROOT / "runs" / f"paper_tests_{run_dir.name}.json").write_text(json.dumps(res, indent=1))
