"""Easy-task plan, step 3, E7 (paper Fig 1o-q, Fig 4b-d): after the rule is reversed
(light now means water), the shares of light- and reward-responsive cells stay the same.

Compares runs/fit_oi151_noise15 on the original rule (VagalOpto->NoReward) with
runs/reversal_oi151 (the same networks trained on VagalOpto->Reward). Both are run on
fresh task trials (15 Hz or no light, closed loop), their 576 cell-copying units are read
through the fit's calcium readout, each cell's own trial noise from OI151 is added, and
the paper's response test (scripts/paper_tests.py) labels each unit:
  light:  1 s after light onset to light offset vs 0-1 s, control = same window, no light
  reward: 2 s after the reward vs 0-1 s on rewarded trials, control = same window on
          unrewarded trials of the same light condition
Pass: the reversed network learns the rule and each share changes by 5 points or less.
Writes runs/easy_e7.json."""
import json, sys
from dataclasses import asdict
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import rnn_workbench.fit as F
from rnn_workbench.analysis import event_responsive
from rnn_workbench.data import load_session
from rnn_workbench.train import load_model
from rnn_workbench.vagal_task import VagalOptoTask, VagalTaskConfig, reward_window

_args = [a for a in sys.argv[1:] if not a.startswith("-")]
BEFORE, AFTER = (ROOT / "runs" / _args[0], ROOT / "runs" / _args[1]) if len(_args) >= 2 else (ROOT / "runs/fit_oi151_noise15", ROOT / "runs/reversal_oi151")
OUT = _args[2] if len(_args) >= 3 else "easy_e7.json"
cfg = F.load_fit_config(json.load(open(BEFORE / "train_config.json"))["fit"])
cfg.session_dir = str(ROOT / cfg.session_dir)
sess = load_session(Path(cfg.session_dir), cfg.events_file, cfg.iscell_file, cfg.signal, cfg.fs, cfg.smooth_frames)
sess = F.select_cells(sess, cfg.n_fit_cells)  # the subset the run was fitted to (no-op for all cells)
real_task = VagalOptoTask(cfg.task)
data = F.build_trials(sess, real_task, np.random.default_rng(0), False)
act = data["act"].numpy()
light_r = np.array([e["freq_hz"] > 0 for e in data["events"]])
cond = light_r * 2 + data["licked"].numpy().astype(int)
resid = act.copy()
for c in np.unique(cond):
    resid[cond == c] -= act[cond == c].mean(0)
steps = data["frame_steps"]
fs, dt = cfg.fs, cfg.task.dt
win = lambda a, b: slice(int(round(a * fs)), int(round(b * fs)))


def run(model_dir, seed, contingency, n=80):
    return run_model(load_model(model_dir / f"seed_{seed}"), seed, contingency, n)


def run_model(m, seed, contingency, n=80):
    """m: a network (or a wrapper with the same call signature)."""
    tc = VagalTaskConfig(**{**asdict(cfg.task), "contingency": contingency, "reward_delay": 200})
    task = VagalOptoTask(tc)
    ck = torch.load(BEFORE / f"seed_{seed}" / "model.pt", weights_only=False)
    ro = F.Readout(act.shape[2], cfg.calcium_tau_ms or 1000, dt, False)
    ro.load_state_dict(ck["readout"], strict=False)
    torch.manual_seed(seed)
    b = task.eval_batch(n, seed=seed, freqs=[15.0])
    out, rates = task.run(m, b)
    licked = task.choices(out, b).astype(bool)
    light = np.array([c["stim"] for c in b.conditions])
    go = np.array([c["go"] for c in b.conditions])
    with torch.no_grad():
        pred = ro(rates[:, :, torch.tensor(ck["unit_of_cell"])], steps).transpose(0, 1).numpy()
    rng = np.random.default_rng(seed)
    pred = pred + resid[rng.integers(len(resid), size=len(pred))]
    on, off = (s * dt / 1000 for s in task.periods["stimulus"])
    qs = reward_window(tc, task.periods, task.T)[0] * dt / 1000
    B, L, R = win(0, 1), win(on + 1, off), win(qs, qs + 2)
    rewarded = licked & go
    lab_light = event_responsive(pred[light][:, L].mean(1), pred[light][:, B].mean(1), control=pred[~light][:, L].mean(1))
    same = light if contingency == "reward" else ~light  # the rewarded light condition
    unrew = same & ~licked
    ctrl = pred[unrew][:, R].mean(1) if unrew.sum() >= 3 else pred[~same][:, R].mean(1)
    lab_rew = event_responsive(pred[rewarded][:, R].mean(1), pred[rewarded][:, B].mean(1), control=ctrl)
    raw = event_responsive(pred[light][:, L].mean(1), pred[light][:, B].mean(1))
    acc = float((licked == go).mean())
    return {"accuracy": round(acc, 3), "light_excited": round(lab_light["excited"], 3),
            "light_suppressed": round(lab_light["suppressed"], 3),
            "light_excited_without_control": round(raw["excited"], 3),
            "light_suppressed_without_control": round(raw["suppressed"], 3),
            "reward_excited": round(lab_rew["excited"], 3), "reward_suppressed": round(lab_rew["suppressed"], 3)}


if __name__ == "__main__":
  res = {}
  for s in sorted(int(d.name.split("_")[1]) for d in AFTER.glob("seed_*")):
    res[f"seed {s}"] = {"before (light = no water)": run(BEFORE, s, "no_reward"),
                        "after (light = water)": run(AFTER, s, "reward")}
    print(s, res[f"seed {s}"], flush=True)
  (ROOT / "runs" / OUT).write_text(json.dumps(res, indent=1))
