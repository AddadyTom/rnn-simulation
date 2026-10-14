"""Does a fitted network predict the teacher's response to perturbations it never saw?

    python scripts/fit_perturbation_check.py runs/need_fit

Silences recorded neurons (5 I or 10 E, cue + delay) in the teacher and their paired
units in the fitted network, and correlates the change in the other recorded neurons'
activity and in licking. "wrong" silences different recorded neurons in the fitted
network, the baseline for how much the pairing matters.
"""
import sys, json
from pathlib import Path
import numpy as np, torch
from rnn_workbench.fit import load_fit_config, condition_index, psth
from rnn_workbench.train import load_model
from rnn_workbench.tasks import eval_trials, choices_from_outputs, behavior_summary
torch.set_num_threads(2)
run = Path(sys.argv[1])
cfg = load_fit_config(json.loads((run / "fit_config.json").read_text()))
rec = np.load(run / "recordings.npz")
teacher = load_model(Path(cfg.teacher)); student = load_model(run)
tu, pu, isx = rec["teacher_units"], rec["paired"], rec["is_exc"]
batch = eval_trials(cfg.task, 40, seed=777)
idx, keys = condition_index(batch); C = len(keys)
T, B, _ = batch.inputs.shape
s0, _ = batch.periods["stimulus"]; _, d1 = batch.periods["delay"]

def run_net(net, units):
    sil = torch.zeros(T, B, net.cfg.n_units, dtype=torch.bool)
    if len(units): sil[s0:d1, :, units] = True
    torch.manual_seed(0)
    with torch.no_grad(): out, r = net(batch.inputs, silence=sil)
    return out, r

def lick(out): return np.array(behavior_summary(choices_from_outputs(out, batch), batch)["lick"]["p_lick"])
o_t0, r_t0 = run_net(teacher, []); o_s0, r_s0 = run_net(student, [])
P_t0, P_s0 = psth(r_t0[:, :, tu], idx, C).numpy(), psth(r_s0[:, :, pu], idx, C).numpy()
rng = np.random.default_rng(1)
rows = []
inh = np.flatnonzero(~isx); exc = np.flatnonzero(isx)
for name, pool, k in [("I", inh, 5), ("E", exc, 10)]:
    for rep in range(10):
        pick = rng.choice(pool, k, replace=False)
        wrong = rng.choice(np.setdiff1d(pool, pick), k, replace=False)
        keep = np.setdiff1d(np.arange(len(tu)), np.concatenate([pick, wrong]))
        o_t, r_t = run_net(teacher, tu[pick]); o_s, r_s = run_net(student, pu[pick]); o_w, r_w = run_net(student, pu[wrong])
        dt = psth(r_t[:, :, tu], idx, C).numpy()[..., keep] - P_t0[..., keep]
        ds = psth(r_s[:, :, pu], idx, C).numpy()[..., keep] - P_s0[..., keep]
        dw = psth(r_w[:, :, pu], idx, C).numpy()[..., keep] - P_s0[..., keep]
        cc = lambda a, b: float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        lt, ls, lw = lick(o_t) - lick(o_t0), lick(o_s) - lick(o_s0), lick(o_w) - lick(o_s0)
        rows.append(dict(pop=name, act_corr=cc(dt, ds), act_corr_wrong=cc(dt, dw),
                         lick_corr=cc(lt, ls), lick_corr_wrong=cc(lt, lw),
                         teacher_dlick=float(np.abs(lt).mean()), student_dlick=float(np.abs(ls).mean())))
for name in ("I", "E"):
    R = [r for r in rows if r["pop"] == name]
    m = {k: round(float(np.nanmean([r[k] for r in R])), 3) for k in R[0] if k != "pop"}
    print(name, m)
json.dump(rows, open(run / "perturbation_check.json", "w"), indent=1)
