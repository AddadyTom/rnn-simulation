"""Data-constrained training: fit an E/I RNN so that some of its units reproduce
recorded insula neurons, trial by trial, while it also reproduces the mouse's licks.

Model:   the usual EIRNN, with n_units = n_recorded + n_hidden_exc + n_inh.
         Recorded neurons are mapped to the first excitatory units. Two-photon imaging
         of layer 2/3 does not tell E from I, and most imaged cells are excitatory, so
         by default every recorded cell is treated as E and the inhibitory population is
         unobserved (set `recorded_inh_fraction` to assign a random share to I instead).
Inputs:  built from each trial's events with the VagalOpto task (vagal drive at the
         trial's frequency, odor at the task's odor time, reward when it was
         actually delivered: at the valve frame when events.csv has `reward_frame`,
         otherwise in the task's reward window).
Targets: preprocessed activity of each recorded cell (see data.py), compared with
         gain_i * rate_i + offset_i at the frame times, plus the mouse's lick choice
         through the lick readout.
Output:  a run directory in the same format as train.py (train_config.json and
         seed_<k>/model.pt), so the fitted networks can be perturbed with
         `python -m rnn_workbench experiment`, plus fit.json with held-out R^2 and the
         ROI index of each recorded unit.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import SessionData, load_session, trial_tensor
from .model import EIRNN, ModelConfig
from .vagal_task import VagalOptoTask, VagalTaskConfig, choice_window, reward_window


@dataclass
class FitConfig:
    name: str = "fit_session"
    session_dir: str = ""
    events_file: str = "events.csv"
    iscell_file: str = "iscell_final.npy"
    signal: str = "spks"  # "spks" (deconvolved), "dff", "dff_ready" (lab dF/F) or "dff_deconv" (lab dF/F + OASIS)
    target: str = "psth"  # "psth": fit condition averages (frequency x lick x reward); "single_trial"
    fs: float = 15.0
    smooth_frames: float = 1.0
    include_inhibition_trials: bool = False
    seeds: list = field(default_factory=lambda: [0])
    task: VagalTaskConfig = field(default_factory=lambda: VagalTaskConfig(gain_cv=0.0, input_noise=0.1))
    n_hidden_exc: int = 50
    n_inh: int = 0  # 0 = choose so that inhibitory units are 20% of the network
    recorded_inh_fraction: float = 0.0
    # Optional downstream "decision" area (area 1) that reads from the insula (area 0)
    # through E projections and holds the lick readout. Not fitted to any cells.
    n_down_exc: int = 0
    n_down_inh: int = 0
    down_feedback: bool = True  # area 1 E units also project back to the insula
    tau: float = 100.0
    noise_std: float = 0.02
    steps: int = 2000
    batch_size: int = 16
    lr: float = 1e-3  # 1e-2 can blow up the E/I balance in one Adam step on larger networks
    behavior_weight: float = 0.02  # lick-choice cross-entropy weight; activity MSE is ~0.01-0.05 for data scaled to [0, 1], so larger values let the lick loss dominate
    rate_l2: float = 1e-4
    weight_l2: float = 1e-5
    test_fraction: float = 0.2
    eval_every: int = 100
    calcium_tau_ms: float = 0.0  # > 0: slow each fitted unit's rate with this calcium decay before comparing (way B)
    learn_calcium_tau: bool = True  # learn one decay per cell, starting at calcium_tau_ms
    out_dir: str = "runs"


def load_fit_config(d: dict) -> FitConfig:
    d = dict(d)
    task = VagalTaskConfig(**{"gain_cv": 0.0, "input_noise": 0.1, **d.pop("task", {})})
    return FitConfig(task=task, **d)


class Readout(nn.Module):
    """Per-neuron map from model rate to the recorded signal: optionally a calcium
    filter (c[t] = a c[t-1] + (1 - a) r[t], a = exp(-dt / tau), one tau per cell), then
    an affine map with gain >= 0.

    forward(r, steps): r is (T, B, n) over all model steps; returns (T_frames, B, n) at
    the model steps in `steps` (the frame times).
    """

    def __init__(self, n: int, calcium_tau_ms: float = 0.0, dt: float = 20.0, learn_tau: bool = True):
        super().__init__()
        self.log_gain = nn.Parameter(torch.zeros(n))
        self.offset = nn.Parameter(torch.zeros(n))
        self.dt = dt
        self.calcium = calcium_tau_ms > 0
        if self.calcium:
            log_tau = torch.full((n,), float(np.log(calcium_tau_ms)))
            if learn_tau:
                self.log_tau = nn.Parameter(log_tau)
            else:
                self.register_buffer("log_tau", log_tau)

    def tau_ms(self) -> torch.Tensor | None:
        return self.log_tau.exp().clamp(min=self.dt) if self.calcium else None

    def forward(self, r, steps=None):
        if self.calcium:
            a = torch.exp(-self.dt / self.tau_ms())
            c, out = torch.zeros_like(r[0]), []
            for t in range(r.shape[0]):
                c = a * c + (1 - a) * r[t]
                out.append(c)
            r = torch.stack(out)
        if steps is not None:
            r = r[steps]
        return F.softplus(self.log_gain) * 1.4427 * r + self.offset  # softplus(0)*1.44 ~= 1


def build_trials(session: SessionData, task: VagalOptoTask, rng: np.random.Generator, include_inhibition: bool):
    """Model inputs and data targets for every usable trial."""
    cfg, per = task.cfg, task.periods
    pre_ms = cfg.timing["baseline"]
    total_ms = task.T * cfg.dt
    act, ok = trial_tensor(session, pre_ms, total_ms)
    keep = ok & np.array([include_inhibition or not e.get("inhibition", 0) for e in session.events])
    events = [e for e, k in zip(session.events, keep) if k]
    act = act[keep]
    # Check the trial structure matches the task timing (odor 4 s after stimulus onset by default).
    expected = (cfg.timing["stimulus"] + cfg.timing.get("delay", 0)) * session.fs / 1000.0
    lag = np.array([e["odor_onset_frame"] - e["stim_onset_frame"] for e in events])
    if len(lag) and np.abs(lag - expected).max() > 1.5:
        raise ValueError(f"odor onset is {lag.min()}-{lag.max()} frames after stimulus onset, but the task "
                         f"timing expects {expected:.1f}; adjust task.timing to match the rig")
    freqs = [e["freq_hz"] for e in events]
    x, y, w, conds = task.build(freqs, rng, licked=np.array([e["licked"] for e in events]))
    x[:, :, 2] = 0.0
    reward_steps = [reward_steps_of(e, task, session.fs) for e in events]
    for b, (e, (qs, qe)) in enumerate(zip(events, reward_steps)):  # reward input = what the mouse actually got
        if e["rewarded"]:
            x[qs:qe, b, 2] = 1.0
    frame_steps = np.round(np.arange(act.shape[1]) * 1000.0 / session.fs / cfg.dt).astype(int)
    frame_steps = np.minimum(frame_steps, task.T - 1)
    licked = np.array([e["licked"] for e in events])
    return {"x": torch.from_numpy(x), "act": torch.from_numpy(act), "licked": torch.from_numpy(licked),
            "frame_steps": torch.from_numpy(frame_steps), "events": events, "conds": conds,
            "reward_steps": reward_steps}


def reward_steps_of(event: dict, task: VagalOptoTask, fs: float) -> tuple[int, int]:
    """Model steps with the reward input on for one recorded trial. With a valve frame
    in the events, reward starts there and lasts `reward_ms`; otherwise the task's
    reward window is used."""
    cfg = task.cfg
    if event.get("reward_frame") is None:
        return reward_window(cfg, task.periods, task.T)
    lag_ms = (event["reward_frame"] - event["stim_onset_frame"]) * 1000.0 / fs
    s = task.periods["stimulus"][0] + int(round(lag_ms / cfg.dt))
    s = int(np.clip(s, 0, task.T))
    return s, min(task.T, s + int(round(cfg.reward_ms / cfg.dt)))


def r2_scores(pred: np.ndarray, data: np.ndarray, groups: list) -> dict:
    """Single-trial R^2 (pooled over cells) and trial-averaged R^2 (PSTHs per condition),
    plus the per-cell trial-averaged R^2."""
    def r2(p, d, axis=None):
        ss_res = ((p - d) ** 2).sum(axis)
        ss_tot = ((d - d.mean(axis=axis, keepdims=True) if axis is not None else d - d.mean()) ** 2).sum(axis)
        return 1 - ss_res / np.maximum(ss_tot, 1e-12)
    single = float(r2(pred, data))
    keys = sorted(set(groups))
    psth_p = np.stack([pred[[g == k for g in groups]].mean(0) for k in keys])
    psth_d = np.stack([data[[g == k for g in groups]].mean(0) for k in keys])
    per_cell = r2(psth_p.reshape(-1, psth_p.shape[-1]), psth_d.reshape(-1, psth_d.shape[-1]), axis=0)
    return {"r2_single_trial": single, "r2_psth": float(r2(psth_p, psth_d)),
            "r2_psth_per_cell_median": float(np.median(per_cell)), "r2_psth_per_cell": per_cell.tolist()}


def fit_one(cfg: FitConfig, session: SessionData, seed: int, out_dir: Path, log=print) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    task = VagalOptoTask(cfg.task)
    data = build_trials(session, task, rng, cfg.include_inhibition_trials)
    n_trials, _, n_rec = data["act"].shape
    n_rec_inh = int(round(cfg.recorded_inh_fraction * n_rec))
    n_ins_exc = (n_rec - n_rec_inh) + cfg.n_hidden_exc
    n_inh = cfg.n_inh or max(1, int(round(0.25 * n_ins_exc)))  # 20% of the insula
    n_exc = n_ins_exc + cfg.n_down_exc  # downstream E units sit at the end of the E block
    n_units = n_exc + n_inh + n_rec_inh + cfg.n_down_inh  # downstream I units at the very end
    # Unit indices of the recorded cells: E units first, then (optionally) some I units.
    exc_ids = np.arange(n_rec - n_rec_inh)
    inh_ids = n_exc + np.arange(n_rec_inh)
    order = rng.permutation(n_rec)
    unit_of_cell = np.empty(n_rec, dtype=int)
    unit_of_cell[order[: n_rec - n_rec_inh]] = exc_ids
    unit_of_cell[order[n_rec - n_rec_inh:]] = inh_ids
    area_kw = {}
    if cfg.n_down_exc > 0:
        area = np.zeros(n_units, dtype=int)
        area[n_ins_exc:n_exc] = 1
        area[n_units - cfg.n_down_inh:] = 1
        # The vagal stimulus reaches only the insula; odor and reward reach both areas.
        area_kw = {"area_of_unit": area.tolist(), "input_areas": [[0], [0, 1], [0, 1]], "readout_area": 1,
                   "feedback": cfg.down_feedback}
    mcfg = ModelConfig(n_units=n_units, frac_exc=n_exc / n_units, n_inputs=3, n_outputs=2, dt=cfg.task.dt,
                       tau=cfg.tau, noise_std=cfg.noise_std, dale=True, **area_kw)
    model = EIRNN(mcfg)
    assert model.n_exc == n_exc, "rounding of frac_exc changed the E/I split"
    readout = Readout(n_rec, cfg.calcium_tau_ms, cfg.task.dt, cfg.learn_calcium_tau)
    params = list(model.parameters()) + list(readout.parameters())
    # Adam moves every weight by about lr per step, and in a network of N units the
    # moves add up over N inputs. Scaling the recurrent weights' step by 100 / N keeps
    # the change in each unit's total input the same as in a 100-unit network, which
    # stops the first steps from breaking the E/I balance in large fitted networks.
    rec_scale = min(1.0, 100.0 / n_units)
    rest = [p for p in params if p is not model.w_raw]
    opt = torch.optim.Adam([{"params": [model.w_raw], "lr": cfg.lr * rec_scale}, {"params": rest, "lr": cfg.lr}])

    # Train / test split, stratified by (frequency, lick).
    groups = [(e["freq_hz"], e["licked"]) for e in data["events"]]
    test = np.zeros(n_trials, dtype=bool)
    for g in set(groups):
        idx = np.flatnonzero([gg == g for gg in groups])
        n_test = int(round(cfg.test_fraction * len(idx)))
        test[rng.choice(idx, size=n_test, replace=False)] = True
    train_idx, test_idx = np.flatnonzero(~test), np.flatnonzero(test)
    if len(train_idx) == 0:
        raise ValueError("no training trials")
    units = torch.from_numpy(unit_of_cell)
    fs_steps = data["frame_steps"]
    rs, re = choice_window(task.cfg, task.periods, task.T)  # licks before the reward

    def forward(idx, noise=True):
        x = data["x"][:, idx]
        out, rates = model(x, noise=noise)
        pred = readout(rates[:, :, units], fs_steps).transpose(0, 1)  # (B, n_frames, n_rec)
        lick_logit = out[rs:re].mean(0)  # (B, 2)
        return pred, lick_logit, rates

    # Condition-average targets: one noise-free input and one mean response per
    # (frequency, lick, reward) condition of the training trials, weighted by trial count.
    cond_key = [(e["freq_hz"], e["licked"], e["rewarded"]) for e in data["events"]]
    cond_list = sorted({cond_key[i] for i in train_idx})
    quiet = VagalOptoTask(VagalTaskConfig(**{**asdict(cfg.task), "input_noise": 0.0, "gain_cv": 0.0}))
    xc, _, _, _ = quiet.build([c[0] for c in cond_list], rng, licked=np.array([c[1] for c in cond_list]))
    xc[:, :, 2] = 0.0
    for j, c in enumerate(cond_list):  # reward at the condition's median valve time
        if c[2]:
            win = np.array([data["reward_steps"][i] for i in train_idx if cond_key[i] == c])
            qs, qe = (int(v) for v in np.median(win, axis=0))
            xc[qs:qe, j, 2] = 1.0
    xc = torch.from_numpy(xc)
    yc = torch.stack([data["act"][[i for i in train_idx if cond_key[i] == c]].mean(0) for c in cond_list])
    wc = torch.tensor([sum(cond_key[i] == c for i in train_idx) for c in cond_list], dtype=torch.float32)
    lc = torch.tensor([c[1] for c in cond_list])

    def forward_x(x, noise=True):
        out, rates = model(x, noise=noise)
        pred = readout(rates[:, :, units], fs_steps).transpose(0, 1)
        return pred, out[rs:re].mean(0), rates

    history, t0 = [], time.time()
    for step in range(1, cfg.steps + 1):
        if cfg.target == "psth":
            pred, lick_logit, rates = forward_x(xc)
            per = ((pred - yc) ** 2).mean((1, 2))
            loss_act = (per * wc).sum() / wc.sum()
            loss_beh = (F.cross_entropy(lick_logit, lc, reduction="none") * wc).sum() / wc.sum()
        elif cfg.target == "single_trial":
            idx = rng.choice(train_idx, size=min(cfg.batch_size, len(train_idx)), replace=False)
            pred, lick_logit, rates = forward(idx)
            loss_act = F.mse_loss(pred, data["act"][idx])
            loss_beh = F.cross_entropy(lick_logit, data["licked"][idx].long())
        else:
            raise ValueError(f"unknown target {cfg.target!r}")
        loss = (loss_act + cfg.behavior_weight * loss_beh + cfg.rate_l2 * rates.pow(2).mean()
                + cfg.weight_l2 * model.w_rec.pow(2).sum())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"seed {seed}: the loss became {loss.item()} at step {step}; "
                                     f"the network's activity blew up, so try a lower lr (now {cfg.lr})")
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % cfg.eval_every == 0 or step == cfg.steps:
            ev = evaluate_fit(forward, data, test_idx if len(test_idx) else train_idx, groups)
            history.append({"step": step, "loss_activity": loss_act.item(), "loss_behavior": loss_beh.item(),
                            **{k: v for k, v in ev.items() if k != "r2_psth_per_cell"}})
            log(f"  seed {seed} step {step:5d}  mse {loss_act.item():.4f}  test R2 psth {ev['r2_psth']:.3f}  "
                f"single {ev['r2_single_trial']:.3f}  lick acc {ev['lick_accuracy']:.2f}  ({time.time() - t0:.0f}s)")
    final_test = evaluate_fit(forward, data, test_idx if len(test_idx) else train_idx, groups)
    if len(test_idx):
        ceiling = noise_ceiling(data["act"].numpy(), groups, train_idx, test_idx)
        final_test["ceiling"] = ceiling
        for k in ("r2_single_trial", "r2_psth"):
            final_test[f"{k}_fraction_of_ceiling"] = final_test[k] / ceiling[k] if ceiling[k] > 0 else float("nan")
        log(f"  seed {seed} test R2 psth {final_test['r2_psth']:.3f} (ceiling {ceiling['r2_psth']:.3f}, baseline "
            f"{ceiling['baseline_r2_psth']:.3f}); single trial {final_test['r2_single_trial']:.3f} (ceiling "
            f"{ceiling['r2_single_trial']:.3f}, baseline {ceiling['baseline_r2_single_trial']:.3f})")
    final_train = evaluate_fit(forward, data, train_idx, groups)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_cfg": asdict(model.cfg),
                "readout": readout.state_dict(), "unit_of_cell": unit_of_cell.tolist()}, out_dir / "model.pt")
    result = {"seed": seed, "n_trials": n_trials, "n_train": len(train_idx), "n_test": len(test_idx),
              "n_recorded": n_rec, "n_units": n_units, "n_exc": n_exc, "model_cfg": asdict(mcfg), "unit_of_cell": unit_of_cell.tolist(),
              "roi_of_cell": session.cell_ids.tolist(), "test": final_test, "train": final_train,
              "history": history, "fit_seconds": time.time() - t0}
    if readout.calcium:
        tau = readout.tau_ms().detach().numpy()
        result["calcium_tau_ms"] = {"median": float(np.median(tau)), "p10": float(np.percentile(tau, 10)),
                                    "p90": float(np.percentile(tau, 90))}
    (out_dir / "fit.json").write_text(json.dumps(result))
    (out_dir / "metrics.json").write_text(json.dumps({"seed": seed, "final": {"accuracy": final_test["lick_accuracy"]}}))
    return result


def noise_ceiling(act: np.ndarray, groups: list, train_idx, test_idx) -> dict:
    """How well the test trials could be predicted at best, and by a simple baseline.

    r2_psth, r2_single_trial: the expected R^2 of the TRUE condition means (frequency x
    lick) against the test data. Trial-to-trial variability inside a condition cannot be
    predicted from its inputs, so it is subtracted: estimated from all trials of the
    condition, and divided by the number of test trials for the averaged (PSTH) score.
    baseline_*: the R^2 of predicting each test trial with the mean of the training
    trials of its condition. A good model should land between the two."""
    test_groups = [groups[i] for i in test_idx]
    mean_all = act[train_idx].mean(0)
    psth = {}
    for g in set(groups[i] for i in train_idx):
        psth[g] = act[[i for i in train_idx if groups[i] == g]].mean(0)
    pred = np.stack([psth.get(groups[i], mean_all) for i in test_idx])
    base = r2_scores(pred, act[test_idx], test_groups)
    data = act[test_idx]
    keys = sorted(set(test_groups))
    noise_single, noise_psth = 0.0, 0.0
    test_psth = []
    for k in keys:
        all_k = act[[i for i, g in enumerate(groups) if g == k]]
        n_test = sum(g == k for g in test_groups)
        var = all_k.var(0, ddof=1) if len(all_k) > 1 else np.zeros(all_k.shape[1:])
        noise_single += n_test * var.sum()
        noise_psth += var.sum() / n_test
        test_psth.append(data[[g == k for g in test_groups]].mean(0))
    test_psth = np.stack(test_psth)
    ss_single = ((data - data.mean()) ** 2).sum()
    ss_psth = ((test_psth - test_psth.mean()) ** 2).sum()
    return {"r2_single_trial": float(1 - noise_single / max(ss_single, 1e-12)),
            "r2_psth": float(1 - noise_psth / max(ss_psth, 1e-12)),
            "baseline_r2_single_trial": base["r2_single_trial"], "baseline_r2_psth": base["r2_psth"]}


def evaluate_fit(forward, data, idx, groups) -> dict:
    with torch.no_grad():
        pred, lick_logit, _ = forward(idx, noise=False)
    p, d = pred.numpy(), data["act"][idx].numpy()
    out = r2_scores(p, d, [groups[i] for i in idx])
    out["lick_accuracy"] = float((lick_logit.argmax(-1) == data["licked"][idx]).float().mean())
    return out


def fit(cfg: FitConfig, log=print) -> Path:
    session = load_session(Path(cfg.session_dir), cfg.events_file, cfg.iscell_file, cfg.signal, cfg.fs,
                           cfg.smooth_frames)
    log(f"Loaded {session.traces.shape[0]} cells, {session.traces.shape[1]} frames, {len(session.events)} trials")
    root = Path(cfg.out_dir) / cfg.name
    root.mkdir(parents=True, exist_ok=True)
    # Same shape as train.py's config, so experiment.py can load the fitted networks.
    train_like = {"name": cfg.name, "seeds": cfg.seeds, "task": asdict(cfg.task), "fit": asdict(cfg)}
    summary = []
    for seed in cfg.seeds:
        log(f"Fitting seed {seed}")
        r = fit_one(cfg, session, seed, root / f"seed_{seed}", log=log)
        summary.append({"seed": seed, "r2_psth": r["test"]["r2_psth"], "r2_single_trial": r["test"]["r2_single_trial"],
                        "lick_accuracy": r["test"]["lick_accuracy"]})
        train_like["model"] = r["model_cfg"]
    (root / "train_config.json").write_text(json.dumps(train_like, indent=2))
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    return root
