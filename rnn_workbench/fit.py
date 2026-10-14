"""Data-constrained training: fit an E/I RNN to recorded neural activity.

Instead of (or in addition to) learning the task, the network learns to reproduce
recordings. Each recorded neuron is paired with one model unit of the same type
(E with E, I with I); the loss is the squared difference between the unit's
condition-averaged rate and the neuron's condition-averaged rate (its PSTH),
summed over neurons, time steps and conditions. Unpaired units are free.

Recordings are an .npz file with
    psth:    (C, T, N) trial-averaged rate of N neurons in C conditions
    is_exc:  (N,) bool, the neuron's type (optional; all E if missing)
and the condition order of `eval_trials` for the task (state x cue for the
need-gated task). Without real recordings, `record_teacher` makes them from a
trained network, which lets us check the method against a known ground truth:
fit a fresh network to part of the teacher's neurons, then ask whether it
predicts the teacher's response to perturbations it never saw.

    python -m rnn_workbench fit configs/fit_need.yaml
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .model import EIRNN, ModelConfig
from .tasks import TaskConfig, behavior_summary, choices_from_outputs, condition_key, eval_trials, task_dims
from .train import load_model


@dataclass
class FitConfig:
    name: str = "need_fit"
    task: TaskConfig = field(default_factory=TaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    recordings: str = ""  # .npz with psth (C, T, N) and is_exc (N,); or empty to use a teacher
    teacher: str = ""  # trained network to record from when no recordings are given
    n_recorded: int = 60  # teacher neurons "recorded", split E/I in proportion
    task_weight: float = 0.0  # 0 = fit recordings only; > 0 adds the behavior loss (hybrid)
    steps: int = 1500
    trials_per_condition: int = 8  # trials averaged into each PSTH during training
    lr: float = 1e-3
    rate_l2: float = 1e-4
    readout_steps: int = 300  # readout trained afterwards on the frozen network
    seed: int = 0
    out_dir: str = "runs"


def load_fit_config(d: dict) -> FitConfig:
    d = dict(d)
    task = TaskConfig(**d.pop("task", {}))
    model = ModelConfig(**d.pop("model", {}))
    return FitConfig(task=task, model=model, **d)


def condition_index(batch) -> tuple[np.ndarray, list]:
    keys = list(dict.fromkeys(condition_key(c) for c in batch.conditions))
    idx = np.array([keys.index(condition_key(c)) for c in batch.conditions])
    return idx, keys


def psth(rates: torch.Tensor, idx: np.ndarray, n_cond: int) -> torch.Tensor:
    """Condition-averaged rates: (T, B, N) -> (C, T, N)."""
    return torch.stack([rates[:, idx == c].mean(1) for c in range(n_cond)])


def record_teacher(teacher: EIRNN, task: TaskConfig, n_recorded: int, rng: np.random.Generator,
                   n_per_condition: int = 50, seed: int = 999) -> dict:
    """Simulated recording session: PSTHs of a random, type-proportional subset of units."""
    n_exc = teacher.n_exc
    n_inh = teacher.cfg.n_units - n_exc
    k_exc = int(round(n_recorded * n_exc / teacher.cfg.n_units))
    k_inh = min(n_recorded - k_exc, n_inh)
    units = np.concatenate([
        np.sort(rng.choice(n_exc, k_exc, replace=False)),
        n_exc + np.sort(rng.choice(n_inh, k_inh, replace=False)),
    ])
    batch = eval_trials(task, n_per_condition, seed=seed)
    idx, keys = condition_index(batch)
    with torch.no_grad():
        _, rates = teacher(batch.inputs)
    return {
        "psth": psth(rates[:, :, units], idx, len(keys)).numpy(),
        "is_exc": units < n_exc,
        "teacher_units": units,
    }


def pair_units(model: EIRNN, is_exc: np.ndarray) -> np.ndarray:
    """Model unit for each recorded neuron, matching cell type."""
    n_e, n_i = int(is_exc.sum()), int((~is_exc).sum())
    n_inh_model = model.cfg.n_units - model.n_exc
    if n_e > model.n_exc or n_i > n_inh_model:
        raise ValueError(f"need {n_e} E and {n_i} I units; model has {model.n_exc} and {n_inh_model}")
    paired = np.empty(len(is_exc), dtype=int)
    paired[is_exc] = np.arange(n_e)
    paired[~is_exc] = model.n_exc + np.arange(n_i)
    return paired


def fit_r2(pred: np.ndarray, target: np.ndarray) -> dict:
    """Variance explained across everything, and the median over neurons."""
    sse = ((pred - target) ** 2).sum()
    sst = ((target - target.mean()) ** 2).sum()
    per = []
    for n in range(target.shape[-1]):
        t, p = target[..., n], pred[..., n]
        v = ((t - t.mean()) ** 2).sum()
        if v > 1e-8:
            per.append(1 - ((p - t) ** 2).sum() / v)
    return {"r2": float(1 - sse / sst), "median_neuron_r2": float(np.median(per))}


def train_readout(model: EIRNN, task: TaskConfig, steps: int, seed: int) -> None:
    """Train only the readout on the frozen network (behavior decoded from activity)."""
    opt = torch.optim.Adam([model.w_out, model.b_out], lr=1e-2)
    ce = nn.CrossEntropyLoss()
    batch = eval_trials(task, 20, seed=seed)
    with torch.no_grad():
        _, rates = model(batch.inputs)
    r_e = rates[:, :, : model.n_exc]
    n_out = model.w_out.shape[0]
    for _ in range(steps):
        out = r_e @ model.w_out.T + model.b_out
        loss = ce(out.reshape(-1, n_out), batch.targets.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()


def fit(cfg: FitConfig, log=print) -> Path:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    out_dir = Path(cfg.out_dir) / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.recordings:
        rec = dict(np.load(cfg.recordings))
        rec.setdefault("is_exc", np.ones(rec["psth"].shape[-1], dtype=bool))
    else:
        rec = record_teacher(load_model(Path(cfg.teacher)), cfg.task, cfg.n_recorded, rng)
    target = torch.tensor(rec["psth"], dtype=torch.float32)
    is_exc = np.asarray(rec["is_exc"], dtype=bool)

    n_in, n_out = task_dims(cfg.task)
    cfg.model.n_inputs, cfg.model.n_outputs, cfg.model.dt = int(n_in), int(n_out), cfg.task.dt
    model = EIRNN(cfg.model)
    paired = pair_units(model, is_exc)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    ce = nn.CrossEntropyLoss()
    t0, history = time.time(), []
    for step in range(1, cfg.steps + 1):
        batch = eval_trials(cfg.task, cfg.trials_per_condition, seed=10_000 + step)
        idx, keys = condition_index(batch)
        out, rates = model(batch.inputs)
        loss_fit = (psth(rates[:, :, paired], idx, len(keys)) - target).pow(2).mean()
        loss = loss_fit + cfg.rate_l2 * rates.pow(2).mean()
        if cfg.task_weight > 0:
            loss = loss + cfg.task_weight * ce(out.reshape(-1, n_out), batch.targets.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 250 == 0 or step == cfg.steps:
            history.append({"step": step, "fit_mse": float(loss_fit)})
            log(f"  step {step:5d}  fit mse {loss_fit:.5f}  ({time.time() - t0:.0f}s)")

    if cfg.task_weight == 0:
        train_readout(model, cfg.task, cfg.readout_steps, seed=cfg.seed + 77)
    model.eval()
    batch = eval_trials(cfg.task, 50, seed=4321)  # held-out trials
    idx, keys = condition_index(batch)
    with torch.no_grad():
        out, rates = model(batch.inputs)
    pred = psth(rates[:, :, paired], idx, len(keys)).numpy()
    metrics = {
        "fit": fit_r2(pred, rec["psth"]),
        "behavior": behavior_summary(choices_from_outputs(out, batch), batch),
        "n_recorded": int(len(is_exc)),
        "task_weight": cfg.task_weight,
        "history": history,
        "train_seconds": time.time() - t0,
    }
    torch.save({"state_dict": model.state_dict(), "model_cfg": asdict(cfg.model)}, out_dir / "model.pt")
    np.savez(out_dir / "recordings.npz", paired=paired, **rec)
    (out_dir / "fit_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log(f"  R2 {metrics['fit']['r2']:.3f}  accuracy {metrics['behavior']['accuracy']:.3f}")
    return out_dir
