"""Task wrappers around NeuroGym environments.

Training uses NeuroGym's Dataset (random trials stitched into sequences).
Evaluation builds individual trials over a fixed grid of conditions so behavior can
be summarized per condition (psychometric curves, lick-probability tables).

Every task needs four periods named fixation, stimulus, delay and decision; the
response is read out during "decision".
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field

import numpy as np
import torch

warnings.filterwarnings("ignore", module="gymnasium")
import neurogym as ngym  # noqa: E402

from .envs import LOCAL_ENVS  # noqa: E402


@dataclass
class TaskConfig:
    name: str = "PerceptualDecisionMaking-v0"
    dt: float = 20.0
    timing: dict = field(
        default_factory=lambda: {"fixation": 200, "stimulus": 1000, "delay": 0, "decision": 200}
    )
    kwargs: dict = field(default_factory=dict)


def make_env(cfg: TaskConfig):
    if cfg.name in LOCAL_ENVS:
        return LOCAL_ENVS[cfg.name](dt=cfg.dt, timing=cfg.timing, **cfg.kwargs)
    return ngym.make(cfg.name, dt=cfg.dt, timing=cfg.timing, **cfg.kwargs)


def make_dataset(cfg: TaskConfig, batch_size: int, seq_len: int, seed: int):
    env = make_env(cfg)
    env.unwrapped.seed(seed)
    return ngym.Dataset(env, batch_size=batch_size, seq_len=seq_len)


def task_dims(cfg: TaskConfig) -> tuple[int, int]:
    env = make_env(cfg)
    return env.observation_space.shape[0], env.action_space.n


def condition_grid(env) -> tuple[list[dict], bool]:
    """Evaluation conditions for a task, and whether action 0 counts as a response.

    2-choice tasks (PerceptualDecisionMaking): every coherence x correct side; action 0
    is fixation and never a valid answer. Go/no-go tasks (NeedGatedCue): every need
    state x cue; action 0 (no lick) is a valid answer.
    """
    if hasattr(env, "states") and hasattr(env, "cues"):
        return [{"state": s, "cue": c} for s, c in itertools.product(env.states, env.cues)], True
    cohs = list(getattr(env, "cohs", [None]))
    sides = list(getattr(env, "choices", [0, 1]))
    grid = []
    for coh, side in itertools.product(cohs, sides):
        grid.append({"ground_truth": side} if coh is None else {"ground_truth": side, "coh": coh})
    return grid, False


@dataclass
class TrialBatch:
    inputs: torch.Tensor  # (T, B, n_inputs)
    targets: torch.Tensor  # (T, B) ground-truth action per step
    conditions: list[dict]  # per-trial conditions; "target_action" is the correct response
    periods: dict[str, tuple[int, int]]  # period name -> (start, end) step indices
    dt: float
    zero_is_response: bool = False


def eval_trials(cfg: TaskConfig, n_per_condition: int, seed: int = 0) -> TrialBatch:
    """Fixed-length evaluation trials covering the task's condition grid."""
    env = make_env(cfg).unwrapped
    env.seed(seed)
    grid, zero_is_response = condition_grid(env)
    obs, gts, conds = [], [], []
    for kw in grid:
        for _ in range(n_per_condition):
            trial = env.new_trial(**kw)
            obs.append(env.ob.copy())
            gts.append(env.gt.copy())
            cond = {k: (v.item() if hasattr(v, "item") else v) for k, v in trial.items()}
            cond["target_action"] = int(env.gt[env.start_ind["decision"]])
            conds.append(cond)
    if len({o.shape[0] for o in obs}) != 1:
        raise ValueError("eval_trials needs fixed trial timing; got variable lengths")
    periods = {p: (int(env.start_ind[p]), int(env.end_ind[p])) for p in env.start_ind}
    return TrialBatch(
        inputs=torch.tensor(np.stack(obs, axis=1), dtype=torch.float32),
        targets=torch.tensor(np.stack(gts, axis=1), dtype=torch.long),
        conditions=conds,
        periods=periods,
        dt=cfg.dt,
        zero_is_response=zero_is_response,
    )


def choices_from_outputs(outputs: torch.Tensor, batch: TrialBatch) -> np.ndarray:
    """Response = most active output averaged over the decision period.

    For 2-choice tasks the fixation output (0) is excluded.
    """
    s, e = batch.periods["decision"]
    mean_out = outputs[s:e].mean(0)  # (B, n_outputs)
    if batch.zero_is_response:
        return mean_out.argmax(-1).cpu().numpy()
    return (mean_out[:, 1:].argmax(-1) + 1).cpu().numpy()


def behavior_summary(choices: np.ndarray, batch: TrialBatch) -> dict:
    target = np.array([c["target_action"] for c in batch.conditions])
    correct = choices == target
    out = {"accuracy": float(correct.mean())}
    first = batch.conditions[0]
    if "coh" in first:
        # Psychometric curve: P(choose action 2) vs signed coherence.
        signed = np.array(
            [c["coh"] * (1 if c["ground_truth"] == 1 else -1) for c in batch.conditions]
        )
        levels = np.unique(signed)
        acc_by_coh = {}
        for coh in sorted({c["coh"] for c in batch.conditions}):
            m = np.array([c["coh"] == coh for c in batch.conditions])
            acc_by_coh[float(coh)] = float(correct[m].mean())
        out.update(
            psychometric={
                "signed_coh": levels.tolist(),
                "p_choice_2": [float((choices[signed == s] == 2).mean()) for s in levels],
            },
            accuracy_by_coh=acc_by_coh,
        )
    if "state" in first and "cue" in first:
        # Lick probability for every need state x cue.
        states = list(dict.fromkeys(c["state"] for c in batch.conditions))
        cues = list(dict.fromkeys(c["cue"] for c in batch.conditions))
        p_lick = []
        for s in states:
            row = []
            for q in cues:
                m = np.array([c["state"] == s and c["cue"] == q for c in batch.conditions])
                row.append(float((choices[m] == 1).mean()))
            p_lick.append(row)
        out["lick"] = {"states": states, "cues": cues, "p_lick": p_lick}
    return out


def condition_key(cond: dict) -> tuple:
    """Grouping key for condition-averaged trajectories."""
    if "state" in cond:
        return (cond["state"], cond["cue"])
    return (cond["ground_truth"], cond.get("coh"))
