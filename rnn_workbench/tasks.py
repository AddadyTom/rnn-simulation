"""Task wrappers around NeuroGym environments.

Training uses NeuroGym's Dataset (random trials stitched into sequences).
Evaluation builds individual trials with controlled conditions so behavior can be
summarized per stimulus strength (psychometric curves).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import torch

warnings.filterwarnings("ignore", module="gymnasium")
import neurogym as ngym  # noqa: E402


@dataclass
class TaskConfig:
    name: str = "PerceptualDecisionMaking-v0"
    dt: float = 20.0
    timing: dict = field(
        default_factory=lambda: {"fixation": 200, "stimulus": 1000, "delay": 0, "decision": 200}
    )
    kwargs: dict = field(default_factory=dict)


def make_env(cfg: TaskConfig):
    return ngym.make(cfg.name, dt=cfg.dt, timing=cfg.timing, **cfg.kwargs)


def make_dataset(cfg: TaskConfig, batch_size: int, seq_len: int, seed: int):
    env = make_env(cfg)
    env.unwrapped.seed(seed)
    return ngym.Dataset(env, batch_size=batch_size, seq_len=seq_len)


def task_dims(cfg: TaskConfig) -> tuple[int, int]:
    env = make_env(cfg)
    return env.observation_space.shape[0], env.action_space.n


@dataclass
class TrialBatch:
    inputs: torch.Tensor  # (T, B, n_inputs)
    targets: torch.Tensor  # (T, B) ground-truth action per step
    conditions: list[dict]  # per-trial conditions, e.g. {"coh": 12.8, "ground_truth": 1}
    periods: dict[str, tuple[int, int]]  # period name -> (start, end) step indices
    dt: float


def eval_trials(cfg: TaskConfig, n_per_condition: int, seed: int = 0) -> TrialBatch:
    """Fixed-length evaluation trials covering every coherence x choice side.

    Only tasks whose trials have fixed timing and a `cohs` attribute are supported
    in v0 (PerceptualDecisionMaking and friends).
    """
    env = make_env(cfg).unwrapped
    env.seed(seed)
    cohs = list(getattr(env, "cohs", [None]))
    sides = list(getattr(env, "choices", [0, 1]))
    obs, gts, conds = [], [], []
    for coh in cohs:
        for side in sides:
            for _ in range(n_per_condition):
                kw = {"ground_truth": side}
                if coh is not None:
                    kw["coh"] = coh
                trial = env.new_trial(**kw)
                obs.append(env.ob.copy())
                gts.append(env.gt.copy())
                conds.append({k: (v.item() if hasattr(v, "item") else v) for k, v in trial.items()})
    lengths = {o.shape[0] for o in obs}
    if len(lengths) != 1:
        raise ValueError("eval_trials needs fixed trial timing; got variable lengths")
    periods = {p: (int(env.start_ind[p]), int(env.end_ind[p])) for p in env.start_ind}
    return TrialBatch(
        inputs=torch.tensor(np.stack(obs, axis=1), dtype=torch.float32),
        targets=torch.tensor(np.stack(gts, axis=1), dtype=torch.long),
        conditions=conds,
        periods=periods,
        dt=cfg.dt,
    )


def choices_from_outputs(outputs: torch.Tensor, batch: TrialBatch) -> np.ndarray:
    """Choice = most active non-fixation output, averaged over the decision period.

    Returns choice indices in action space (1..n_actions-1).
    """
    s, e = batch.periods["decision"]
    mean_out = outputs[s:e].mean(0)  # (B, n_outputs)
    return (mean_out[:, 1:].argmax(-1) + 1).cpu().numpy()


def behavior_summary(choices: np.ndarray, batch: TrialBatch) -> dict:
    """Accuracy and psychometric curve: P(choose action 2) vs signed coherence."""
    gt_action = np.array([c["ground_truth"] for c in batch.conditions]) + 1
    correct = choices == gt_action
    out = {"accuracy": float(correct.mean())}
    if "coh" in batch.conditions[0]:
        signed = np.array(
            [c["coh"] * (1 if c["ground_truth"] == 1 else -1) for c in batch.conditions]
        )
        levels = np.unique(signed)
        p_right = [float((choices[signed == s] == 2).mean()) for s in levels]
        acc_by_coh = {}
        for coh in sorted({c["coh"] for c in batch.conditions}):
            m = np.array([c["coh"] == coh for c in batch.conditions])
            acc_by_coh[float(coh)] = float(correct[m].mean())
        out.update(
            psychometric={"signed_coh": levels.tolist(), "p_choice_2": p_right},
            accuracy_by_coh=acc_by_coh,
        )
    return out
