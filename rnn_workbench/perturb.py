"""Virtual perturbations: lesions, silencing and optogenetic-style current injection.

A perturbation is described declaratively (usually from a YAML config):

    target:  which units
        population: "E" | "I" | "all"   pool to draw from
        fraction:   0.3                 fraction of that pool (or)
        count:      20                  number of units (or)
        units:      [3, 17, 42]         explicit unit indices
        select:     "random" | "top_choice"   how to pick within the pool
    kind:    "silence"  clamp rates to 0 in the window (acute, like optogenetic inhibition)
             "lesion"   clamp rates to 0 for the whole trial (permanent ablation)
             "current"  add `strength` to the units' input in the window (+ excite, - inhibit)
    window:  [start_ms, end_ms] relative to trial start, or a period name such as
             "stimulus", or "all"
    strength: float, only for kind="current"

`control_targets` draws a size-matched random set from the same pool, which is the
baseline every perturbation effect is compared against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .model import EIRNN
from .tasks import TrialBatch


@dataclass
class Perturbation:
    name: str = "perturbation"
    kind: str = "silence"
    target: dict = field(default_factory=lambda: {"population": "I", "fraction": 0.3})
    window: object = "all"
    strength: float = 0.0


def _pool(model: EIRNN, population: str) -> np.ndarray:
    is_exc = model.is_exc.cpu().numpy()
    if population == "E":
        return np.flatnonzero(is_exc)
    if population == "I":
        return np.flatnonzero(~is_exc)
    if population == "all":
        return np.arange(model.cfg.n_units)
    raise ValueError(f"unknown population {population!r}")


def _n_targets(target: dict, pool_size: int) -> int:
    if "count" in target:
        return min(int(target["count"]), pool_size)
    return int(round(float(target.get("fraction", 1.0)) * pool_size))


def choice_selectivity(rates: torch.Tensor, batch: TrialBatch) -> np.ndarray:
    """|mean rate on choice-2 trials - mean rate on choice-1 trials| during the stimulus."""
    s, e = batch.periods["stimulus"]
    gt = np.array([c["ground_truth"] for c in batch.conditions])
    r = rates[s:e].mean(0).cpu().numpy()  # (B, N)
    return np.abs(r[gt == 1].mean(0) - r[gt == 0].mean(0))


def resolve_targets(
    model: EIRNN, target: dict, rng: np.random.Generator, selectivity: np.ndarray | None = None
) -> np.ndarray:
    if "units" in target:
        return np.asarray(target["units"], dtype=int)
    pool = _pool(model, target.get("population", "all"))
    n = _n_targets(target, len(pool))
    select = target.get("select", "random")
    if select == "random":
        return np.sort(rng.choice(pool, size=n, replace=False))
    if select == "top_choice":
        if selectivity is None:
            raise ValueError("select='top_choice' needs choice selectivity")
        return np.sort(pool[np.argsort(selectivity[pool])[::-1][:n]])
    raise ValueError(f"unknown select {select!r}")


def control_targets(model: EIRNN, target: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    """Size-matched random units from the same pool (the null for any targeted effect)."""
    pool = _pool(model, target.get("population", "all"))
    return np.sort(rng.choice(pool, size=min(n, len(pool)), replace=False))


def window_steps(window, batch: TrialBatch, T: int) -> tuple[int, int]:
    if window in (None, "all"):
        return 0, T
    if isinstance(window, str):
        return batch.periods[window]
    start_ms, end_ms = window
    return int(round(start_ms / batch.dt)), int(round(end_ms / batch.dt))


def build_masks(
    pert: Perturbation, units: np.ndarray, batch: TrialBatch, n_units: int
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    T, B, _ = batch.inputs.shape
    s, e = (0, T) if pert.kind == "lesion" else window_steps(pert.window, batch, T)
    if pert.kind in ("silence", "lesion"):
        silence = torch.zeros(T, B, n_units, dtype=torch.bool)
        silence[s:e, :, units] = True
        return silence, None
    if pert.kind == "current":
        current = torch.zeros(T, B, n_units)
        current[s:e, :, units] = float(pert.strength)
        return None, current
    raise ValueError(f"unknown perturbation kind {pert.kind!r}")
