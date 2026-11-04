"""Run perturbation experiments across a trained ensemble.

For every network (seed) and every perturbation in the config, the network is run on
the same evaluation trials three ways: intact, with the targeted perturbation, and
with a size-matched random control perturbation. The effect of a perturbation is its
accuracy drop relative to intact, compared with the control's drop.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .perturb import (Perturbation, build_masks, choice_selectivity, control_targets, resolve_targets,
                      stimulus_response)
from .tasks import get_task, task_config_from_dict
from .train import load_model


def run_condition(model, batch, silence=None, current=None, seed=0, task=None):
    torch.manual_seed(seed)  # identical noise across conditions
    out, rates = task.run(model, batch, silence=silence, current=current)
    return task.behavior(out, batch), rates


def control_population(target: dict, cfg: dict) -> str:
    """Pool for the size-matched control.

    A selected target (e.g. the most choice-selective E units) is compared with random
    units from its own population: "is it these units, or any E units?". A population
    target (e.g. a random 30% of I units) is compared with random units from the whole
    network: "is it this population, or any units?". Override per perturbation with
    target.control_population, or globally with control_population in the config.
    """
    if "control_population" in target:
        return target["control_population"]
    if "control_population" in cfg:
        return cfg["control_population"]
    if "units" in target or target.get("select", "random") != "random":
        return target.get("population", "all")
    return "all"


def average_behavior(runs: list[dict]) -> dict:
    """Average every numeric field (scalars, lists, nested dicts) across runs; keep
    labels and x values from the first run. NaNs (e.g. undefined thresholds) are skipped."""
    def avg(vals):
        v0 = vals[0]
        if isinstance(v0, dict):
            return {k: avg([v[k] for v in vals]) for k in v0}
        if isinstance(v0, str):
            return v0
        if isinstance(v0, (list, tuple)):
            if v0 and isinstance(v0[0], str):
                return list(v0)
            return np.nanmean(np.array(vals, dtype=float), 0).tolist()
        arr = np.array(vals, dtype=float)
        return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")
    out = avg(runs)
    for k in ("psychometric",):
        if k in out:
            for key in ("x", "signed_coh"):
                if key in out[k]:
                    out[k][key] = runs[0][k][key]
    return out


def pca_trajectories(rates_by_cond: dict, batch, n_comp: int = 2, key=None) -> dict:
    """Project condition-averaged trajectories (by choice side or stimulus presence, and
    stimulus strength) on the intact network's top PCs, so all conditions share axes."""
    key = key or (lambda c: (c["ground_truth"], c.get("coh")))
    keys = sorted({key(c) for c in batch.conditions}, key=lambda k: (k[0], -1 if k[1] is None else k[1]))
    def averages(rates):
        r = rates.cpu().numpy()
        avg = []
        for k in keys:
            m = np.array([key(c) == k for c in batch.conditions])
            avg.append(r[:, m].mean(1))
        return np.stack(avg)  # (K, T, N)
    intact = averages(rates_by_cond["intact"])
    flat = intact.reshape(-1, intact.shape[-1])
    mean = flat.mean(0)
    _, s, vt = np.linalg.svd(flat - mean, full_matrices=False)
    comps = vt[:n_comp]
    var_expl = (s[:n_comp] ** 2 / (s**2).sum()).tolist()
    proj = {name: ((averages(r) - mean) @ comps.T).tolist() for name, r in rates_by_cond.items()}
    return {"keys": [list(k) for k in keys], "proj": proj, "var_explained": var_expl}


def run_experiment(cfg: dict, log=print) -> Path:
    run_dir = Path(cfg["model_dir"])
    train_cfg = json.loads((run_dir / "train_config.json").read_text())
    # task_overrides changes the test task only, e.g. reward timing or the frequencies
    # probed, for networks fitted with a different setting.
    task = get_task(task_config_from_dict({**train_cfg["task"], **cfg.get("task_overrides", {})}))
    seeds = cfg.get("seeds") or train_cfg["seeds"]
    perts = [Perturbation(**p) for p in cfg["perturbations"]]
    n_trials = int(cfg.get("trials_per_condition", 40))
    n_draws = int(cfg.get("control_draws", 10))
    batch = task.eval_batch(n_trials, seed=int(cfg.get("eval_seed", 4321)))
    out_dir = Path(cfg.get("out_dir", "experiments")) / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {"name": cfg["name"], "model_dir": str(run_dir), "task": train_cfg["task"],
               "model": train_cfg["model"], "perturbations": [asdict(p) for p in perts],
               "seeds": {}, "periods": batch.periods, "dt": batch.dt}
    example = None
    for seed in seeds:
        model = load_model(run_dir / f"seed_{seed}")
        rng = np.random.default_rng(seed)
        intact, intact_rates = run_condition(model, batch, task=task)
        sel = choice_selectivity(intact_rates, batch)
        resp = stimulus_response(intact_rates, batch)
        seed_res = {"intact": intact, "perturbations": {}}
        rates_by_cond = {"intact": intact_rates}
        n = model.cfg.n_units
        for p in perts:
            # Random targets are redrawn n_draws times and averaged; selected targets are fixed.
            is_random = "units" not in p.target and p.target.get("select", "random") == "random"
            target_sets = [resolve_targets(model, p.target, rng, selectivity=sel, response=resp)
                           for _ in range(n_draws if is_random else 1)]
            ctrl_pool = control_population(p.target, cfg)
            ctrl_target = {**p.target, "population": ctrl_pool}
            ctrl_sets = [control_targets(model, ctrl_target, len(target_sets[0]), rng) for _ in range(n_draws)]
            # Keep the rates of the first draw only: in large fitted networks one set of
            # rates is hundreds of MB, so keeping every draw runs out of memory.
            pert_beh, rates = [], None
            for u in target_sets:
                b, r = run_condition(model, batch, *build_masks(p, u, batch, n), task=task)
                pert_beh.append(b)
                rates = r if rates is None else rates
                del r
            ctrl_runs = [run_condition(model, batch, *build_masks(p, u, batch, n), task=task)[0] for u in ctrl_sets]
            beh = average_behavior(pert_beh)
            ctrl = average_behavior(ctrl_runs)
            seed_res["perturbations"][p.name] = {
                "units": target_sets[0].tolist(), "n_target_draws": len(target_sets),
                "control_population": ctrl_pool, "n_control_draws": len(ctrl_sets),
                "perturbed": beh, "control": ctrl,
            }
            if example is None:  # only the first seed's rates feed the example plots
                rates_by_cond[p.name] = rates
            del rates
            log(f"  seed {seed}  {p.name:28s} acc intact {intact['accuracy']:.3f}  "
                f"perturbed {beh['accuracy']:.3f}  control {ctrl['accuracy']:.3f}")
        results["seeds"][str(seed)] = seed_res
        if example is None:
            example = {"seed": seed, "pca": pca_trajectories(rates_by_cond, batch, key=task.condition_key),
                       "rates": {k: v.mean(1).cpu().numpy().tolist() for k, v in rates_by_cond.items()},
                       "is_exc": model.is_exc.cpu().numpy().tolist()}

    results["summary"] = summarize(results, metrics=cfg.get("metrics"))
    results["example"] = example
    (out_dir / "results.json").write_text(json.dumps(results))
    return out_dir


def _get(d: dict, path: str) -> float:
    """Metric by path: "accuracy", "cr_rate", or "p_report_stim:3.0" for a field
    keyed by stimulus level (keys may be floats or their string form after JSON)."""
    for part in path.split(":"):
        if part in d:
            d = d[part]
        else:
            k = float(part)
            d = d[k] if k in d else d[str(k)]
    return float(d)


def summarize(results: dict, metrics: list | None = None) -> dict:
    """Per perturbation: mean +- sem accuracy drop across seeds, targeted vs control.

    `metrics` adds the same summary for other behavior fields, as changes
    (perturbed - intact, so a positive value means the metric went up)."""
    summary = {}
    for p in results["perturbations"]:
        name = p["name"]
        d_pert, d_ctrl = [], []
        for s in results["seeds"].values():
            base = s["intact"]["accuracy"]
            r = s["perturbations"][name]
            d_pert.append(base - r["perturbed"]["accuracy"])
            d_ctrl.append(base - r["control"]["accuracy"])
        d_pert, d_ctrl = np.array(d_pert), np.array(d_ctrl)
        diff = d_pert - d_ctrl
        k = len(diff)
        sem = lambda a: float(np.nanstd(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else float("nan")
        summary[name] = {
            "n_networks": k,
            "drop_perturbed_mean": float(d_pert.mean()), "drop_perturbed_sem": sem(d_pert),
            "drop_control_mean": float(d_ctrl.mean()), "drop_control_sem": sem(d_ctrl),
            "targeted_minus_control_mean": float(diff.mean()), "targeted_minus_control_sem": sem(diff),
            # Sign-flip permutation test on paired differences (exact for small k).
            "p_value": sign_flip_p(diff),
        }
        if metrics:
            summary[name]["metrics"] = {}
            for m in metrics:
                c_p, c_c = [], []
                for s in results["seeds"].values():
                    base = _get(s["intact"], m)
                    r = s["perturbations"][name]
                    c_p.append(_get(r["perturbed"], m) - base)
                    c_c.append(_get(r["control"], m) - base)
                c_p, c_c = np.array(c_p), np.array(c_c)
                summary[name]["metrics"][m] = {
                    "change_perturbed_mean": float(np.nanmean(c_p)), "change_perturbed_sem": sem(c_p),
                    "change_control_mean": float(np.nanmean(c_c)), "change_control_sem": sem(c_c),
                    "p_vs_intact": sign_flip_p(c_p), "p_vs_control": sign_flip_p(c_p - c_c),
                }
    return summary


def sign_flip_p(diff: np.ndarray, n_perm: int = 10000, seed: int = 0) -> float:
    k = len(diff)
    if k < 2:
        return float("nan")
    obs = abs(diff.mean())
    if k <= 14:
        signs = np.array(np.meshgrid(*[[-1, 1]] * k)).reshape(k, -1).T
    else:
        signs = np.random.default_rng(seed).choice([-1, 1], size=(n_perm, k))
    null = np.abs((signs * diff).mean(1))
    return float((null >= obs - 1e-12).mean())
