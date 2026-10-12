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

from .perturb import Perturbation, build_masks, choice_selectivity, control_targets, resolve_targets
from .tasks import TaskConfig, behavior_summary, choices_from_outputs, eval_trials
from .train import load_model


def run_condition(model, batch, silence=None, current=None, seed=0):
    torch.manual_seed(seed)  # identical noise across conditions
    with torch.no_grad():
        out, rates = model(batch.inputs, silence=silence, current=current)
    beh = behavior_summary(choices_from_outputs(out, batch), batch)
    return beh, rates


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
    out = {"accuracy": float(np.mean([r["accuracy"] for r in runs]))}
    if "psychometric" in runs[0]:
        out["psychometric"] = {
            "signed_coh": runs[0]["psychometric"]["signed_coh"],
            "p_choice_2": np.mean([r["psychometric"]["p_choice_2"] for r in runs], 0).tolist(),
        }
        out["accuracy_by_coh"] = {
            k: float(np.mean([r["accuracy_by_coh"][k] for r in runs])) for k in runs[0]["accuracy_by_coh"]
        }
    return out


def pca_trajectories(rates_by_cond: dict, batch, n_comp: int = 2) -> dict:
    """Project condition-averaged trajectories (by choice side and coherence) on the
    intact network's top PCs, so all conditions share axes."""
    keys = sorted({(c["ground_truth"], c.get("coh")) for c in batch.conditions})
    def averages(rates):
        r = rates.cpu().numpy()
        avg = []
        for gt, coh in keys:
            m = np.array([(c["ground_truth"], c.get("coh")) == (gt, coh) for c in batch.conditions])
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
    task = TaskConfig(**train_cfg["task"])
    seeds = cfg.get("seeds") or train_cfg["seeds"]
    perts = [Perturbation(**p) for p in cfg["perturbations"]]
    n_trials = int(cfg.get("trials_per_condition", 40))
    n_draws = int(cfg.get("control_draws", 10))
    batch = eval_trials(task, n_trials, seed=int(cfg.get("eval_seed", 4321)))
    out_dir = Path(cfg.get("out_dir", "experiments")) / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {"name": cfg["name"], "model_dir": str(run_dir), "task": train_cfg["task"],
               "model": train_cfg["model"], "perturbations": [asdict(p) for p in perts],
               "seeds": {}, "periods": batch.periods, "dt": batch.dt}
    example = None
    for seed in seeds:
        model = load_model(run_dir / f"seed_{seed}")
        rng = np.random.default_rng(seed)
        intact, intact_rates = run_condition(model, batch)
        sel = choice_selectivity(intact_rates, batch)
        seed_res = {"intact": intact, "perturbations": {}}
        rates_by_cond = {"intact": intact_rates}
        n = model.cfg.n_units
        for p in perts:
            # Random targets are redrawn n_draws times and averaged; selected targets are fixed.
            is_random = "units" not in p.target and p.target.get("select", "random") == "random"
            target_sets = [resolve_targets(model, p.target, rng, selectivity=sel)
                           for _ in range(n_draws if is_random else 1)]
            ctrl_pool = control_population(p.target, cfg)
            ctrl_target = {**p.target, "population": ctrl_pool}
            ctrl_sets = [control_targets(model, ctrl_target, len(target_sets[0]), rng) for _ in range(n_draws)]
            pert_runs = [run_condition(model, batch, *build_masks(p, u, batch, n)) for u in target_sets]
            ctrl_runs = [run_condition(model, batch, *build_masks(p, u, batch, n))[0] for u in ctrl_sets]
            beh, rates = average_behavior([b for b, _ in pert_runs]), pert_runs[0][1]
            ctrl = average_behavior(ctrl_runs)
            seed_res["perturbations"][p.name] = {
                "units": target_sets[0].tolist(), "n_target_draws": len(target_sets),
                "control_population": ctrl_pool, "n_control_draws": len(ctrl_sets),
                "perturbed": beh, "control": ctrl,
            }
            rates_by_cond[p.name] = rates
            log(f"  seed {seed}  {p.name:28s} acc intact {intact['accuracy']:.3f}  "
                f"perturbed {beh['accuracy']:.3f}  control {ctrl['accuracy']:.3f}")
        results["seeds"][str(seed)] = seed_res
        if example is None:
            example = {"seed": seed, "pca": pca_trajectories(rates_by_cond, batch),
                       "rates": {k: v.mean(1).cpu().numpy().tolist() for k, v in rates_by_cond.items()},
                       "is_exc": model.is_exc.cpu().numpy().tolist()}

    results["summary"] = summarize(results)
    results["example"] = example
    (out_dir / "results.json").write_text(json.dumps(results))
    return out_dir


def summarize(results: dict) -> dict:
    """Per perturbation: mean +- sem accuracy drop across seeds, targeted vs control."""
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
        sem = lambda a: float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else float("nan")
        summary[name] = {
            "n_networks": k,
            "drop_perturbed_mean": float(d_pert.mean()), "drop_perturbed_sem": sem(d_pert),
            "drop_control_mean": float(d_ctrl.mean()), "drop_control_sem": sem(d_ctrl),
            "targeted_minus_control_mean": float(diff.mean()), "targeted_minus_control_sem": sem(diff),
            # Sign-flip permutation test on paired differences (exact for small k).
            "p_value": sign_flip_p(diff),
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
