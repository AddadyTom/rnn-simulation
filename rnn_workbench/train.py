"""Train an ensemble of E/I RNNs on a task, one network per seed.

Works with NeuroGym tasks and with the interoceptive VagalOpto task. `init_from`
starts each seed from an existing ensemble's checkpoint of the same seed, which is
how staged protocols are run: train at 15 Hz, then continue on the psychophysics mix,
or train VagalOpto->NoReward, then retrain on the reversed contingency.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .model import EIRNN, ModelConfig
from .tasks import TaskConfig, get_task, task_config_from_dict


@dataclass
class TrainConfig:
    name: str = "pdm_ei"
    seeds: list = field(default_factory=lambda: [0, 1, 2])
    task: object = field(default_factory=TaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    steps: int = 1500
    batch_size: int = 32
    seq_len: int = 100  # NeuroGym only: length of the stitched trial sequence
    lr: float = 1e-3
    rate_l2: float = 1e-3  # penalty on mean squared firing rate
    weight_l2: float = 1e-4
    eval_every: int = 250
    eval_trials: int = 10  # per condition, during training
    target_accuracy: float = 0.0  # stop early once eval accuracy reaches this (0 = never)
    init_from: str = ""  # ensemble directory to start each seed from (same seed)
    save_every: int = 0  # also save checkpoints every N steps (0 = only at the end)
    out_dir: str = "runs"


def load_train_config(d: dict) -> TrainConfig:
    d = dict(d)
    task = task_config_from_dict(d.pop("task", {}))
    model = ModelConfig(**d.pop("model", {}))
    return TrainConfig(task=task, model=model, **d)


def evaluate(model: EIRNN, task_cfg, n_per_condition: int = 20, seed: int = 1234) -> dict:
    task = get_task(task_cfg)
    batch = task.eval_batch(n_per_condition, seed=seed)
    torch.manual_seed(seed)
    out, _ = task.run(model, batch)
    return task.behavior(out, batch)


def _save(model: EIRNN, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_cfg": asdict(model.cfg)}, path)


def train_one(cfg: TrainConfig, seed: int, out_dir: Path, log=print) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    task = get_task(cfg.task)
    n_in, n_out = task.dims()
    cfg.model.n_inputs, cfg.model.n_outputs = n_in, n_out
    cfg.model.dt = cfg.task.dt
    if cfg.init_from:
        model = load_model(Path(cfg.init_from) / f"seed_{seed}")
        model.train()
        if (model.cfg.n_inputs, model.cfg.n_outputs) != (n_in, n_out):
            raise ValueError("init_from checkpoint has different input/output sizes than the task")
        cfg.model = model.cfg
    else:
        model = EIRNN(cfg.model)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history = []
    t0 = time.time()
    for step in range(1, cfg.steps + 1):
        if hasattr(task, "_ds"):
            x, y, w = task.sample_batch(cfg.batch_size, rng, seq_len=cfg.seq_len)
        else:
            x, y, w = task.sample_batch(cfg.batch_size, rng)
        out, rates = model(x)
        ce = F.cross_entropy(out.reshape(-1, n_out), y.reshape(-1), reduction="none")
        w = w.reshape(-1)
        loss_task = (ce * w).sum() / w.sum()
        loss = loss_task + cfg.rate_l2 * rates.pow(2).mean() + cfg.weight_l2 * model.w_rec.pow(2).sum()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if cfg.save_every and step % cfg.save_every == 0:
            _save(model, out_dir / "checkpoints" / f"step_{step}.pt")
        if step % cfg.eval_every == 0 or step == cfg.steps:
            acc = evaluate(model, cfg.task, n_per_condition=cfg.eval_trials)["accuracy"]
            history.append({"step": step, "loss": loss_task.item(), "accuracy": acc})
            log(f"  seed {seed} step {step:5d}  loss {loss_task:.3f}  acc {acc:.3f}  ({time.time() - t0:.0f}s)")
            if cfg.target_accuracy and acc >= cfg.target_accuracy:
                break
    final = evaluate(model, cfg.task, n_per_condition=40)
    _save(model, out_dir / "model.pt")
    metrics = {"seed": seed, "history": history, "final": final, "train_seconds": time.time() - t0}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def train(cfg: TrainConfig, log=print) -> Path:
    root = Path(cfg.out_dir) / cfg.name
    root.mkdir(parents=True, exist_ok=True)
    (root / "train_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    summary = []
    for seed in cfg.seeds:
        log(f"Training seed {seed}")
        m = train_one(cfg, seed, root / f"seed_{seed}", log=log)
        summary.append({"seed": seed, "accuracy": m["final"]["accuracy"]})
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    return root


def load_model(seed_dir: Path, checkpoint: str = "model.pt") -> EIRNN:
    ckpt = torch.load(Path(seed_dir) / checkpoint, weights_only=False)
    model = EIRNN(ModelConfig(**ckpt["model_cfg"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
