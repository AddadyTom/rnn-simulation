"""Train an ensemble of E/I RNNs on a NeuroGym task, one network per seed."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .model import EIRNN, ModelConfig
from .tasks import TaskConfig, behavior_summary, choices_from_outputs, eval_trials, make_dataset, task_dims


@dataclass
class TrainConfig:
    name: str = "pdm_ei"
    seeds: list = field(default_factory=lambda: [0, 1, 2])
    task: TaskConfig = field(default_factory=TaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    steps: int = 1500
    batch_size: int = 32
    seq_len: int = 100
    lr: float = 1e-3
    rate_l2: float = 1e-3  # penalty on mean squared firing rate
    weight_l2: float = 1e-4
    eval_every: int = 250
    target_accuracy: float = 0.0  # stop early once eval accuracy reaches this (0 = never)
    out_dir: str = "runs"


def load_train_config(d: dict) -> TrainConfig:
    d = dict(d)
    task = TaskConfig(**d.pop("task", {}))
    model = ModelConfig(**d.pop("model", {}))
    return TrainConfig(task=task, model=model, **d)


def evaluate(model: EIRNN, task: TaskConfig, n_per_condition: int = 20, seed: int = 1234) -> dict:
    batch = eval_trials(task, n_per_condition, seed=seed)
    with torch.no_grad():
        out, _ = model(batch.inputs)
    return behavior_summary(choices_from_outputs(out, batch), batch)


def train_one(cfg: TrainConfig, seed: int, out_dir: Path, log=print) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_in, n_out = task_dims(cfg.task)
    cfg.model.n_inputs, cfg.model.n_outputs = n_in, n_out
    cfg.model.dt = cfg.task.dt
    model = EIRNN(cfg.model)
    ds = make_dataset(cfg.task, cfg.batch_size, cfg.seq_len, seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    ce = nn.CrossEntropyLoss()
    history = []
    t0 = time.time()
    for step in range(1, cfg.steps + 1):
        x, y = ds()
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        out, rates = model(x)
        loss_task = ce(out.reshape(-1, n_out), y.reshape(-1))
        loss = loss_task + cfg.rate_l2 * rates.pow(2).mean() + cfg.weight_l2 * model.w_rec.pow(2).sum()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % cfg.eval_every == 0 or step == cfg.steps:
            acc = evaluate(model, cfg.task, n_per_condition=10)["accuracy"]
            history.append({"step": step, "loss": float(loss_task), "accuracy": acc})
            log(f"  seed {seed} step {step:5d}  loss {loss_task:.3f}  acc {acc:.3f}  ({time.time() - t0:.0f}s)")
            if cfg.target_accuracy and acc >= cfg.target_accuracy:
                break
    final = evaluate(model, cfg.task, n_per_condition=40)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_cfg": asdict(cfg.model)}, out_dir / "model.pt")
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


def load_model(seed_dir: Path) -> EIRNN:
    ckpt = torch.load(Path(seed_dir) / "model.pt", weights_only=False)
    model = EIRNN(ModelConfig(**ckpt["model_cfg"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
