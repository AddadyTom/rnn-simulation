import json

import numpy as np
import pytest
import torch

from rnn_workbench.calcium import deconvolve_ar1, psth_from_calcium, resample
from rnn_workbench.fit import fit_rank_sweep, record_teacher
from rnn_workbench.model import EIRNN, ModelConfig
from rnn_workbench.tasks import TaskConfig
from rnn_workbench.train import TrainConfig, load_model, train_one

NEED = TaskConfig(name="NeedGatedCue-v0", timing={"fixation": 300, "stimulus": 500, "delay": 500, "decision": 300})


@pytest.mark.parametrize("rank", [1, 3])
def test_lowrank_dale_signs_and_rank(rank):
    torch.manual_seed(0)
    m = EIRNN(ModelConfig(n_units=40, rank=rank))
    with torch.no_grad():  # push raw factors negative too: the constraint must still hold
        m.m_raw.normal_()
        m.n_raw.normal_()
    w = m.w_rec.detach()
    assert (w[:, m.is_exc] >= 0).all() and (w[:, ~m.is_exc] <= 0).all()
    assert torch.linalg.matrix_rank(w) <= 2 * rank
    assert not hasattr(m, "w_raw")


def test_lowrank_without_dale_has_rank_r():
    torch.manual_seed(0)
    m = EIRNN(ModelConfig(n_units=40, rank=2, dale=False))
    assert torch.linalg.matrix_rank(m.w_rec.detach()) == 2


def test_lowrank_checkpoint_roundtrip(tmp_path):
    cfg = TrainConfig(seeds=[0], steps=5, eval_every=5, model=ModelConfig(n_units=30, rank=2))
    train_one(cfg, 0, tmp_path / "s0", log=lambda *_: None)
    m = load_model(tmp_path / "s0")
    assert m.cfg.rank == 2 and torch.linalg.matrix_rank(m.w_rec.detach()) <= 4


def test_rank_sweep_writes_summary(tmp_path):
    torch.manual_seed(0)
    teacher = EIRNN(ModelConfig(n_units=20, n_inputs=6, n_outputs=2))
    rec = record_teacher(teacher, NEED, 10, np.random.default_rng(0), n_per_condition=2)
    np.savez(tmp_path / "rec.npz", psth=rec["psth"], is_exc=rec["is_exc"])
    d = {"name": "sweep", "task": NEED.__dict__, "model": {"n_units": 20}, "recordings": str(tmp_path / "rec.npz"),
         "steps": 5, "trials_per_condition": 1, "readout_steps": 2, "out_dir": str(tmp_path), "ranks": [1, None]}
    root = fit_rank_sweep(d, log=lambda *_: None)
    rows = json.loads((root / "rank_sweep.json").read_text())
    assert [r["rank"] for r in rows] == [1, None]
    assert (root / "rank_1" / "model.pt").exists() and (root / "full" / "model.pt").exists()


def test_ar1_deconvolution_recovers_spikes():
    fr, tau = 30.0, 0.5
    g = np.exp(-1 / (tau * fr))
    spikes = np.zeros((1, 200, 1))
    spikes[0, [20, 90, 150], 0] = 1.0
    f = np.zeros_like(spikes)
    for t in range(1, 200):
        f[0, t] = g * f[0, t - 1] + spikes[0, t]
    assert np.allclose(deconvolve_ar1(f, fr, tau), spikes)


def test_psth_from_calcium_shapes_and_order():
    rng = np.random.default_rng(0)
    traces = rng.random((6, 60, 3))  # 6 trials, 2 s at 30 Hz, 3 neurons
    traces[3:, :, 0] += 5  # neuron 0 is much more active in condition "b"
    conds = ["a"] * 3 + ["b"] * 3
    p = psth_from_calcium(traces, conds, ["b", "a"], frame_rate=30, dt_ms=20, n_steps=90, deconvolve="none")
    assert p.shape == (2, 90, 3)
    assert np.isclose(p.max(), 1.0) and p[0, :, 0].mean() > p[1, :, 0].mean()
    with pytest.raises(ValueError):
        resample(traces, 30, 20, 200)  # 4 s asked from a 2 s recording
