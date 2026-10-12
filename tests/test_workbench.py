import numpy as np
import pytest
import torch

from rnn_workbench.experiment import sign_flip_p
from rnn_workbench.model import EIRNN, ModelConfig
from rnn_workbench.perturb import Perturbation, build_masks, control_targets, resolve_targets
from rnn_workbench.tasks import TaskConfig, eval_trials
from rnn_workbench.train import TrainConfig, train_one


@pytest.fixture(scope="module")
def batch():
    return eval_trials(TaskConfig(), n_per_condition=2, seed=0)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return EIRNN(ModelConfig(n_units=50, frac_exc=0.8, n_inputs=3, n_outputs=3))


def check_dale(m: EIRNN):
    w = m.w_rec.detach()
    exc = m.is_exc
    assert (w[:, exc] >= 0).all(), "excitatory columns must be non-negative"
    assert (w[:, ~exc] <= 0).all(), "inhibitory columns must be non-positive"
    assert torch.diagonal(w).abs().max() == 0, "no self-connections"
    assert (m.w_in.detach() >= 0).all()


def test_dale_at_init(model):
    assert model.n_exc == 40
    check_dale(model)


def test_dale_holds_after_training(tmp_path):
    cfg = TrainConfig(seeds=[0], steps=30, eval_every=30, model=ModelConfig(n_units=40))
    train_one(cfg, 0, tmp_path / "s0", log=lambda *_: None)
    from rnn_workbench.train import load_model
    check_dale(load_model(tmp_path / "s0"))


def test_silence_zeroes_only_targets_in_window(model, batch):
    pert = Perturbation(kind="silence", target={"units": [1, 2, 45]}, window="stimulus")
    units = resolve_targets(model, pert.target, np.random.default_rng(0))
    silence, current = build_masks(pert, units, batch, model.cfg.n_units)
    assert current is None
    torch.manual_seed(0)
    _, r_int = model(batch.inputs, noise=False)
    _, r_pert = model(batch.inputs, silence=silence, noise=False)
    s, e = batch.periods["stimulus"]
    assert r_pert[s:e][:, :, units].abs().max() == 0
    # Before the window nothing differs.
    assert torch.allclose(r_int[:s], r_pert[:s])
    # Silencing during the window changes the rest of the network.
    assert not torch.allclose(r_int[s:e], r_pert[s:e])


def test_lesion_covers_whole_trial(model, batch):
    pert = Perturbation(kind="lesion", target={"units": [3]}, window="stimulus")
    silence, _ = build_masks(pert, np.array([3]), batch, model.cfg.n_units)
    assert silence[:, :, 3].all() and not silence[:, :, 4].any()


def test_current_injection(model, batch):
    pert = Perturbation(kind="current", target={"units": [5]}, window=[200, 400], strength=2.0)
    _, current = build_masks(pert, np.array([5]), batch, model.cfg.n_units)
    assert current[10:20, :, 5].eq(2.0).all() and current[:10].abs().sum() == 0


def test_population_targets_and_controls(model):
    rng = np.random.default_rng(0)
    inh = resolve_targets(model, {"population": "I", "fraction": 0.5}, rng)
    assert len(inh) == 5 and not model.is_exc[inh].any()
    ctrl = control_targets(model, {"population": "E"}, 7, rng)
    assert len(ctrl) == 7 and model.is_exc[ctrl].all()


def test_sign_flip_p():
    assert sign_flip_p(np.array([0.1] * 8)) == pytest.approx(2 / 256)
    assert sign_flip_p(np.array([0.1, -0.1, 0.1, -0.1])) == 1.0


@pytest.mark.slow
def test_network_learns_task(tmp_path):
    cfg = TrainConfig(seeds=[0], steps=800, eval_every=400)
    m = train_one(cfg, 0, tmp_path / "s0", log=lambda *_: None)
    assert m["final"]["accuracy_by_coh"][51.2] > 0.95
