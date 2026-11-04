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


def test_select_most_suppressed_and_excited(model):
    rng = np.random.default_rng(0)
    response = np.arange(50, dtype=float) - 20.0  # unit 0 drops most, unit 39 (last E) rises most
    sup = resolve_targets(model, {"population": "E", "count": 3, "select": "most_suppressed"}, rng, response=response)
    exc = resolve_targets(model, {"population": "E", "count": 3, "select": "most_excited"}, rng, response=response)
    assert sup.tolist() == [0, 1, 2] and exc.tolist() == [37, 38, 39]


def test_silenced_units_do_not_build_up_input(model):
    """A silenced unit is held down, so right after silencing ends it starts from rest
    instead of from the input it received while silenced."""
    T, B, n = 30, 2, model.cfg.n_units
    u = torch.ones(T, B, 3)
    silence = torch.zeros(T, B, n, dtype=torch.bool)
    silence[:20] = True
    _, rates = model(u, silence=silence, noise=False)
    _, fresh = model(u[20:], noise=False)
    assert rates[:20].abs().max() == 0
    assert torch.allclose(rates[20], fresh[0], atol=1e-5)


def test_two_areas_connect_only_through_e_units():
    area = [0] * 6 + [1] * 2 + [0] * 1 + [1] * 1  # 8 E (6 + 2), then 2 I (one per area)
    m = EIRNN(ModelConfig(n_units=10, frac_exc=0.8, n_inputs=3, n_outputs=2, area_of_unit=area,
                          input_areas=[[0], [0, 1], [0, 1]], readout_area=1, feedback=False))
    w, a = m.w_rec.detach(), torch.tensor(area)
    assert w[a == 0][:, a == 1].abs().sum() == 0  # no feedback
    assert w[6:8][:, :6].abs().sum() > 0  # insula E feeds area 1
    assert w[a == 1][:, 8].abs().sum() == 0  # the insula I unit stays local
    assert m.w_in.detach()[a == 1, 0].abs().sum() == 0  # stimulus reaches the insula only
    assert (m.out_mask == torch.tensor([0.0] * 6 + [1.0] * 2)).all()


def test_sign_flip_p():
    assert sign_flip_p(np.array([0.1] * 8)) == pytest.approx(2 / 256)
    assert sign_flip_p(np.array([0.1, -0.1, 0.1, -0.1])) == 1.0


@pytest.mark.slow
def test_network_learns_task(tmp_path):
    cfg = TrainConfig(seeds=[0], steps=800, eval_every=400)
    m = train_one(cfg, 0, tmp_path / "s0", log=lambda *_: None)
    assert m["final"]["accuracy_by_coh"][51.2] > 0.95
