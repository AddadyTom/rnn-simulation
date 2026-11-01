import json

import numpy as np
import pytest
import torch

from rnn_workbench import analysis
from rnn_workbench.data import load_session, trial_tensor, write_synthetic_session
from rnn_workbench.experiment import run_experiment
from rnn_workbench.fit import FitConfig, fit
from rnn_workbench.model import ModelConfig
from rnn_workbench.train import TrainConfig, load_model, train
from rnn_workbench.vagal_task import VagalOptoTask, VagalTaskConfig, behavior_summary, drive_trace, fit_threshold

FAST = dict(timing={"baseline": 200, "stimulus": 800, "delay": 0, "response": 200, "outcome": 200})


def test_drive_scales_with_frequency_and_decays():
    cfg = VagalTaskConfig()
    task = VagalOptoTask(cfg)
    s, e = task.periods["stimulus"]
    d15, d3, d0 = (drive_trace(f, cfg) for f in (15, 3, 0))
    assert abs(d15[s + 25:e].mean() - 1.0) < 0.05
    assert abs(d3[s + 25:e].mean() - 0.2) < 0.03
    assert d0.max() == 0 and d15[:s].max() == 0
    assert 0 < d15[e + 5] < d15[e - 1]  # kernel tail after stimulus offset


@pytest.mark.parametrize("contingency", ["no_reward", "reward"])
def test_targets_follow_contingency(contingency):
    task = VagalOptoTask(VagalTaskConfig(contingency=contingency, input_noise=0, gain_cv=0))
    x, y, w, conds = task.build([0.0, 15.0], np.random.default_rng(0))
    rs, re = task.periods["response"]
    os_, oe = task.periods["outcome"]
    go = [c["go"] for c in conds]
    assert go == ([True, False] if contingency == "no_reward" else [False, True])
    for b in range(2):
        assert (y[rs:re, b] == int(go[b])).all() and y[:rs, b].sum() == 0
        assert x[os_:oe, b, 2].max() == float(go[b])  # reward only on go trials
    assert (x[rs:re, :, 1] == 1).all() and x[:rs, :, 1].sum() == 0
    assert (w[rs:re] > 1).all()


def test_inhibition_window_is_1s_before_stimulus_to_odor():
    task = VagalOptoTask(VagalTaskConfig())
    assert task.periods["inhibition"] == (0, task.periods["response"][0])
    assert task.periods["stimulus"][0] - task.periods["inhibition"][0] == 1000 / 20


def test_behavior_summary_no_reward():
    conds = [{"freq": f, "go": f == 0} for f in [0, 0, 0, 0, 2, 2, 15, 15]]
    licked = np.array([1, 1, 1, 0, 1, 0, 0, 0])
    b = behavior_summary(licked, conds, "no_reward")
    assert b["hit_rate"] == 0.75 and b["cr_rate"] == 0.75
    assert b["p_report_stim"][2.0] == 0.5 and b["p_report_stim"][15.0] == 1.0
    assert b["accuracy"] == pytest.approx(6 / 8)
    assert b["dprime"][15.0] > b["dprime"][2.0]


def test_fit_threshold():
    lv = [0, 0.5, 1, 2, 3, 5, 15]
    p = [0.0, 0.02, 0.1, 0.5, 0.8, 0.97, 1.0]
    assert 1.6 < fit_threshold(lv, p) < 2.5


def small_vagal_cfg(tmp_path, **kw):
    task = VagalTaskConfig(**FAST, train_frequencies=[15.0], eval_frequencies=[3.0, 15.0])
    return TrainConfig(name="v", seeds=[0, 1], task=task, model=ModelConfig(n_units=30), steps=4,
                       batch_size=4, eval_every=4, eval_trials=2, out_dir=str(tmp_path), **kw)


def test_train_vagal_keeps_dale_and_init_from(tmp_path):
    root = train(small_vagal_cfg(tmp_path), log=lambda *_: None)
    m = load_model(root / "seed_0")
    w = m.w_rec.detach()
    assert (w[:, m.is_exc] >= 0).all() and (w[:, ~m.is_exc] <= 0).all()
    final = json.loads((root / "seed_0" / "metrics.json").read_text())["final"]
    assert {"accuracy", "p_report_stim", "psychometric"} <= set(final)
    cfg2 = small_vagal_cfg(tmp_path, init_from=str(root), save_every=2)
    cfg2.name, cfg2.task.contingency = "v_rev", "reward"
    root2 = train(cfg2, log=lambda *_: None)
    assert (root2 / "seed_0" / "checkpoints" / "step_2.pt").exists()


def test_experiment_on_vagal_ensemble(tmp_path):
    root = train(small_vagal_cfg(tmp_path), log=lambda *_: None)
    cfg = {"name": "inh", "model_dir": str(root), "trials_per_condition": 3, "control_draws": 2,
           "out_dir": str(tmp_path / "exp"), "metrics": ["cr_rate", "hit_rate", "p_report_stim:3.0"],
           "perturbations": [{"name": "opto", "kind": "current", "target": {"population": "E", "fraction": 0.47},
                              "window": "inhibition", "strength": -0.5}]}
    out = run_experiment(cfg, log=lambda *_: None)
    res = json.loads((out / "results.json").read_text())
    assert set(res["summary"]["opto"]["metrics"]) == {"cr_rate", "hit_rate", "p_report_stim:3.0"}
    from rnn_workbench.report import build_report
    assert build_report(out).exists()


def test_closed_loop_reward_matches_licks():
    from rnn_workbench.model import EIRNN
    torch.manual_seed(0)
    task = VagalOptoTask(VagalTaskConfig(**FAST, eval_frequencies=[15.0]))
    model = EIRNN(ModelConfig(n_units=20, n_inputs=3, n_outputs=2))
    with torch.no_grad():
        model.b_out[1] = 0.3  # lick on most trials so both cases occur
    batch = task.eval_batch(6, seed=0)
    seen = []
    orig = model.forward
    model.forward = lambda u, **kw: (seen.append(u), orig(u, **kw))[1]
    out, _ = task.run(model, batch)
    licked = task.choices(out, batch)
    go = np.array([c["go"] for c in batch.conditions])
    os_, oe = batch.periods["outcome"]
    got_reward = seen[-1][os_:oe, :, 2].amax(0).numpy() > 0
    assert (got_reward == (licked.astype(bool) & go)).all()


# -- data and fitting ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def session_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    write_synthetic_session(d, n_rois=20, n_trials=40, seed=0)
    return d


def test_load_session_filters_cells_and_aligns_trials(session_dir):
    s = load_session(session_dir)
    assert s.traces.shape[0] == 20 and len(s.cell_ids) == 20  # 5 non-cell ROIs dropped
    assert s.traces.min() >= 0
    act, ok = trial_tensor(s, 1000, 7500)
    assert act.shape == (40, int(7.5 * 15), 20) and ok.all()


def test_fit_runs_and_is_perturbable(session_dir, tmp_path):
    cfg = FitConfig(name="f", session_dir=str(session_dir), steps=3, eval_every=3, batch_size=4,
                    n_hidden_exc=5, out_dir=str(tmp_path))
    root = fit(cfg, log=lambda *_: None)
    r = json.loads((root / "seed_0" / "fit.json").read_text())
    assert r["n_recorded"] == 20 and len(set(r["unit_of_cell"])) == 20
    m = load_model(root / "seed_0")
    assert m.is_exc[r["unit_of_cell"]].all()
    exp = {"name": "e", "model_dir": str(root), "trials_per_condition": 2, "control_draws": 1,
           "out_dir": str(tmp_path / "exp"),
           "perturbations": [{"name": "s", "kind": "silence", "target": {"population": "E", "fraction": 0.5},
                              "window": "stimulus"}]}
    run_experiment(exp, log=lambda *_: None)


# -- analyses -------------------------------------------------------------------------------

def test_auroc():
    assert analysis.auroc([2, 3], [0, 1]) == 1.0
    assert analysis.auroc([0, 1], [2, 3]) == 0.0
    assert analysis.auroc([1, 1], [1, 1]) == 0.5


def test_choice_probability_finds_planted_unit():
    rng = np.random.default_rng(0)
    n = 200
    det = rng.random(n) < 0.5
    freqs = rng.choice([1.0, 2.0], n)
    resp = rng.normal(size=(n, 5))
    resp[:, 0] += 1.0 * det
    cp = analysis.choice_probability(resp, det, strata=freqs, n_perm=200)
    assert cp["cp"][0] > 0.7 and cp["p"][0] < 0.01 and (cp["p"][1:] > 0.01).mean() >= 0.75


def test_tuning_classes_separates_sigmoid_and_gaussian():
    rng = np.random.default_rng(0)
    levels = np.array([0, 0.5, 1, 2, 3, 5, 8, 15.0])
    freqs = np.repeat(levels, 20)
    lf = np.log10(np.where(freqs > 0, freqs, 0.25))
    sig = 1 / (1 + np.exp(-6 * (lf - np.log10(3))))
    gau = np.exp(-0.5 * ((lf - np.log10(1.5)) / 0.2) ** 2)
    resp = np.stack([sig, gau, np.zeros_like(sig)], 1) + 0.1 * rng.normal(size=(len(freqs), 3))
    tc = analysis.tuning_classes(resp, freqs, n_shuffle=30)
    assert [t["kind"] for t in tc] == ["sigmoid", "gaussian", "none"]
    assert 2 < tc[0]["threshold_hz"] < 4.5 and 1 < tc[1]["peak_hz"] < 2.2


def test_decoding_and_neurometric():
    rng = np.random.default_rng(0)
    freqs = np.repeat([0.0, 1.0, 15.0], 40)
    resp = rng.normal(size=(120, 10))
    resp[freqs == 15.0, :3] += 3
    nm = analysis.neurometric(resp, freqs)
    assert nm[15.0] > 0.9 and 0.3 < nm[1.0] < 0.75
