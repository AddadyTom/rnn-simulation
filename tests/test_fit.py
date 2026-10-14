import numpy as np
import torch

from rnn_workbench.fit import FitConfig, condition_index, fit, fit_r2, pair_units, psth, record_teacher
from rnn_workbench.model import EIRNN, ModelConfig
from rnn_workbench.tasks import TaskConfig, eval_trials

NEED = TaskConfig(name="NeedGatedCue-v0", timing={"fixation": 300, "stimulus": 500, "delay": 500, "decision": 300})


def small_model(n=20):
    return EIRNN(ModelConfig(n_units=n, n_inputs=6, n_outputs=2))


def test_psth_averages_per_condition():
    batch = eval_trials(NEED, 2, seed=0)
    idx, keys = condition_index(batch)
    rates = torch.arange(batch.inputs.shape[1], dtype=torch.float32).view(1, -1, 1).expand(3, -1, 1)
    p = psth(rates, idx, len(keys))
    assert p.shape == (9, 3, 1)
    assert torch.allclose(p[0, 0, 0], torch.tensor(0.5))  # trials 0 and 1 are condition 0


def test_recording_pairs_types():
    torch.manual_seed(0)
    teacher = small_model()
    rec = record_teacher(teacher, NEED, 10, np.random.default_rng(0), n_per_condition=2)
    assert rec["psth"].shape[0] == 9 and rec["psth"].shape[-1] == 10
    assert rec["is_exc"].sum() == 8  # 80% E, like the teacher
    paired = pair_units(small_model(), rec["is_exc"])
    student = small_model()
    assert (student.is_exc[paired].numpy() == rec["is_exc"]).all()


def test_fit_r2_perfect_and_mean():
    t = np.random.default_rng(0).random((9, 5, 4))
    assert fit_r2(t, t)["r2"] == 1.0
    assert abs(fit_r2(np.full_like(t, t.mean()), t)["r2"]) < 1e-9


def test_fit_reduces_error(tmp_path):
    torch.manual_seed(0)
    teacher = small_model()
    rec = record_teacher(teacher, NEED, 10, np.random.default_rng(0), n_per_condition=4)
    np.savez(tmp_path / "rec.npz", psth=rec["psth"], is_exc=rec["is_exc"])
    cfg = FitConfig(name="t", task=NEED, model=ModelConfig(n_units=20), recordings=str(tmp_path / "rec.npz"),
                    steps=60, trials_per_condition=2, lr=1e-2, readout_steps=5, out_dir=str(tmp_path))
    out = fit(cfg, log=lambda *_: None)
    import json
    m = json.loads((out / "metrics.json").read_text())
    assert m["history"][-1]["fit_mse"] < m["history"][0]["fit_mse"] or len(m["history"]) == 1
    assert (out / "model.pt").exists()
