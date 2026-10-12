import numpy as np
import pytest

from rnn_workbench.envs import NeedGatedCue
from rnn_workbench.tasks import (
    TaskConfig, behavior_summary, choices_from_outputs, condition_key, eval_trials, task_dims,
)

NEED = TaskConfig(name="NeedGatedCue-v0", timing={"fixation": 300, "stimulus": 500, "delay": 500, "decision": 300})


@pytest.fixture(scope="module")
def batch():
    return eval_trials(NEED, n_per_condition=3, seed=0)


def test_dims():
    assert task_dims(NEED) == (6, 2)


@pytest.mark.parametrize("cue,hunger,thirst,lick", [
    ("food", 0.8, 0.1, 1), ("food", 0.1, 0.8, 0), ("food", 0.1, 0.1, 0),
    ("water", 0.1, 0.8, 1), ("water", 0.8, 0.1, 0), ("water", 0.1, 0.1, 0),
    ("neutral", 0.8, 0.8, 0),
])
def test_lick_rule(cue, hunger, thirst, lick):
    assert NeedGatedCue().lick_target(cue, hunger, thirst) == lick


def test_grid_targets_and_inputs(batch):
    assert len(batch.conditions) == 9 * 3 and batch.zero_is_response
    go = {("hungry", "food"), ("thirsty", "water")}
    for i, c in enumerate(batch.conditions):
        assert c["target_action"] == int((c["state"], c["cue"]) in go)
    s, e = batch.periods["stimulus"]
    d0, d1 = batch.periods["decision"]
    x = batch.inputs.numpy()
    assert (x[:d0, :, 0] == 1).all() and (x[d0:, :, 0] == 0).all()  # fixation off at decision
    assert np.abs(x[:s, :, 1:4]).max() == 0 and np.abs(x[e:, :, 1:4]).max() == 0  # cue only in stimulus
    assert (x[:, :, 4:].mean(0) > 0).all()  # need inputs tonic over the trial


def test_lick_summary_from_perfect_outputs(batch):
    import torch
    out = torch.zeros(batch.inputs.shape[0], len(batch.conditions), 2)
    target = torch.tensor([c["target_action"] for c in batch.conditions])
    out[:, torch.arange(len(target)), target] = 1.0
    beh = behavior_summary(choices_from_outputs(out, batch), batch)
    assert beh["accuracy"] == 1.0
    p = np.array(beh["lick"]["p_lick"])
    assert beh["lick"]["states"] == ["hungry", "thirsty", "sated"]
    assert np.array_equal(p, [[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    assert condition_key(batch.conditions[0]) == ("hungry", "food")
