# rnn-simulation

An in-silico neuroscience workbench. Train continuous-time rate RNNs with separate
excitatory and inhibitory populations (Dale's law) on cognitive tasks, then run virtual
experiments on them: lesions, acute silencing, and optogenetic-style current injection
at chosen time windows, always compared against size-matched random controls across an
ensemble of networks.

## Quick start

```bash
pip install -e ".[test]"
python -m rnn_workbench train configs/train_pdm.yaml          # train 8 networks (~1 min each on CPU)
python -m rnn_workbench experiment configs/exp_first_look.yaml  # run perturbations, write report
open experiments/first_look/report.html
pytest -m "not slow"                                          # fast tests
```

## Tasks

| Task | What the network must do |
| --- | --- |
| `PerceptualDecisionMaking-v0` (NeuroGym) | Integrate two noisy evidence streams and report the stronger side after the stimulus ends |
| `NeedGatedCue-v0` (`envs.py`) | Livneh-style need-gated cue responding. A cue predicts food, water or nothing; tonic hunger and thirst inputs (standing in for AgRP / SFO drive) set the need state. After a delay the network must lick only if the cue matches the need: hungry+food, thirsty+water. Report shows a P(lick) table for every state x cue |

```bash
python -m rnn_workbench train configs/train_need.yaml
python -m rnn_workbench experiment configs/exp_need_first_look.yaml
```

Any NeuroGym task with fixation / stimulus / delay / decision periods works; add custom
tasks to `LOCAL_ENVS` in `envs.py`.

## How it fits together

| Module | What it does |
| --- | --- |
| `model.py` | `EIRNN`: continuous-time rate RNN, Dale's law by construction (80% E / 20% I by default), readout from E units only, hooks for silencing and current injection |
| `envs.py` | Custom NeuroGym tasks (`NeedGatedCue-v0`) |
| `tasks.py` | Task wrappers; controlled evaluation trials over each task's condition grid; accuracy, psychometric curves, lick tables |
| `train.py` | Trains one network per seed (BPTT, Adam, rate and weight L2) and saves checkpoints |
| `perturb.py` | Declarative perturbations: targets (E, I, all; fraction, count, explicit units, most selective), kinds (`silence`, `lesion`, `current`), time windows |
| `experiment.py` | Runs intact, targeted and matched-control conditions on every network; ensemble statistics with an exact sign-flip test |
| `report.py` | One self-contained HTML report per experiment |

## Writing an experiment

```yaml
name: my_experiment
model_dir: runs/pdm_ei
perturbations:
  - name: silence_I_late
    kind: silence              # silence | lesion | current
    target: {population: I, fraction: 0.3}
    window: [700, 1200]        # ms from trial start, or a period name like "stimulus"
  - name: excite_top_choice_E
    kind: current
    target: {population: E, count: 10, select: top_selective}
    window: stimulus
    strength: 0.5
```

Every perturbation is automatically paired with a control that hits the same number
of randomly chosen units from the same population, in the same window.

## Model

```
tau dx/dt = -x + W_rec r + W_in u + b + I_pert + noise,   r = relu(x),   z = W_out r_E
W_rec = relu(W_raw) * sign   (column j: +1 if unit j is excitatory, -1 if inhibitory)
```

## Example

[`docs/first_look_report.html`](docs/first_look_report.html) is the report from `configs/exp_first_look.yaml` on 8 networks (download and open in a browser).
[`docs/need_first_look_report.html`](docs/need_first_look_report.html) is the same for the need-gated cue task.
