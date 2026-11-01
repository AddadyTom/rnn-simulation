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

## Interoceptive psychophysics (insula)

A model of the task in Rafael, Shtiglitz, … & Livneh (bioRxiv 2026): mice report
optogenetic activation of gut-stretch vagal afferents at 0.5–15 Hz. The plan,
including eight research questions, is in [`docs/vagal_research_plan.md`](docs/vagal_research_plan.md).

```bash
python -m rnn_workbench train configs/train_vagal_single.yaml          # 15 Hz detection (VagalOpto->NoReward)
python -m rnn_workbench train configs/train_vagal_psychophysics.yaml   # continue on 0.5-15 Hz, from the 15 Hz nets
python -m rnn_workbench train configs/train_vagal_reversal.yaml        # reversal to VagalOpto->Reward, with checkpoints
python -m rnn_workbench experiment configs/exp_vagal_inhibition.yaml   # partial E inhibition, drug-like silencing
python -m rnn_workbench fit configs/fit_ss38.yaml                      # data-constrained fit to a suite2p session
```

The fit needs the session's suite2p output plus an `events.csv`, one row per trial
(the format is in `rnn_workbench/data.py`). `data.write_synthetic_session` writes a fake
session in the same format for testing.

## How it fits together

| Module | What it does |
| --- | --- |
| `model.py` | `EIRNN`: continuous-time rate RNN, Dale's law by construction (80% E / 20% I by default), readout from E units only, hooks for silencing and current injection |
| `tasks.py` | NeuroGym task wrappers; controlled evaluation trials; accuracy and psychometric curves; the task adapter interface |
| `vagal_task.py` | The interoceptive detection and psychophysics task: vagal pulse-train input, odor, closed-loop reward; paper-style metrics (correct rejections, hits, d′, threshold) |
| `data.py` | suite2p session loader, `events.csv` format, trial alignment, synthetic sessions |
| `fit.py` | Data-constrained training: recorded cells mapped to E units, held-out R², output perturbable like any ensemble |
| `analysis.py` | Sigmoid vs Gaussian tuning, choice probability, SVM decoding and neurometric curves, for model units or recorded cells |
| `train.py` | Trains one network per seed (BPTT, Adam, rate and weight L2) and saves checkpoints |
| `perturb.py` | Declarative perturbations: targets (E, I, all; fraction, count, explicit units, most choice-selective), kinds (`silence`, `lesion`, `current`), time windows |
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
    target: {population: E, count: 10, select: top_choice}
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
