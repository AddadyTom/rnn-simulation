# Research plan: an E/I RNN of insula during interoceptive psychophysics

Source paper: Rafael, Shtiglitz, Miller, Prilutski, Talpir, Lavi & Livneh, *A cortical
basis for perception of internal mechanosensations*, bioRxiv 2026
(doi 10.64898/2026.02.11.705298, v2). Figure references below are to that paper. The
Methods and supplementary legends were not available when this plan was written, so
task timings marked *(assumed)* should be checked against them.

## 1. The task as network input

`rnn_workbench/vagal_task.py`, configs `configs/train_vagal_*.yaml`.

| Paper | Model |
| --- | --- |
| 4 s of red-light pulses on the neck at 0.5–15 Hz (ChRmine in NG-OxtR⁺ neurons) | input 0: pulse train at *f* Hz convolved with an exponential kernel (τ = 150 ms, *assumed*), normalised so 15 Hz has mean 1; per-trial lognormal gain (CV 0.3) for light absorption; white noise |
| Odor (same odor on every trial) | input 1: on during the response window |
| Sugar water when the mouse licks on a rewarded trial | input 2: on during the outcome period, closed loop on the network's own lick |
| Masking red light on every trial | no input (uninformative) |
| Lick / no lick | 2-class output; choice = mean lick probability in the response window > 0.5 |
| VagalOpto→NoReward / →Reward | `contingency: no_reward / reward` |
| Single-frequency task (15 Hz) | `train_frequencies: [15]` |
| Psychophysics (0.5–15 Hz interleaved, every stimulation trial unrewarded) | continue training with `train_frequencies` = all levels (`init_from`) |
| Timing *(assumed)*: baseline 1 s, stimulus 4 s, response 1.5 s, outcome 1 s | `timing` |
| stGtACR2 window: 1 s before stimulus to odor onset (5 s) | period `inhibition` |

First check (one network, seed 0, 300 + 300 training steps): after the 15 Hz stage the
threshold sits at about 5 Hz. After the psychophysics stage it drops to 1.2 Hz with a
sigmoid curve: P(report) ≈ 0.03 / 0.07 / 0.35 / 0.85 / 0.9 / 1.0 at 0 / 0.5 / 1 / 2 / 3 / 5 Hz.
The paper's mice sit at 1–3 Hz.

## 2. Data-constrained fits

`rnn_workbench/data.py` (loader), `rnn_workbench/fit.py` (training), `configs/fit_ss38.yaml`.

- Input files: suite2p output (`spks.npy` or `F.npy`/`Fneu.npy`, `iscell_final.npy`) plus an
  `events.csv` with one row per trial: `trial, stim_onset_frame, freq_hz,
  odor_onset_frame, licked, rewarded` (optional `inhibition`, `contingency`). Frames index
  the concatenated recording at 15 Hz.
- The first session is SS38, 26 Aug 2024 (`stav22`). It has two runs of 27,000 frames
  (1 h total), 1,201 ROIs and 763 cells in `iscell_final`. Its events file is still to come.
- Each recorded cell is tied to one excitatory unit. Layer 2/3 imaging does not identify
  cell type, so inhibitory units are hidden by default. Hidden excitatory units stand in
  for unrecorded neurons.
- Loss: MSE between `gain_i · rate_i + offset_i` and the cell's activity at frame times,
  plus a small-weight cross-entropy between the lick readout and the mouse's lick.
  The default `target: psth` fits condition averages (frequency × lick × reward) of
  the training trials, with noise-free inputs. `target: single_trial` fits individual
  trials. On synthetic data, single-trial fitting barely moved past the mean in 1,000
  steps, while condition averages fit quickly. Keep `behavior_weight` small (0.02):
  at 0.5 the lick loss dominated and activity R² stayed near 0.
- Check on a synthetic session (60 cells, 120 trials, 400 steps): held-out PSTH R² is
  0.20 and single-trial R² 0.17. The noise ceiling on the same test trials is 0.19 and
  0.21 (`fit.json` → `test.ceiling`). The ceiling is a condition-only model: each test
  trial predicted by its condition's mean over the training trials. So the fit is at
  the ceiling for averages and at 80% of it for single trials. The absolute numbers
  are low because the synthetic spikes are Poisson-noisy and each test condition has
  only a few trials.
- `python -m rnn_workbench mock-events data/stav22/events.csv` writes a made-up events
  file with SS38's layout (2 × 27,000 frames at 15 Hz, about 210 trials) to test the
  pipeline end to end before the real file arrives.
- Fitted networks are saved in the same format as task-trained ones, so every
  perturbation config runs on them unchanged.

## 3. Research questions

Each question names the paper result it builds on, the model experiment, and what would
count as an answer. Questions 1–4 can run now on task-trained networks; 5–8 need the
data-constrained fits.

**Q1. Can one circuit produce both inhibition results?** (Fig 3g–o)
In the paper, partial stGtACR2 inhibition (~47% of E cells, 5 s) *raised* "felt it"
reports, and only in the psychophysics task. Broad GABA-agonist inhibition *lowered*
them in both tasks, with hits unchanged.
Experiment: `configs/exp_vagal_inhibition.yaml` on the psychophysics and the
single-frequency ensembles. Sweep strength (−0.25 … −3) and fraction (20 / 47 / 100% of
E), and silence I units as well.
First result (one network, seed 0): partial E inhibition *lowers* reports (correct
rejections −67 points), the opposite of the paper. That is a real, useful mismatch.
A detector trained only to report stimulation does not reproduce the opto result.
Candidate fixes to test: (a) a lick-promoting "reward anticipation" population that the
stimulus suppresses (see Q4 and Fig 4); (b) noise or bias in the readout so that
pushing activity down shifts the criterion; (c) a two-area model (Q6).

**Q2. Where do sigmoidal and Gaussian cells come from?** (Fig 2e–i)
About 14.5% of cells are sigmoidal, with thresholds near behavior, and about 17.4% are
Gaussian, peaking at 1–3 Hz.
Experiment: `analysis.tuning_classes` on model units after each training stage. On a
synthetic session with planted cells (300 trials), it labels all 7 sigmoidal and all 7
Gaussian cells correctly. It also mislabels 2 of 3 stimulus-suppressed cells and 1 of
12 reward cells as Gaussian, so add a sign check before using it on real data. Ask
whether Gaussian tuning appears at all, and whether it needs the psychophysics stage.
Then test whether Gaussian units get more inhibitory input than sigmoidal units
(rows of `w_rec` from I units). Prediction: low-frequency preference arises when
inhibition grows faster with input than excitation does. Lesioning I units should turn
Gaussian units into sigmoidal ones.

**Q3. Does a choice signal emerge only in the hard task?** (Fig 3a–d)
Choice probability is significant in 15% of cells in the psychophysics task and 6% in
the single-frequency task (5% chance).
Experiment: `analysis.choice_probability` stratified by frequency, on stimulation trials
of each ensemble, plus `decode_accuracy` with `balance_groups=freq`. Match the number of
error trials between ensembles, because the paper's easy task has very few errors.
Answer: does CP rise with mixed-frequency training even when trial counts are matched?

**Q4. Information present vs information used.** (Fig 2j–k)
A population decoder beats the mouse two frequency steps below threshold.
Experiment: `analysis.neurometric` on hidden units vs the network's own psychometric
curve. Measure the angle between the decoder weights and the readout, the noise along
each, and decoding over time (the adaptation tension in Q5).
Answer: which of misalignment, readout noise, criterion or decay explains the gap.

**Q5. Adaptation vs integration.** (Fig 2, photometry; Fig S1 duration and delay tests)
Insula responses peak and fall to about 36% of peak by stimulus offset, yet behavior
improves with longer stimuli and decays over 2–8 s delays.
Experiment: test the task-trained networks with 1 s, 4-pulse and single-pulse stimuli,
and with 2–8 s delays (timing changes only). Check whether units ramp or adapt. Compare
with the data fit, which is constrained to adapt.
Answer: is integration local to the fitted insula units, or does the fit need hidden
units or a downstream area to integrate?

**Q6. One area or two?** (Fig 3m–o and the Discussion)
In a one-area model the lick readout comes from the same units that are inhibited. Broad
inhibition therefore removes licking altogether, which mice do not show (hits are
unchanged).
Experiment: add a second module ("downstream decision") that reads from the insula
module and sends feedback. Inhibit insula only. Then switch feedback off and test
whether choice probability in insula units survives (Q3).
Answer: is CP generated locally or fed back, and does the two-area model reproduce
both inhibition results?

**Q7. Is the stimulus code stable through reversal?** (Fig 1o–q, Fig 4c–f)
Insula keeps the same fraction of stimulus- and reward-responsive cells across
NoReward and Reward.
Experiment: `configs/train_vagal_reversal.yaml` saves checkpoints every 100 steps.
Track the same units' stimulus tuning, the input weights and the readout across
relearning. In the data fits, fit both contingencies with a shared network and
contingency-specific inputs if the lab has both sessions in one mouse.
Answer: does relearning change mostly the readout (paper's claim), or also the
recurrent code?

**Q8. Which cells did the stGtACR2 experiment actually move?** (Fig 3, in-silico perturbation)
The paper lowered the 20% most-suppressed cells *in a decoder* and pushed FA trials
toward CR, best at moderate strength. The real manipulation hit a random 47% of E cells.
Experiment: on the data-fitted network, (a) inhibit a random 47% of E units with
dynamics; (b) lower the "20% most suppressed on CR trials" units with dynamics;
(c) the paper's no-dynamics version. Compare the shifts in FA-trial decoding and in the
lick readout, as a function of strength.
Answer: does recurrence make random partial inhibition act mainly through the
suppressed population? This also gives a prediction for holographic inhibition of those
specific cells.

## 4. Open items before fitting SS38

1. `events.csv` for the session (or the rig's behavior file, and a converter).
2. Which task this session is (psychophysics or single frequency) and which contingency.
3. Exact timings from the Methods: response window length, inter-trial interval, and
   whether the odor overlaps the end of stimulation.
4. Whether this mouse also has inhibition trials. These are excluded from fitting by
   default and are the natural held-out test of the fitted model.
