"""Interoceptive psychophysics task (Rafael, Shtiglitz, ... Livneh, bioRxiv 2026).

A head-fixed mouse gets non-invasive optogenetic activation of gut-stretch vagal
afferents (NG-OxtR+ neurons, ChRmine, red light on the neck) for 4 s, then an odor.
Licking during the odor is rewarded with sugar water or not, depending on whether the
gut stimulus came first:

    contingency "no_reward"  (VagalOpto->NoReward):  stim -> withhold licking (correct rejection)
                                                     no stim -> lick for reward (hit)
    contingency "reward"     (VagalOpto->Reward):    stim -> lick for reward
                                                     no stim -> withhold

Stimulus strength is the pulse frequency (0.5-15 Hz) at fixed duration, because the
opsin's slow kinetics sum pulses. "Single-frequency" training uses one frequency
(15 Hz in the paper's imaging comparison); "psychophysics" mixes frequencies.

Inputs to the network (all non-negative, one column each):

    0  vagal   afferent drive: the pulse train filtered by an exponential kernel
               (opsin + afferent summation), normalised so 15 Hz gives a mean of 1,
               times a per-trial gain (light absorption varies) plus white noise
    1  odor    1 during the odor period (or, when that period has length 0, during
               the response window), 0 otherwise
    2  reward  1 on trials where the network licked and the contingency rewards
               licking (closed loop at evaluation; assumed correct during training).
               By default it is on for the whole outcome period. With `reward_delay`
               set, it starts that many ms into the response window and lasts
               `reward_ms` (the valve opens on the first lick in the window).

The masking red light is on in every trial, so it carries no information and has no
input channel.

Output: two classes per time step, 0 = no lick, 1 = lick. The training target is
"lick" during the odor and response periods on go trials and "no lick" everywhere
else (mice do not lick during the stimulus). Odor steps have weight `odor_weight`,
response steps `response_weight`. The choice on a trial is "lick" when the mean lick
probability over the response window exceeds 0.5.

Trial timing (ms, configurable):  baseline 1000 | stimulus 4000 | delay 0 | odor 0 |
response 1500 | outcome 1000. The lab's recorded sessions (e.g. OI151) use odor 2000
and response 2000 instead: light at 0 s, opto 1-5 s, odor 5-7 s, reward window 7-9 s
(configs/train_vagal_oi151.yaml). A 5 s inhibition window "1 s before the stimulus to
the odor", as in the paper's stGtACR2 experiment, is the window from 1 s before the
stimulus to odor onset (period name "inhibition").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .tasks import TrialBatch

PERIODS = ("baseline", "stimulus", "delay", "odor", "response", "outcome")
INPUTS = ("vagal", "odor", "reward")
F_REF = 15.0  # Hz; the trained frequency, drive normalised to mean 1 here


@dataclass
class VagalTaskConfig:
    name: str = "VagalOptoDetection"
    dt: float = 20.0
    timing: dict = field(
        default_factory=lambda: {"baseline": 1000, "stimulus": 4000, "delay": 0, "response": 1500, "outcome": 1000}
    )
    contingency: str = "no_reward"  # "no_reward" | "reward"
    train_frequencies: list = field(default_factory=lambda: [15.0])
    eval_frequencies: list = field(default_factory=lambda: [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 15.0])
    p_stim: float = 0.5  # fraction of stimulation trials
    opsin_tau: float = 150.0  # ms, decay of the summed pulse response
    saturation: float = 0.0  # 0 = linear drive; k > 0 gives d * (1 + k) / (d + k)
    gain_cv: float = 0.3  # per-trial lognormal gain on the vagal drive
    input_noise: float = 0.5  # white noise on the vagal channel, per sqrt(step) at dt=20 ms
    response_weight: float = 5.0  # loss weight of response-window steps relative to others
    odor_weight: float = 1.0  # loss weight of odor-period steps (licking is allowed but optional there)
    reward_delay: float | None = None  # ms into the response window when reward starts; None = outcome period
    reward_ms: float = 1000.0  # reward input duration when reward_delay is set
    kwargs: dict = field(default_factory=dict)  # unused; keeps the TaskConfig shape


def steps_per_period(cfg: VagalTaskConfig) -> dict[str, tuple[int, int]]:
    out, t = {}, 0
    for p in PERIODS:
        n = int(round(cfg.timing.get(p, 0) / cfg.dt))
        out[p] = (t, t + n)
        t += n
    s_inh = max(0, out["stimulus"][0] - int(round(1000 / cfg.dt)))
    odor_start = out["odor"][0] if out["odor"][1] > out["odor"][0] else out["response"][0]
    out["inhibition"] = (s_inh, odor_start)
    return out


def odor_window(per: dict) -> tuple[int, int]:
    """Steps with the odor input on: the odor period, or the response window if there is none."""
    return per["odor"] if per["odor"][1] > per["odor"][0] else per["response"]


def reward_window(cfg: VagalTaskConfig, per: dict, T: int) -> tuple[int, int]:
    """Steps with the reward input on, for a rewarded trial."""
    if cfg.reward_delay is None:
        return per["outcome"]
    s = per["response"][0] + int(round(cfg.reward_delay / cfg.dt))
    return min(s, T), min(s + int(round(cfg.reward_ms / cfg.dt)), T)


def choice_window(cfg: VagalTaskConfig, per: dict, T: int) -> tuple[int, int]:
    """Steps where the lick choice is read. With reward inside the response window
    (reward_delay set), the choice is the licking before the reward arrives, from odor
    onset to reward onset, as the mouse's first lick opens the valve. Otherwise the
    response window."""
    if cfg.reward_delay is None:
        return per["response"]
    return odor_window(per)[0], reward_window(cfg, per, T)[0]


def n_steps(cfg: VagalTaskConfig) -> int:
    return steps_per_period(cfg)["outcome"][1]


def pulse_drive(freq: float, cfg: VagalTaskConfig, n_stim_steps: int) -> np.ndarray:
    """Mean-normalised afferent drive for a pulse train at `freq` Hz (noise-free).

    Pulses are delta functions at t_k = k / freq, binned at dt and convolved with an
    exponential kernel of time constant opsin_tau. The kernel has unit area per ms, so
    the steady-state mean is freq (pulses per ms * 1000), divided by F_REF.
    """
    if freq <= 0 or n_stim_steps == 0:
        return np.zeros(n_stim_steps)
    dur_ms = n_stim_steps * cfg.dt
    pulses = np.zeros(n_stim_steps)
    times = np.arange(0.0, dur_ms, 1000.0 / freq)
    np.add.at(pulses, np.minimum((times / cfg.dt).astype(int), n_stim_steps - 1), 1.0)
    decay = np.exp(-cfg.dt / cfg.opsin_tau)
    out = np.zeros(n_stim_steps)
    acc = 0.0
    for t in range(n_stim_steps):
        acc = acc * decay + pulses[t]
        out[t] = acc
    out *= (1 - decay) * 1000.0 / cfg.dt / F_REF  # unit-area kernel, then normalise
    if cfg.saturation > 0:
        k = cfg.saturation
        out = out * (1 + k) / (out + k)
    return out


def drive_trace(freq: float, cfg: VagalTaskConfig, T: int | None = None) -> np.ndarray:
    """Noise-free vagal drive over a whole trial. The drive keeps decaying after the
    stimulus ends (the kernel tail), which matters for delay experiments."""
    per = steps_per_period(cfg)
    T = T or n_steps(cfg)
    s, e = per["stimulus"]
    d = np.zeros(T)
    d[s:e] = pulse_drive(freq, cfg, e - s)
    if freq > 0 and e < T:
        decay = np.exp(-cfg.dt / cfg.opsin_tau)
        d[e:] = d[e - 1] * decay ** np.arange(1, T - e + 1)
    return d


def is_go(stim: bool, contingency: str) -> bool:
    if contingency == "no_reward":
        return not stim
    if contingency == "reward":
        return stim
    raise ValueError(f"unknown contingency {contingency!r}")


class VagalOptoTask:
    """Task adapter used by train.py and experiment.py."""

    def __init__(self, cfg: VagalTaskConfig):
        self.cfg = cfg
        self.periods = steps_per_period(cfg)
        self.T = n_steps(cfg)

    # -- shapes ---------------------------------------------------------------
    def dims(self) -> tuple[int, int]:
        return len(INPUTS), 2

    # -- trial construction -----------------------------------------------------
    def build(self, freqs: list[float], rng: np.random.Generator, licked: np.ndarray | None = None):
        """Inputs, targets and loss weights for trials with the given frequencies
        (0 = no stimulation). `licked` sets the reward input closed-loop; by default
        the network is assumed to choose correctly."""
        cfg, per, T, B = self.cfg, self.periods, self.T, len(freqs)
        x = np.zeros((T, B, len(INPUTS)), dtype=np.float32)
        y = np.zeros((T, B), dtype=np.int64)
        w = np.ones((T, B), dtype=np.float32)
        rs, re = per["response"]
        ds, de = per["odor"]
        os_, oe = odor_window(per)
        qs, qe = reward_window(cfg, per, T)
        w[ds:de] = cfg.odor_weight
        w[rs:re] = cfg.response_weight
        noise_scale = cfg.input_noise * (20.0 / cfg.dt) ** 0.5
        conds = []
        for b, f in enumerate(freqs):
            stim = f > 0
            go = is_go(stim, cfg.contingency)
            gain = rng.lognormal(-0.5 * np.log1p(cfg.gain_cv**2), np.sqrt(np.log1p(cfg.gain_cv**2))) if cfg.gain_cv > 0 else 1.0
            drive = gain * drive_trace(f, cfg, T)
            x[:, b, 0] = np.maximum(drive + noise_scale * rng.standard_normal(T) * (noise_scale > 0), 0.0)
            x[os_:oe, b, 1] = 1.0
            lick = go if licked is None else bool(licked[b])
            if lick and go:
                x[qs:qe, b, 2] = 1.0
            if go:
                y[ds:re, b] = 1
            conds.append({"freq": float(f), "stim": bool(stim), "go": bool(go), "ground_truth": int(stim),
                          "gain": float(gain)})
        return x, y, w, conds

    def sample_batch(self, batch_size: int, rng: np.random.Generator):
        stim = rng.random(batch_size) < self.cfg.p_stim
        freqs = np.where(stim, rng.choice(self.cfg.train_frequencies, size=batch_size), 0.0)
        x, y, w, _ = self.build(list(freqs), rng)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(w)

    def eval_batch(self, n_per_condition: int, seed: int = 0, freqs: list[float] | None = None) -> TrialBatch:
        rng = np.random.default_rng(seed)
        levels = [0.0] + [float(f) for f in (freqs or self.cfg.eval_frequencies)]
        fl = [f for f in levels for _ in range(n_per_condition)]
        x, y, _, conds = self.build(fl, rng)
        batch = TrialBatch(inputs=torch.from_numpy(x), targets=torch.from_numpy(y), conditions=conds,
                           periods=dict(self.periods), dt=self.cfg.dt)
        return batch

    # -- running and scoring ------------------------------------------------------
    def choices(self, outputs: torch.Tensor, batch: TrialBatch) -> np.ndarray:
        """1 = licked, 0 = withheld (mean lick probability over the choice window > 0.5)."""
        s, e = choice_window(self.cfg, batch.periods, outputs.shape[0])
        p_lick = torch.softmax(outputs[s:e], -1)[..., 1].mean(0)
        return (p_lick > 0.5).cpu().numpy().astype(int)

    def run(self, model, batch: TrialBatch, silence=None, current=None, noise=True):
        """Closed loop: run to get the choice, then set the reward input from the
        network's own licks and run again. The choice is taken from the first pass,
        which gets no reward input, so a perturbed network cannot read its answer from
        a reward it has not earned yet.
        With the default timing reward arrives after the response window, so the
        second pass gives the same choice when the noise is reseeded; with
        `reward_delay` set, reward lands inside the window and only later steps change."""
        state = torch.random.get_rng_state()
        x0 = batch.inputs.clone()
        x0[:, :, 2] = 0.0  # the choice is made before any reward
        with torch.no_grad():
            out, _ = model(x0, silence=silence, current=current, noise=noise)
        licked = self.choices(out, batch)
        x = batch.inputs.clone()
        qs, qe = reward_window(self.cfg, batch.periods, x.shape[0])
        x[:, :, 2] = 0.0
        for b, c in enumerate(batch.conditions):
            if licked[b] and c["go"]:
                x[qs:qe, b, 2] = 1.0
        torch.random.set_rng_state(state)
        with torch.no_grad():
            out, rates = model(x, silence=silence, current=current, noise=noise)
        return out, rates

    def behavior(self, outputs: torch.Tensor, batch: TrialBatch) -> dict:
        return behavior_summary(self.choices(outputs, batch), batch.conditions, self.cfg.contingency)

    def condition_key(self, c: dict):
        return (c["ground_truth"], c["freq"])


# -- behavior -----------------------------------------------------------------------

def _z(p: float, n: int) -> float:
    from scipy.stats import norm
    p = min(max(p, 0.5 / n), 1 - 0.5 / n)  # standard 1/(2N) correction for 0 and 1
    return float(norm.ppf(p))


def behavior_summary(licked: np.ndarray, conds: list[dict], contingency: str) -> dict:
    """Paper-style metrics.

    "Reported stim" = withheld licking (no_reward) or licked (reward). For each
    frequency: P(report stim | freq), and d' and criterion against no-stim trials.
    In the no_reward contingency, P(report stim | stim) is the correct-rejection rate
    and P(lick | no stim) the hit rate.
    """
    licked = np.asarray(licked).astype(bool)
    freqs = np.array([c["freq"] for c in conds])
    go = np.array([c["go"] for c in conds])
    report_stim = ~licked if contingency == "no_reward" else licked
    correct = licked == go
    levels = sorted(set(freqs.tolist()))
    p_report = [float(report_stim[freqs == f].mean()) for f in levels]
    none = freqs == 0
    out = {
        "accuracy": float(correct.mean()),
        "p_report_stim": dict(zip(map(float, levels), p_report)),
        "psychometric": {"x": levels, "y": p_report, "xlabel": "Vagal stimulation frequency (Hz)",
                         "ylabel": "P(report stimulus)"},
    }
    if none.any():
        fa_rate = float(report_stim[none].mean())  # "felt it" with nothing there
        out["p_lick_no_stim"] = float(licked[none].mean())
        dprime, crit = {}, {}
        for f in levels:
            if f == 0:
                continue
            m = freqs == f
            zh, zf = _z(report_stim[m].mean(), m.sum()), _z(fa_rate, none.sum())
            dprime[float(f)] = zh - zf
            crit[float(f)] = -0.5 * (zh + zf)
        out["dprime"], out["criterion"] = dprime, crit
    stim = ~none
    if stim.any():
        out["p_report_stim_all"] = float(report_stim[stim].mean())
    if contingency == "no_reward":
        if stim.any():
            out["cr_rate"] = float((~licked[stim]).mean())
        if none.any():
            out["hit_rate"] = float(licked[none].mean())
    out["threshold_hz"] = fit_threshold(levels, p_report)
    return out


def fit_threshold(levels, p_report) -> float:
    """Frequency at 50% report from a logistic fit in log10(frequency) between the
    no-stim rate (floor) and the 15 Hz rate (ceiling). NaN if the curve never crosses."""
    lv = np.asarray([l for l in levels if l > 0], dtype=float)
    pr = np.asarray([p for l, p in zip(levels, p_report) if l > 0], dtype=float)
    if len(lv) < 2:
        return float("nan")
    floor = float(p_report[0]) if levels[0] == 0 else 0.0
    ceil = float(pr.max())
    if ceil - floor < 0.2:
        return float("nan")
    x = np.log10(lv)
    best, best_err = float("nan"), np.inf
    for x0 in np.linspace(x.min() - 0.5, x.max() + 0.5, 121):
        for k in (1, 2, 3, 5, 8, 12, 20):
            pred = floor + (ceil - floor) / (1 + np.exp(-k * (x - x0)))
            err = ((pred - pr) ** 2).sum()
            if err < best_err:
                best, best_err = x0, err
    return float(10**best)
