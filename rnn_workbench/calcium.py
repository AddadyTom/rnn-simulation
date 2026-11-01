"""Turn two-photon calcium recordings into PSTHs that `fit.py` can fit.

The RNN's units have rates, but GCaMP fluorescence is a slow, filtered copy of
spiking (decay of roughly 0.2-1 s depending on the indicator). Fitting dF/F directly
would make the network learn the indicator's kinetics as if they were circuit
dynamics, so we first undo the indicator, as most data-constrained RNN papers on
imaging data do (e.g. Rajan, Harvey & Tank 2016; Bondanelli et al. 2021):

    1. deconvolve:  "ar1" inverts a first-order indicator model, s[t] = f[t] - g f[t-1]
                    with g = exp(-1 / (tau_decay * frame_rate)), rectified;
                    "none" uses the traces as given (e.g. suite2p `spks` or OASIS output,
                    which are better deconvolutions than ar1 and the recommended input)
    2. smooth       with a Gaussian kernel (smooth_ms)
    3. resample     from the imaging frame rate to the model's time step dt
    4. average      trials within each condition, in the order the task's eval_trials uses
    5. normalize    each neuron to its peak over conditions and time (rates in [0, 1])

    traces: (trials, frames, neurons) aligned to trial start
    conditions: per-trial condition key, e.g. ("hungry", "food_cue")
    order: the condition keys in fit order (see fit.condition_index on eval_trials)
"""

from __future__ import annotations

import numpy as np


def deconvolve_ar1(traces: np.ndarray, frame_rate: float, tau_decay: float) -> np.ndarray:
    g = np.exp(-1.0 / (tau_decay * frame_rate))
    s = traces.copy()
    s[:, 1:] = traces[:, 1:] - g * traces[:, :-1]
    return np.maximum(s, 0.0)


def gaussian_smooth(x: np.ndarray, sigma_frames: float, axis: int = 1) -> np.ndarray:
    if sigma_frames <= 0:
        return x
    half = int(np.ceil(4 * sigma_frames))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma_frames) ** 2)
    k /= k.sum()
    pad = [(0, 0)] * x.ndim
    pad[axis] = (half, half)
    xp = np.pad(x, pad, mode="edge")
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), axis, xp)


def resample(x: np.ndarray, frame_rate: float, dt_ms: float, n_steps: int, axis: int = 1) -> np.ndarray:
    """Linear interpolation from frame times to model step times (step t ends at (t+1)*dt)."""
    t_frames = np.arange(x.shape[axis]) / frame_rate * 1000.0
    t_model = (np.arange(n_steps) + 1) * dt_ms
    if t_model[-1] > t_frames[-1] + 1e-9:
        raise ValueError(f"recording is {t_frames[-1]:.0f} ms long, the task needs {t_model[-1]:.0f} ms")
    return np.apply_along_axis(lambda v: np.interp(t_model, t_frames, v), axis, x)


def psth_from_calcium(traces, conditions, order, frame_rate: float, dt_ms: float, n_steps: int,
                      deconvolve: str = "ar1", tau_decay: float = 0.7, smooth_ms: float = 50.0,
                      normalize: bool = True) -> np.ndarray:
    """(trials, frames, neurons) fluorescence -> (C, n_steps, neurons) PSTH for fit.py."""
    x = np.asarray(traces, dtype=float)
    if deconvolve == "ar1":
        x = deconvolve_ar1(x, frame_rate, tau_decay)
    elif deconvolve != "none":
        raise ValueError(f"unknown deconvolve {deconvolve!r}")
    x = gaussian_smooth(x, smooth_ms / 1000.0 * frame_rate)
    x = resample(x, frame_rate, dt_ms, n_steps)
    keys = [tuple(c) if isinstance(c, (list, tuple, np.ndarray)) else c for c in conditions]
    out = []
    for key in order:
        sel = np.array([k == key for k in keys])
        if not sel.any():
            raise ValueError(f"no trials for condition {key!r}")
        out.append(x[sel].mean(0))
    p = np.stack(out)
    if normalize:
        peak = p.max(axis=(0, 1), keepdims=True)
        p = p / np.where(peak > 0, peak, 1.0)
    return p
