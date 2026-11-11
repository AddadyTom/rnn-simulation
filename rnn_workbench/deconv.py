"""Turn dF/F back into estimated firing (plan step 4, way A).

Two-photon data-constrained RNN papers deconvolve before fitting (Rajan, Harvey &
Tank 2016; Bondanelli et al. 2021), because GCaMP fluorescence is a slow copy of the
spiking and a network fit to dF/F would otherwise have to imitate the indicator.

`deconvolve` uses OASIS with an AR(1) indicator model (Friedrich, Zhou & Paninski
2017). With the `oasis-deconv` package installed it uses that package, which also
estimates each cell's decay, baseline, noise and sparsity penalty. Without it, a
NumPy version of the same algorithm runs with a fixed decay `tau_s` and no penalty
(slower, and less sparse).

The other way (way B), slowing the model's rates with a calcium decay before
comparing them with dF/F, lives in fit.py (`calcium_tau_ms`).
"""

from __future__ import annotations

import numpy as np


def oasis_ar1(y: np.ndarray, g: float, lam: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """OASIS for an AR(1) model: minimise 1/2 |c - y|^2 + lam |s|_1 subject to
    s_t = c_t - g c_{t-1} >= 0. Returns the denoised calcium c and the spikes s."""
    y = np.asarray(y, dtype=float).copy()
    T = len(y)
    y[:-1] -= lam * (1 - g)
    y[-1] -= lam
    v, w, start, length = [], [], [], []
    for t in range(T):
        v.append(y[t]); w.append(1.0); start.append(t); length.append(1)
        while len(v) > 1 and v[-1] < g ** length[-2] * v[-2]:
            gl = g ** length[-2]
            w_new = w[-2] + gl * gl * w[-1]
            v[-2] = (w[-2] * v[-2] + gl * w[-1] * v[-1]) / w_new
            w[-2] = w_new
            length[-2] += length[-1]
            v.pop(); w.pop(); start.pop(); length.pop()
    c = np.empty(T)
    for vi, ti, li in zip(v, start, length):
        c[ti:ti + li] = max(vi, 0.0) * g ** np.arange(li)
    s = c.copy()
    s[1:] -= g * c[:-1]
    return c, np.maximum(s, 0.0)


def deconvolve(dff: np.ndarray, fs: float, tau_s: float = 1.0, baseline_pct: float = 10.0,
               method: str = "auto", log=None) -> np.ndarray:
    """(n_cells, n_frames) dF/F -> (n_cells, n_frames) non-negative spike estimates.

    method: "auto" (the oasis package if installed, else NumPy), "oasis" or "numpy".
    """
    dff = np.asarray(dff, dtype=np.float64)
    use_pkg = method in ("auto", "oasis")
    if use_pkg:
        try:
            from oasis.functions import deconvolve as oasis_deconvolve
        except ImportError:
            if method == "oasis":
                raise
            use_pkg = False
    out = np.zeros_like(dff, dtype=np.float32)
    g = float(np.exp(-1.0 / (tau_s * fs)))
    for i, y in enumerate(dff):
        if use_pkg:
            # Estimates g, baseline, noise and the sparsity penalty for each cell.
            _, s, _, _, _ = oasis_deconvolve(y, penalty=1)
        else:
            _, s = oasis_ar1(y - np.percentile(y, baseline_pct), g)
        out[i] = s
        if log and (i + 1) % 100 == 0:
            log(f"  deconvolved {i + 1}/{len(dff)} cells")
    return out
