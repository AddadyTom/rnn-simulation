"""The paper's single-cell and population analyses, for model units and recorded cells alike.

All functions take trial-by-unit response matrices, so the same code runs on network
rates (from `unit_responses`) and on imaging data (from data.trial_tensor).

    tuning_classes      sigmoidal vs Gaussian frequency tuning, with a shuffle test (Fig 2e-i)
    choice_probability  auROC of detected vs undetected trials at matched stimuli (Fig 3a-c)
    decode_accuracy     cross-validated linear SVM (Fig 2k, Fig 3d)
    neurometric         population decoding of each frequency vs no stimulation (Fig 2k)
"""

from __future__ import annotations

import warnings

import numpy as np


def unit_responses(rates, periods: dict, window: str = "stimulus", baseline: str = "baseline") -> np.ndarray:
    """Mean rate in `window` minus mean rate in `baseline`, per trial and unit.
    rates: (T, B, N) tensor or array. Returns (B, N)."""
    r = rates.detach().cpu().numpy() if hasattr(rates, "detach") else np.asarray(rates)
    s, e = periods[window]
    out = r[s:e].mean(0)
    if baseline:
        bs, be = periods[baseline]
        out = out - r[bs:be].mean(0)
    return out


# -- tuning ---------------------------------------------------------------------------

def _sigmoid(x, base, amp, x0, k):
    return base + amp / (1 + np.exp(-k * (x - x0)))


def _gauss(x, base, amp, mu, sd):
    return base + amp * np.exp(-0.5 * ((x - mu) / sd) ** 2)


def _fit_r2(fn, x, y, p0, bounds):
    from scipy.optimize import curve_fit
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p, _ = curve_fit(fn, x, y, p0=p0, bounds=bounds, maxfev=4000)
    except (RuntimeError, ValueError):
        return -np.inf, None
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1 - ((fn(x, *p) - y) ** 2).sum() / max(ss_tot, 1e-12), p


def _fit_both(x, y):
    lo, hi = x.min(), x.max()
    span = max(np.ptp(y), 1e-6)
    r_s, p_s = _fit_r2(_sigmoid, x, y, [y.min(), span, (lo + hi) / 2, 5.0],
                       ([-np.inf, 0, lo - 1, 0.5], [np.inf, np.inf, hi + 1, 50]))
    j = int(np.argmax(y))
    r_g, p_g = _fit_r2(_gauss, x, y, [np.median(y), span, x[j], 0.3],
                       ([-np.inf, 0, lo, 0.05], [np.inf, np.inf, hi, 2.0]))
    # A Gaussian whose peak sits at the top frequency is a saturating, sigmoid-like curve.
    if p_g is not None and p_g[2] >= hi - 1e-6:
        r_g = -np.inf
    return (r_s, p_s), (r_g, p_g)


def tuning_classes(resp: np.ndarray, freqs: np.ndarray, n_shuffle: int = 100, alpha: float = 0.05,
                   seed: int = 0) -> list[dict]:
    """Classify each unit's frequency tuning as "sigmoid", "gaussian" or "none".

    resp:  (trials, units) responses (e.g. from unit_responses); freqs: (trials,) Hz, 0 = none.
    For each unit, sigmoid and Gaussian curves are fitted to the mean response per
    frequency, in log10(frequency), with no-stimulation trials placed half a log unit
    below the lowest frequency. The better fit counts only if its R^2 beats the
    (1 - alpha) quantile of the best R^2 on frequency-shuffled data. The paper fits the
    fraction of trials with a significant response instead of the mean; the logic of
    the comparison is the same.
    """
    freqs = np.asarray(freqs, dtype=float)
    levels = np.unique(freqs)
    pos = levels[levels > 0]
    x_of = {f: (np.log10(f) if f > 0 else np.log10(pos.min()) - 0.5) for f in levels}
    x = np.array([x_of[f] for f in levels])
    rng = np.random.default_rng(seed)
    out = []
    for u in range(resp.shape[1]):
        y = np.array([resp[freqs == f, u].mean() for f in levels])
        (r_s, p_s), (r_g, p_g) = _fit_both(x, y)
        best = max(r_s, r_g)
        null = []
        for _ in range(n_shuffle):
            fs = rng.permutation(freqs)
            ys = np.array([resp[fs == f, u].mean() for f in levels])
            (a, _), (b, _) = _fit_both(x, ys)
            null.append(max(a, b))
        thr = np.quantile(null, 1 - alpha) if null else -np.inf
        kind = "none"
        if best > thr and np.isfinite(best):
            kind = "sigmoid" if r_s >= r_g else "gaussian"
        rec = {"unit": u, "kind": kind, "r2_sigmoid": float(r_s), "r2_gaussian": float(r_g), "r2_null": float(thr)}
        if kind == "sigmoid":
            rec["threshold_hz"] = float(10 ** p_s[2])
        if kind == "gaussian":
            rec["peak_hz"] = float(10 ** p_g[2])
        out.append(rec)
    return out


# -- choice probability -----------------------------------------------------------------

def tuning_classes_checked(resp: np.ndarray, freqs: np.ndarray, min_amp_frac: float = 0.1,
                           gauss_drop: float = 0.3, **kw) -> list[dict]:
    """tuning_classes with the checks a model population needs before counting.

    Model trials are much less noisy than imaging trials, so the shuffle test alone
    passes almost every unit, and a suppressed unit can fit a (negative) Gaussian. So:
    - "suppressed": the strongest change from the no-stimulation level is a drop.
    - "weak": the largest rise is under `min_amp_frac` of the population's 95th
      percentile rise (scale-free, so it works for dF/F and model rates alike).
    - "gaussian" needs a peak below the top frequency and a fall at the top frequency of
      at least `gauss_drop` of the rise; otherwise the unit is refitted as sigmoid-like.
    - "sigmoid" needs its threshold inside the tested range; a curve still steepening at
      the top frequency is "rising" instead.
    The remaining units go through tuning_classes (with its shuffle test, `kw`)."""
    freqs = np.asarray(freqs, dtype=float)
    levels = np.unique(freqs)
    pos = levels[levels > 0]
    mean = np.array([resp[freqs == f].mean(0) for f in levels])  # (levels, units)
    rise = mean[1:].max(0) - mean[0]
    drop = mean[0] - mean[1:].min(0)
    ref = np.percentile(np.maximum(rise, 0), 95) if resp.shape[1] else 0.0
    out = [None] * resp.shape[1]
    keep = []
    for u in range(resp.shape[1]):
        if drop[u] > rise[u]:
            out[u] = {"unit": u, "kind": "suppressed"}
        elif ref <= 0 or rise[u] < min_amp_frac * ref:
            out[u] = {"unit": u, "kind": "weak"}
        else:
            keep.append(u)
    if keep:
        for rec, u in zip(tuning_classes(resp[:, keep], freqs, **kw), keep):
            rec = {**rec, "unit": u}
            if rec["kind"] == "gaussian":
                y = mean[1:, u]
                peak = y.max()
                if np.argmax(y) == len(y) - 1 or peak - y[-1] < gauss_drop * rise[u]:
                    rec["kind"] = "sigmoid" if rec["r2_sigmoid"] > rec["r2_null"] else "none"
                    rec.pop("peak_hz", None)
            if rec["kind"] == "sigmoid":
                if "threshold_hz" not in rec:  # relabelled from Gaussian: fit the sigmoid's threshold
                    x = np.array([np.log10(f) if f > 0 else np.log10(pos.min()) - 0.5 for f in levels])
                    (_, p_s), _ = _fit_both(x, mean[:, u])
                    rec["threshold_hz"] = float(10 ** p_s[2]) if p_s is not None else float("nan")
                if not (pos.min() <= rec["threshold_hz"] <= pos.max()):
                    rec["kind"] = "rising"
            out[u] = rec
    return out


def auroc(a: np.ndarray, b: np.ndarray) -> float:
    """P(random a > random b) + 0.5 P(tie), i.e. the area under the ROC curve."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    from scipy.stats import rankdata
    ranks = rankdata(np.concatenate([a, b]))
    return float((ranks[: len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def choice_probability(resp: np.ndarray, detected: np.ndarray, strata: np.ndarray | None = None,
                       n_perm: int = 500, seed: int = 0) -> dict:
    """Per-unit choice probability: auROC of responses on detected vs undetected trials.

    Use stimulation trials only. `strata` (e.g. frequency) makes the comparison within
    each stimulus level and averages the auROCs weighted by trial count, so stimulus
    strength cannot masquerade as choice; shuffles also stay within strata.
    Returns cp (units,), p-values (two-sided) and the fraction significant at 0.05.
    """
    detected = np.asarray(detected).astype(bool)
    strata = np.zeros(len(detected)) if strata is None else np.asarray(strata)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    groups = [g for g in groups if detected[g].any() and (~detected[g]).any()]
    if not groups:
        raise ValueError("no stratum has both detected and undetected trials")

    def cp_all(det):
        num, den = np.zeros(resp.shape[1]), 0.0
        for g in groups:
            w = len(g)
            num += w * np.array([auroc(resp[g][det[g], u], resp[g][~det[g], u]) for u in range(resp.shape[1])])
            den += w
        return num / den

    cp = cp_all(detected)
    null = np.empty((n_perm, resp.shape[1]))
    for i in range(n_perm):
        d = detected.copy()
        for g in groups:
            d[g] = rng.permutation(d[g])
        null[i] = cp_all(d)
    p = ((np.abs(null - 0.5) >= np.abs(cp - 0.5) - 1e-12).sum(0) + 1) / (n_perm + 1)
    return {"cp": cp, "p": p, "fraction_significant": float((p < 0.05).mean())}


# -- decoding -------------------------------------------------------------------------

def decode_accuracy(X: np.ndarray, y: np.ndarray, n_splits: int = 5, C: float = 1.0, seed: int = 0,
                    balance_groups: np.ndarray | None = None) -> float:
    """Cross-validated accuracy of a linear SVM (standardized features), stratified by y.
    `balance_groups` subsamples so that each class has the same count within each group
    (e.g. frequency), which removes the frequency confound from choice decoding."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    X, y = np.asarray(X), np.asarray(y)
    rng = np.random.default_rng(seed)
    if balance_groups is not None:
        keep = []
        for g in np.unique(balance_groups):
            idx = np.flatnonzero(balance_groups == g)
            classes = [idx[y[idx] == c] for c in np.unique(y)]
            if len(classes) < 2:
                continue
            n = min(len(c) for c in classes)
            keep += [i for c in classes for i in rng.choice(c, size=n, replace=False)]
        keep = np.array(sorted(keep))
        X, y = X[keep], y[keep]
    n_min = np.bincount(np.unique(y, return_inverse=True)[1]).min()
    k = min(n_splits, n_min)
    if k < 2:
        return float("nan")
    clf = make_pipeline(StandardScaler(), LinearSVC(C=C, max_iter=5000))
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return float(cross_val_score(clf, X, y, cv=cv).mean())


def neurometric(resp: np.ndarray, freqs: np.ndarray, **kw) -> dict:
    """Decoding accuracy of each frequency vs no-stimulation trials (Fig 2k)."""
    freqs = np.asarray(freqs, float)
    none = freqs == 0
    out = {}
    for f in np.unique(freqs[freqs > 0]):
        m = none | (freqs == f)
        out[float(f)] = decode_accuracy(resp[m], (freqs[m] == f).astype(int), **kw)
    return out
