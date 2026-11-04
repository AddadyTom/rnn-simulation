"""Load two-photon sessions (suite2p output + a trial events table) for data-constrained fits.

Expected layout of a session directory (suite2p `plane0` output plus one CSV):

    F.npy, Fneu.npy, spks.npy        (n_rois, n_frames) float32, all runs concatenated
    iscell_final.npy                 (n_rois, 2); column 0 = 1 for cells (file name configurable)
    events.csv                       one row per trial, see EVENT_COLUMNS

Frame indices in events.csv count frames of the concatenated recording (suite2p's
order of runs), starting at 0. Pickled suite2p files (ops.npy, stat.npy) are not
needed; the frame rate is a parameter (15 Hz for the SS38 session).

events.csv columns (required):
    trial              int, trial number
    stim_onset_frame   int, first frame of the 4 s vagal stimulation. For no-stimulation
                       trials, the frame where it would have started, i.e. 4 s before
                       the odor (the trial structure is the same)
    freq_hz            float, stimulation frequency; 0 = no stimulation
    odor_onset_frame   int, first frame of the odor (start of the response window)
    licked             0/1, the mouse licked in the response window
    rewarded           0/1, sugar water was delivered
optional:
    inhibition         0/1, insula opto-inhibition on this trial (excluded by default)
    contingency        "no_reward" | "reward"
    reward_frame       int, frame when the reward valve opened (blank if no reward);
                       sets the reward input at the real time instead of the task's window
    outcome            hit | miss | cr | fa (kept for analyses)

The lab's own export format (per-trial pickles plus dF/F) is converted by
lab_format.convert_session; the resulting session uses signal "dff_ready" (the lab's
dF/F) or "dff_deconv" (that dF/F deconvolved with OASIS, see deconv.py).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EVENT_COLUMNS = ("trial", "stim_onset_frame", "freq_hz", "odor_onset_frame", "licked", "rewarded")


@dataclass
class SessionData:
    traces: np.ndarray  # (n_cells, n_frames), preprocessed, non-negative
    cell_ids: np.ndarray  # suite2p ROI index of each row
    fs: float  # frames per second
    events: list[dict]


def load_events(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            missing = [c for c in EVENT_COLUMNS if c not in r]
            if missing:
                raise ValueError(f"{path}: missing columns {missing}")
            ev = {
                "trial": int(r["trial"]), "stim_onset_frame": int(r["stim_onset_frame"]),
                "freq_hz": float(r["freq_hz"]), "odor_onset_frame": int(r["odor_onset_frame"]),
                "licked": int(float(r["licked"])), "rewarded": int(float(r["rewarded"])),
            }
            for k in ("inhibition", "reward_frame"):
                if r.get(k, "") not in ("", None):
                    ev[k] = int(float(r[k]))
            if r.get("outcome"):
                ev["outcome"] = r["outcome"]
            if r.get("contingency"):
                ev["contingency"] = r["contingency"]
            rows.append(ev)
    return rows


def _smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return x
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(x, ((0, 0), (r, r)), mode="edge")
    return np.stack([np.convolve(row, k, mode="valid") for row in pad])


def preprocess(raw: np.ndarray, signal: str, fneu: np.ndarray | None = None, smooth_frames: float = 1.0,
               neucoeff: float = 0.7, baseline_pct: float = 8.0, scale_pct: float = 99.0) -> np.ndarray:
    """Turn suite2p traces into non-negative rate-like targets scaled to about [0, 1].

    signal="spks": deconvolved activity, Gaussian-smoothed, divided by each cell's
                   `scale_pct` percentile.
    signal="dff":  (F - neucoeff * Fneu), dF/F against the `baseline_pct` percentile,
                   clipped at 0 and scaled the same way.
    signal="dff_ready": dF/F computed by the lab (dff.npy), smoothed and divided by the
                   `scale_pct` percentile, not clipped (dF/F dips below 0 are real and
                   the per-cell readout has an offset).
    """
    x = raw.astype(np.float64)
    if signal == "dff":
        if fneu is not None:
            x = x - neucoeff * fneu
        f0 = np.percentile(x, baseline_pct, axis=1, keepdims=True)
        x = (x - f0) / np.maximum(np.abs(f0), 1e-6)
    elif signal == "dff_ready":
        x = _smooth(x, smooth_frames)
        scale = np.percentile(x, scale_pct, axis=1, keepdims=True)
        return (x / np.maximum(scale, 1e-9)).astype(np.float32)
    elif signal != "spks":
        raise ValueError(f"unknown signal {signal!r}")
    x = _smooth(np.maximum(x, 0.0), smooth_frames)
    scale = np.percentile(x, scale_pct, axis=1, keepdims=True)
    return (x / np.maximum(scale, 1e-9)).astype(np.float32)


def load_session(session_dir: Path, events_file: str = "events.csv", iscell_file: str = "iscell_final.npy",
                 signal: str = "spks", fs: float = 15.0, smooth_frames: float = 1.0) -> SessionData:
    d = Path(session_dir)
    if signal == "dff_ready":  # lab-computed dF/F of curated cells (lab_format.convert_session)
        raw = np.load(d / "dff.npy", mmap_mode="r", allow_pickle=False)
        traces = preprocess(raw, "dff_ready", smooth_frames=smooth_frames)
        return SessionData(traces=traces, cell_ids=np.arange(raw.shape[0]), fs=fs, events=load_events(d / events_file))
    if signal == "dff_deconv":  # the same dF/F, deconvolved with OASIS (cached next to it)
        cache = d / "spks_deconv.npy"
        if cache.exists():
            raw = np.load(cache, allow_pickle=False)
        else:
            from .deconv import deconvolve
            raw = deconvolve(np.load(d / "dff.npy", allow_pickle=False), fs)
            np.save(cache, raw)
        traces = preprocess(raw, "spks", smooth_frames=smooth_frames)
        return SessionData(traces=traces, cell_ids=np.arange(raw.shape[0]), fs=fs, events=load_events(d / events_file))
    iscell = np.load(d / iscell_file, allow_pickle=False)
    cells = np.flatnonzero(iscell[:, 0] > 0.5)
    if signal == "spks":
        raw = np.load(d / "spks.npy", mmap_mode="r", allow_pickle=False)[cells]
        traces = preprocess(raw, "spks", smooth_frames=smooth_frames)
    else:
        raw = np.load(d / "F.npy", mmap_mode="r", allow_pickle=False)[cells]
        fneu = np.load(d / "Fneu.npy", mmap_mode="r", allow_pickle=False)[cells] if (d / "Fneu.npy").exists() else None
        traces = preprocess(raw, "dff", fneu=fneu, smooth_frames=smooth_frames)
    return SessionData(traces=traces, cell_ids=cells, fs=fs, events=load_events(d / events_file))


def trial_frames(session: SessionData, pre_ms: float, total_ms: float) -> np.ndarray:
    """Frame index grid per trial, aligned on stimulus onset: (n_trials, n_frames_trial).
    Frame k of a trial sits at k * 1000 / fs ms after the window start."""
    n = int(np.floor(total_ms * session.fs / 1000.0))
    pre = int(round(pre_ms * session.fs / 1000.0))
    starts = np.array([e["stim_onset_frame"] for e in session.events]) - pre
    return starts[:, None] + np.arange(n)[None, :]


def trial_tensor(session: SessionData, pre_ms: float, total_ms: float) -> tuple[np.ndarray, np.ndarray]:
    """Activity per trial (n_trials, n_frames_trial, n_cells) and a mask of trials whose
    window lies fully inside the recording."""
    idx = trial_frames(session, pre_ms, total_ms)
    ok = (idx[:, 0] >= 0) & (idx[:, -1] < session.traces.shape[1])
    out = np.zeros(idx.shape + (session.traces.shape[0],), dtype=np.float32)
    out[ok] = session.traces[:, idx[ok]].transpose(1, 2, 0)
    return out, ok


# -- mock events and synthetic sessions (tests and demos) -----------------------------------

def write_mock_events(path: Path, frames_per_run=(27000, 27000), fs: float = 15.0,
                      freqs=(0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 15.0), p_stim: float = 0.5,
                      trial_s: float = 16.0, threshold_hz: float = 2.0, contingency: str = "no_reward",
                      pre_s: float = 1.0, post_s: float = 6.5, inhibition_fraction: float = 0.0,
                      seed: int = 0) -> list[dict]:
    """A made-up events.csv with the layout of a real session, to test the pipeline.

    Trials start every `trial_s` seconds (with up to 2 s of jitter) within each run.
    No trial window (from `pre_s` before stimulus onset to `post_s` after it) crosses a
    run boundary or the end of the recording. The stimulus lasts 4 s and the odor starts
    at its offset. Licks follow a logistic psychometric curve in log10(frequency) with
    the given threshold, a 5% false-"felt it" rate without stimulation, and the chosen
    contingency. None of this comes from real behavior.
    """
    rng = np.random.default_rng(seed)
    rows, offset, trial = [], 0, 0
    stim_frames = int(round(4.0 * fs))
    for n in frames_per_run:
        t = offset + int(round((pre_s + 2.0) * fs))
        end = offset + n - int(np.ceil(post_s * fs)) - 1
        while t <= end:
            stim = rng.random() < p_stim
            f = float(rng.choice(freqs)) if stim else 0.0
            if stim:
                p_report = 0.05 + 0.95 / (1 + np.exp(-4 * (np.log10(f) - np.log10(threshold_hz))))
            else:
                p_report = 0.05
            reported = rng.random() < p_report
            licked = int(not reported) if contingency == "no_reward" else int(reported)
            go = (not stim) if contingency == "no_reward" else stim
            row = {"trial": trial, "stim_onset_frame": int(t), "freq_hz": f,
                   "odor_onset_frame": int(t + stim_frames), "licked": licked, "rewarded": int(licked and go),
                   "contingency": contingency}
            if inhibition_fraction > 0:
                row["inhibition"] = int(rng.random() < inhibition_fraction)
            rows.append(row)
            trial += 1
            t += int(round((trial_s + rng.uniform(0, 2.0)) * fs))
        offset += n
    fields = list(EVENT_COLUMNS) + ["contingency"] + (["inhibition"] if inhibition_fraction > 0 else [])
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return rows


def write_synthetic_session(out_dir: Path, n_rois: int = 60, n_trials: int = 120, fs: float = 15.0,
                            freqs=(0.5, 1.0, 2.0, 3.0, 5.0, 15.0), threshold_hz: float = 2.0,
                            seed: int = 0) -> dict:
    """A fake session in the expected format, with the cell classes the paper reports:
    sigmoidal (increase with frequency), Gaussian (prefer ~1-3 Hz), reward-responsive,
    and stimulus-suppressed cells, plus untuned cells. Returns the ground truth."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    iti, stim_s, resp_s, out_s = 3.0, 4.0, 1.5, 1.0
    trial_s = iti + stim_s + resp_s + out_s
    n_frames = int((n_trials * trial_s + iti) * fs)
    kinds = rng.choice(["sigmoid", "gaussian", "reward", "suppressed", "none"], size=n_rois,
                       p=[0.2, 0.2, 0.2, 0.15, 0.25])
    thr = 10 ** rng.uniform(np.log10(1.0), np.log10(5.0), n_rois)
    peak = 10 ** rng.uniform(0.0, np.log10(3.0), n_rois)
    rate = 0.05 * np.ones((n_rois, n_frames))
    events = []
    for t in range(n_trials):
        start = int((iti + t * trial_s) * fs)
        stim = rng.random() < 0.5
        f = float(rng.choice(freqs)) if stim else 0.0
        s0, s1 = start, start + int(stim_s * fs)
        o0, o1 = s1, s1 + int(resp_s * fs)
        r0, r1 = o1, o1 + int(out_s * fs)
        p_detect = 1 / (1 + np.exp(-4 * (np.log10(max(f, 1e-3)) - np.log10(threshold_hz)))) if stim else 0.05
        detected = rng.random() < p_detect
        licked = int(not detected)  # NoReward contingency
        rewarded = int(licked and not stim)
        events.append({"trial": t, "stim_onset_frame": s0, "freq_hz": f, "odor_onset_frame": o0,
                       "licked": licked, "rewarded": rewarded})
        if stim:
            lf = np.log10(f)
            tt = np.arange(s1 - s0) / fs
            adapt = 0.35 + 0.65 * np.exp(-tt / 0.85)  # early peak then adaptation, as in the paper
            for i in range(n_rois):
                if kinds[i] == "sigmoid":
                    a = 1 / (1 + np.exp(-5 * (lf - np.log10(thr[i]))))
                elif kinds[i] == "gaussian":
                    a = np.exp(-0.5 * ((lf - np.log10(peak[i])) / 0.25) ** 2)
                elif kinds[i] == "suppressed":
                    a = -0.04 / 0.6 * min(1.0, f / 5.0)
                else:
                    a = 0.0
                rate[i, s0:s1] += 0.6 * a * adapt * (1.0 + 0.3 * detected)
        if rewarded:
            rate[kinds == "reward", r0:r1] += 0.6
    rate = np.maximum(rate, 0)
    spks = rng.poisson(rate * 3).astype(np.float32)
    decay = np.exp(-1 / (0.7 * fs))
    calcium = np.zeros_like(spks)
    acc = np.zeros(n_rois, dtype=np.float32)
    for k in range(n_frames):
        acc = acc * decay + spks[:, k]
        calcium[:, k] = acc
    F = (100 + 20 * calcium + rng.normal(0, 2, calcium.shape)).astype(np.float32)
    np.save(out_dir / "F.npy", F)
    np.save(out_dir / "Fneu.npy", (0.3 * F + 30).astype(np.float32))
    np.save(out_dir / "spks.npy", spks)
    iscell = np.zeros((n_rois + 5, 2))
    iscell[:n_rois, 0] = 1
    iscell[:, 1] = rng.random(n_rois + 5)
    # Five extra non-cell ROIs so that the iscell filter is exercised.
    for name in ("F", "Fneu", "spks"):
        a = np.load(out_dir / f"{name}.npy")
        np.save(out_dir / f"{name}.npy", np.concatenate([a, a[:5] * 0 + 50], 0))
    np.save(out_dir / "iscell_final.npy", iscell)
    with open(out_dir / "events.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(EVENT_COLUMNS))
        w.writeheader()
        w.writerows(events)
    return {"kinds": kinds.tolist(), "thresholds": thr.tolist(), "peaks": peak.tolist()}
