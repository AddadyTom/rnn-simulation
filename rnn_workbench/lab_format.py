"""Convert the lab's 2-photon export (one folder per session) into a session directory
that data.load_session reads with signal="dff_ready".

Source layout (example: mouse OI151, 2024-08-26; see the README that comes with it):

    dff_running_4.5min.pkl     pandas DataFrame, neurons x frames, dF/F (all runs concatenated)
    trials_ind.npy             (n_runs, n_trials_per_run) trial start frame within each run
    run_len.npy                frames in each run
    conditions_played.npy      (n_trials,) 6 = go (no opto, rewarded), 5 = no-go (opto)
    responses.npy              (n_trials,) 3 hit, 2 miss, 4 correct reject, 5 false alarm
    licks_all.pkl, opto_open_all.pkl, valve1_open_all.pkl, distract_light_on_all.pkl
                               lists with one array per trial, from that trial's start to the
                               next trial's start; licks: 5 = lick onset; others: analog volts

Trial structure (go/no-go, frames from trial start at 15 Hz): light at 0 s, opto
(15 Hz) at 1-5 s on no-go trials, odor 5-7 s, reward window 7-9 s, valve at the first
lick (about 7.2 s) on hit trials.

Output directory:

    dff.npy          (n_cells, n_frames) float32
    events.csv       one row per trial (data.EVENT_COLUMNS + contingency, reward_frame,
                     outcome, opto_onset_frame)
    licks.npy        (n_frames,) int8, 1 = lick onset, for analyses
    session.json     frame rate, counts and the checks below

The stimulus onset of every trial is taken as trial start + `stim_offset_s` (1 s, per
the README), and the odor onset as trial start + `odor_offset_s` (5 s). The measured
opto onset is kept in `opto_onset_frame` to check that assumption.
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
import types
from pathlib import Path

import numpy as np

CONDITION_GO, CONDITION_NOGO = 6, 5
OUTCOMES = {3: "hit", 2: "miss", 4: "cr", 5: "fa"}


class _ArraysOnly(pickle.Unpickler):
    """Unpickler for the per-trial event lists: allows numpy arrays and nothing else."""

    def find_class(self, module, name):
        if module.split(".")[0] == "numpy":
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"unexpected object {module}.{name} in an event file")


def load_trial_list(path: Path) -> list[np.ndarray]:
    with open(path, "rb") as f:
        out = _ArraysOnly(f).load()
    if not isinstance(out, list):
        raise ValueError(f"{path}: expected a list of per-trial arrays")
    return [np.asarray(a, dtype=float) for a in out]


def load_dff_pickle(path: Path) -> np.ndarray:
    """The dF/F DataFrame was saved with pandas < 2.0; add a shim for its old index class.
    This is a full pickle load, so only use it on files from a trusted source."""
    import pandas as pd
    if not hasattr(pd, "Int64Index") and "pandas.core.indexes.numeric" not in sys.modules:
        m = types.ModuleType("pandas.core.indexes.numeric")
        m.Int64Index = m.Float64Index = m.UInt64Index = m.NumericIndex = pd.Index
        sys.modules["pandas.core.indexes.numeric"] = m
    return pd.read_pickle(path).to_numpy(np.float32)


def convert_session(src: Path, out: Path, fs: float = 15.0, dff_file: str = "dff_running_4.5min.pkl",
                    stim_offset_s: float = 1.0, odor_offset_s: float = 5.0, threshold_v: float = 2.0) -> dict:
    src, out = Path(src), Path(out)
    dff = load_dff_pickle(src / dff_file) if dff_file.endswith(".pkl") else np.load(src / dff_file)
    run_len = np.load(src / "run_len.npy", allow_pickle=False)
    trials_ind = np.load(src / "trials_ind.npy", allow_pickle=False)
    cond = np.load(src / "conditions_played.npy", allow_pickle=False).astype(int)
    resp = np.load(src / "responses.npy", allow_pickle=False).astype(int)
    ev = {k: load_trial_list(src / f"{k}_all.pkl") for k in ("licks", "opto_open", "valve1_open")}
    offset = np.r_[0, np.cumsum(run_len)[:-1]]
    start = (trials_ind + offset[:, None]).ravel().astype(int)
    n = len(start)
    if dff.shape[1] != int(run_len.sum()):
        raise ValueError(f"dF/F has {dff.shape[1]} frames but runs add up to {run_len.sum()}")
    if not (len(cond) == len(resp) == n and all(len(v) == n for v in ev.values())):
        raise ValueError("trial counts disagree between files")

    licks = np.zeros(dff.shape[1], dtype=np.int8)
    rows, problems = [], []
    s_off, o_off = int(round(stim_offset_s * fs)), int(round(odor_offset_s * fs))
    for i in range(n):
        lk = np.flatnonzero(ev["licks"][i] > 3)
        licks[np.minimum(start[i] + lk, dff.shape[1] - 1)] = 1
        opto = np.flatnonzero(ev["opto_open"][i] > threshold_v)
        valve = np.flatnonzero(ev["valve1_open"][i] > threshold_v)
        outcome = OUTCOMES.get(resp[i], "?")
        nogo = cond[i] == CONDITION_NOGO
        if nogo != bool(len(opto)):
            problems.append(f"trial {i}: condition {cond[i]} but opto {'on' if len(opto) else 'off'}")
        if (outcome == "hit") != bool(len(valve)):
            problems.append(f"trial {i}: outcome {outcome} but valve {'open' if len(valve) else 'closed'}")
        rows.append({
            "trial": i, "stim_onset_frame": int(start[i] + s_off), "freq_hz": 15.0 if nogo else 0.0,
            "odor_onset_frame": int(start[i] + o_off), "licked": int(outcome in ("hit", "fa")),
            "rewarded": int(len(valve) > 0), "contingency": "no_reward",
            "reward_frame": int(start[i] + valve[0]) if len(valve) else "",
            "outcome": outcome, "opto_onset_frame": int(start[i] + opto[0]) if len(opto) else "",
        })
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "dff.npy", dff)
    np.save(out / "licks.npy", licks)
    with open(out / "events.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lags = [r["opto_onset_frame"] - r["stim_onset_frame"] for r in rows if r["opto_onset_frame"] != ""]
    info = {"source": str(src), "fs": fs, "n_cells": int(dff.shape[0]), "n_frames": int(dff.shape[1]),
            "n_trials": n, "outcomes": {k: sum(r["outcome"] == k for r in rows) for k in OUTCOMES.values()},
            "measured_opto_onset_minus_assumed_frames": {"median": float(np.median(lags)) if lags else None,
                                                         "min": int(min(lags)) if lags else None,
                                                         "max": int(max(lags)) if lags else None},
            "problems": problems}
    (out / "session.json").write_text(json.dumps(info, indent=1))
    return info


def write_fake_lab_session(out: Path, n_cells: int = 12, trials_per_run: int = 6, run_frames: int = 2000,
                           fs: float = 15.0, seed: int = 0) -> None:
    """A tiny folder in the lab's format (dF/F saved as .npy, not a pandas pickle), for tests."""
    rng = np.random.default_rng(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    gap = 270
    ti = np.array([[200 + k * gap for k in range(trials_per_run)] for _ in range(2)])
    n = 2 * trials_per_run
    cond = np.array([CONDITION_GO, CONDITION_NOGO] * (n // 2), dtype=np.uint8)
    resp = np.where(cond == CONDITION_GO, 3, 4).astype(np.uint8)
    resp[-1] = 5  # one false alarm
    starts = (ti + np.array([0, run_frames])[:, None]).ravel()
    ends = np.r_[starts[1:], 2 * run_frames]
    lists = {"licks": [], "opto_open": [], "valve1_open": [], "distract_light_on": []}
    for i in range(n):
        L = ends[i] - starts[i]
        lk, op, va, li = np.zeros(L), np.zeros(L), np.zeros(L), np.zeros(L)
        li[:3] = 3.3
        if cond[i] == CONDITION_NOGO:
            op[17:76] = 4.8
        if resp[i] in (3, 5):
            lk[[90, 108, 112]] = 5
        if resp[i] == 3:
            va[108:110] = 3.2
        for k, v in zip(lists, (lk, op, va, li)):
            lists[k].append(v)
    dff = rng.normal(0, 0.05, (n_cells, 2 * run_frames)).astype(np.float32)
    np.save(out / "dff.npy", dff)
    np.save(out / "trials_ind.npy", ti)
    np.save(out / "run_len.npy", np.array([run_frames, run_frames]))
    np.save(out / "conditions_played.npy", cond)
    np.save(out / "responses.npy", resp)
    for k, v in lists.items():
        with open(out / f"{k}_all.pkl", "wb") as f:
            pickle.dump(v, f)
