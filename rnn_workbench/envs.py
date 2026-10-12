"""Custom NeuroGym environments.

NeedGatedCue-v0 is modeled on the head-fixed paradigms of Livneh et al. (2017, 2020)
and Talpir & Livneh (2024): a visual cue predicts food, water, or nothing, and the
mouse should lick in a response window only when the cue predicts the outcome its
current physiological need calls for. Need state enters as tonic inputs (hunger and
thirst drive, standing in for hypothalamic AgRP / SFO signals) present for the whole
trial, so the network must gate cue responses by state.

Trial structure (trace conditioning):  fixation -> stimulus (cue) -> delay -> decision
Observations: [fixation, cue_food, cue_water, cue_neutral, hunger, thirst]
Actions:      0 = no lick, 1 = lick
"""

from __future__ import annotations

import numpy as np
from neurogym.core import TrialEnv
from neurogym.utils import spaces

CUES = ("food", "water", "neutral")
STATES = ("hungry", "thirsty", "sated")
GO = {("hungry", "food"), ("thirsty", "water")}


class NeedGatedCue(TrialEnv):
    metadata = {  # noqa: RUF012
        "paper_link": "https://doi.org/10.1038/nature22375",
        "paper_name": "Homeostatic circuits selectively gate food cue responses in insular cortex",
        "tags": ["interoception", "go-no-go", "supervised"],
    }

    def __init__(
        self,
        dt: int = 20,
        timing: dict | None = None,
        sigma: float = 0.5,
        need_noise: float = 0.05,
        threshold: float = 0.4,
        rewards: dict | None = None,
    ) -> None:
        super().__init__(dt=dt)
        self.timing = {"fixation": 300, "stimulus": 500, "delay": 500, "decision": 300}
        if timing:
            self.timing.update(timing)
        self.sigma = sigma / np.sqrt(dt)  # cue noise, scaled like NeuroGym's own tasks
        self.need_noise = need_noise
        self.threshold = threshold
        self.rewards = {"correct": 1.0, "fail": 0.0}
        if rewards:
            self.rewards.update(rewards)
        self.cues = list(CUES)
        self.states = list(STATES)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(6,), dtype=np.float32,
            name={"fixation": 0, "cue": [1, 2, 3], "need": [4, 5]},
        )
        self.action_space = spaces.Discrete(2, name={"no_lick": 0, "lick": 1})

    def sample_need(self, state: str) -> tuple[float, float]:
        high = lambda: float(self.rng.uniform(0.6, 1.0))  # noqa: E731
        low = lambda: float(self.rng.uniform(0.0, 0.2))  # noqa: E731
        if state == "hungry":
            return high(), low()
        if state == "thirsty":
            return low(), high()
        return low(), low()

    def lick_target(self, cue: str, hunger: float, thirst: float) -> int:
        if cue == "food":
            return int(hunger > self.threshold and hunger >= thirst)
        if cue == "water":
            return int(thirst > self.threshold and thirst > hunger)
        return 0

    def _new_trial(self, **kwargs):
        # Go conditions (hungry+food, thirsty+water) are drawn twice as often as each
        # no-go condition, so about a third of trials call for licking.
        combos = [(s, c) for s in self.states for c in self.cues]
        weights = np.array([2.0 if (s, c) in GO else 1.0 for s, c in combos])
        state, cue = combos[self.rng.choice(len(combos), p=weights / weights.sum())]
        trial = {"state": state, "cue": cue}
        trial.update(kwargs)
        if "hunger" not in trial or "thirst" not in trial:
            trial["hunger"], trial["thirst"] = self.sample_need(trial["state"])
        trial["ground_truth"] = self.lick_target(trial["cue"], trial["hunger"], trial["thirst"])

        self.add_period(["fixation", "stimulus", "delay", "decision"])
        self.add_ob(1, period=["fixation", "stimulus", "delay"], where="fixation")
        cue_vec = np.zeros(3)
        cue_vec[self.cues.index(trial["cue"])] = 1.0
        self.add_ob(cue_vec, period="stimulus", where="cue")
        self.add_randn(0, self.sigma, period="stimulus", where="cue")
        need = np.array([trial["hunger"], trial["thirst"]])
        for p in ("fixation", "stimulus", "delay", "decision"):
            self.add_ob(need, period=p, where="need")
        self.add_randn(0, self.need_noise, where="need")
        self.set_groundtruth(trial["ground_truth"], period="decision")
        return trial

    def _step(self, action):
        new_trial, reward = False, 0.0
        gt = self.gt_now
        if self.in_period("decision"):
            if action == gt:
                reward = self.rewards["correct"]
                self.performance = 1
            else:
                reward = self.rewards["fail"]
            new_trial = action == 1  # a lick ends the trial
        return self.ob_now, reward, False, False, {"new_trial": new_trial, "gt": gt}


LOCAL_ENVS = {"NeedGatedCue-v0": NeedGatedCue}
