"""E7 follow-up: why does the reversal retrain lose the light-specific units?
Three retrains of runs/fit_oi151_noise15 on the reversed rule (light = water), each
changing one thing from configs/train_reversal_oi151.yaml (same task, 600 steps of 32
trials, lr 1e-4):
  anchor     + a loss keeping the 576 cell-copying units' average light-window response
               on light trials equal to the pre-reversal network's (which was fitted to
               the OI151 cells)
  anchor_all the same, but on light and no-light trials, each averaged separately
  rate_l2    activity penalty 1e-3 instead of 1e-4
  new_odor   the reversed rule uses a second, new odor input, as the mice got a new odor
Then the E7 test (scripts/easy_e7.py) on each. Usage: python scripts/reversal_variants.py VARIANT
Writes runs/reversal_<variant>/seed_*/model.pt and runs/easy_e7_<variant>.json."""
import copy, json, sys
from dataclasses import asdict
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import easy_e7 as E
from rnn_workbench.train import load_model, _save
from rnn_workbench.vagal_task import VagalOptoTask, VagalTaskConfig

variant = sys.argv[1]
assert variant in ("anchor", "anchor_all", "rate_l2", "new_odor")
STEPS, BATCH, LR = 600, 32, 1e-4
RATE_L2 = 1e-3 if variant == "rate_l2" else 1e-4
ANCHOR_W = 10.0
tc = VagalTaskConfig(**{**asdict(E.cfg.task), "contingency": "reward", "reward_delay": 200, "p_stim": 0.5,
                        "train_frequencies": [15.0], "eval_frequencies": [15.0]})
task = VagalOptoTask(tc)
on, off = task.periods["stimulus"]


class NewOdor(nn.Module):
    """Adds a 4th input: the odor goes to the new channel, the old odor channel stays 0."""
    def __init__(self, m):
        super().__init__()
        self.m = m
        w = m.w_in_raw.data
        new = torch.rand(w.shape[0], 1) * 0.5 * w[:, 1:2].mean() / 0.25  # same scale as the old odor column
        m.w_in_raw = nn.Parameter(torch.cat([w, new], 1))
        m.in_mask = torch.cat([m.in_mask, m.in_mask[:, 1:2]], 1)
        m.cfg.n_inputs = 4

    def forward(self, x, **kw):
        x4 = torch.cat([x, x[:, :, 1:2]], 2)
        x4[:, :, 1] = 0.0
        return self.m(x4, **kw)


res = {}
for seed in (0, 1, 2):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = load_model(E.BEFORE / f"seed_{seed}")
    m.train()
    ref = copy.deepcopy(m).eval()
    net = NewOdor(m) if variant == "new_odor" else m
    units = torch.tensor(torch.load(E.BEFORE / f"seed_{seed}" / "model.pt", weights_only=False)["unit_of_cell"])
    opt = torch.optim.Adam([{"params": [m.w_raw], "lr": LR * min(1.0, 100 / m.cfg.n_units)},
                            {"params": [p for p in m.parameters() if p is not m.w_raw], "lr": LR}])
    log = []
    for step in range(1, STEPS + 1):
        x, y, w = task.sample_batch(BATCH, rng)
        out, rates = net(x)
        ce = F.cross_entropy(out.reshape(-1, 2), y.reshape(-1), reduction="none")
        loss = (ce * w.reshape(-1)).sum() / w.sum() + RATE_L2 * rates.pow(2).mean() + 1e-5 * m.w_rec.pow(2).sum()
        if variant.startswith("anchor"):
            light = x[on:off, :, 0].sum(0) > 0
            with torch.no_grad():
                _, r0 = ref(x)
            for grp in ([light, ~light] if variant == "anchor_all" else [light]):
                a = rates[on:off][:, grp][:, :, units].mean(1)
                b = r0[on:off][:, grp][:, :, units].mean(1)
                loss = loss + ANCHOR_W * (a - b).pow(2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step % 50 == 0:
            b_ = task.eval_batch(20, seed=1234, freqs=[15.0])
            o, _ = task.run(net, b_)
            acc = float((task.choices(o, b_) == np.array([c["go"] for c in b_.conditions])).mean())
            log.append((step, round(acc, 3)))
            print(variant, seed, step, round(loss.item(), 4), acc, flush=True)
    m.eval()
    out_dir = ROOT / "runs" / f"reversal_{variant}" / f"seed_{seed}"
    _save(m, out_dir / "model.pt")
    res[f"seed {seed}"] = {"learning": log, "before (light = no water)": E.run(E.BEFORE, seed, "no_reward"),
                           "after (light = water)": E.run_model(net, seed, "reward")}
    print(seed, res[f"seed {seed}"], flush=True)
(ROOT / "runs" / f"easy_e7_{variant}.json").write_text(json.dumps(res, indent=1))
