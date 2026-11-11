"""Easy-task plan, step 2, E8: shorter light and gaps before the odor (paper Extended
Data Fig 1p-s). The fitted one-area model, trained only on 4 s of 15 Hz light with the
odor right after, is tested (no training) with 1 s, 267 ms and one-step (20 ms) light,
the odor right after the light, and with 2-8 s gaps after a 4 s light.
Pass: false alarms rise as the light gets shorter or the gap longer, and hits stay high.
Writes runs/easy_e8.json."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rnn_workbench.experiment import run_experiment

base = {"baseline": 1000, "stimulus": 4000, "delay": 0, "odor": 2000, "response": 2000, "outcome": 1000}
conds = {"4 s": {}, "1 s": {"stimulus": 1000}, "267 ms (4 pulses)": {"stimulus": 260},
         "20 ms (1 pulse)": {"stimulus": 20},
         **{f"4 s + {d} s gap": {"delay": d * 1000} for d in (2, 4, 6, 8)}}
MODEL = sys.argv[1] if len(sys.argv) > 1 else "runs/fit_oi151"
GAIN_CV = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
OUT = sys.argv[3] if len(sys.argv) > 3 else "easy_e8.json"
out = {}
for name, t in conds.items():
    cfg = {"name": "easy_e8_" + Path(MODEL).name + "_" + name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus"),
           "model_dir": MODEL, "trials_per_condition": 40, "out_dir": "experiments",
           "task_overrides": {"reward_delay": 200, "gain_cv": GAIN_CV, "eval_frequencies": [15.0], "timing": {**base, **t}},
           "metrics": ["hit_rate", "p_report_stim:15.0"], "perturbations": []}
    d = run_experiment(cfg, log=lambda *a: None)
    r = json.load(open(Path(d) / "results.json"))
    out[name] = {s: {"fa_rate": round(1 - v["intact"]["p_report_stim"]["15.0"], 3),
                     "hit_rate": round(v["intact"]["hit_rate"], 3)} for s, v in r["seeds"].items()}
    print(name, out[name], flush=True)
(ROOT / "runs" / OUT).write_text(json.dumps(out, indent=1))
