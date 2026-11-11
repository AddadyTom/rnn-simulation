"""Run the easy-task tests E1-E8 on one data-constrained fit (e.g. a fewer-cells run).

Steps, each skipped when its output already exists:
  1. noisy refit: 300 more steps with training noise 0.15, as fit_oi151_noise15 (runs/<run>_noise15)
  2. E2/E3: scripts/paper_tests.py <run>                  -> runs/paper_tests_<run>.json
  3. E4:    scripts/easy_e4.py <run>                      -> runs/easy_e4_<run>.json
  4. E5:    partial stGtACR2 on <run> and <run>_noise15   -> experiments/easy_e5_<run>[_noise15]
  5. E7:    reversal training from <run>_noise15 (as configs/train_reversal_oi151.yaml),
            then scripts/easy_e7.py                       -> runs/easy_e7_<run>.json
  6. E8:    scripts/easy_e8.py on <run>_noise15, gain_cv 1.0 -> runs/easy_e8_<run>.json
E1 (task accuracy) is in runs/<run>/summary.json.
Usage: python scripts/size_tests.py <run> [scratch_dir]"""
import json, subprocess, sys, tempfile
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import rnn_workbench.fit as F
from rnn_workbench.experiment import run_experiment

run = sys.argv[1]
tmp = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(tempfile.mkdtemp())
runs = ROOT / "runs"
py = lambda *a: subprocess.run([sys.executable, *map(str, a)], cwd=ROOT, check=True)

# 1. noisy refit
noisy = f"{run}_noise15"
if not (runs / noisy / "summary.json").exists():
    d = json.load(open(runs / run / "train_config.json"))["fit"]
    d.update(name=noisy, noise_std=0.15, steps=300, init_from=f"runs/{run}")
    F.fit(F.load_fit_config(d))
# 2-3. E2/E3 and E4
if not (runs / f"paper_tests_{run}.json").exists():
    py("scripts/paper_tests.py", run)
if not (runs / f"easy_e4_{run}.json").exists():
    py("scripts/easy_e4.py", run)
# 4. E5
for m in (run, noisy):
    if not (ROOT / "experiments" / f"easy_e5_{m}" / "results.json").exists():
        c = yaml.safe_load(open(ROOT / "configs/exp_easy_e5.yaml"))
        c.update(name=f"easy_e5_{m}", model_dir=f"runs/{m}")
        run_experiment(c, log=lambda *a: None)
# 5. E7
rev = f"reversal_{run}"
if not (runs / rev / "train_config.json").exists():
    c = yaml.safe_load(open(ROOT / "configs/train_reversal_oi151.yaml"))
    c.update(name=rev, init_from=f"runs/{noisy}", seeds=json.load(open(runs / run / "train_config.json"))["seeds"])
    p = tmp / f"{rev}.yaml"; p.write_text(yaml.safe_dump(c))
    py("-m", "rnn_workbench", "train", p)
if not (runs / f"easy_e7_{run}.json").exists():
    py("scripts/easy_e7.py", noisy, rev, f"easy_e7_{run}.json")
# 6. E8
if not (runs / f"easy_e8_{run}.json").exists():
    py("scripts/easy_e8.py", f"runs/{noisy}", "1.0", f"easy_e8_{run}.json")
print("done", run)
