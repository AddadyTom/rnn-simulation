"""Command line entry point.

    python -m rnn_workbench train configs/train_pdm.yaml
    python -m rnn_workbench experiment configs/exp_silence.yaml
    python -m rnn_workbench report experiments/<name>
"""

import argparse
import sys
from pathlib import Path

import yaml


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rnn_workbench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_ in (("train", "train an ensemble from a YAML config"),
                        ("experiment", "run perturbations from a YAML config and write a report")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("config", type=Path)
    p = sub.add_parser("report", help="rebuild the HTML report for an experiment directory")
    p.add_argument("exp_dir", type=Path)
    args = ap.parse_args(argv)

    if args.cmd == "train":
        from .train import load_train_config, train
        root = train(load_train_config(yaml.safe_load(args.config.read_text())))
        print(f"Saved ensemble to {root}")
    elif args.cmd == "experiment":
        from .experiment import run_experiment
        from .report import build_report
        exp_dir = run_experiment(yaml.safe_load(args.config.read_text()))
        print(f"Report: {build_report(exp_dir)}")
    elif args.cmd == "report":
        from .report import build_report
        print(f"Report: {build_report(args.exp_dir)}")


if __name__ == "__main__":
    sys.exit(main())
