"""Command line entry point.

    python -m rnn_workbench train configs/train_pdm.yaml
    python -m rnn_workbench fit configs/fit_ss38.yaml
    python -m rnn_workbench mock-events data/stav22/events.csv
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
                        ("fit", "fit networks to a recorded session (data-constrained) from a YAML config"),
                        ("experiment", "run perturbations from a YAML config and write a report")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("config", type=Path)
    p = sub.add_parser("mock-events", help="write a made-up events.csv with a session's layout, to test the pipeline")
    p.add_argument("out", type=Path)
    p.add_argument("--frames-per-run", type=int, nargs="+", default=[27000, 27000])
    p.add_argument("--fs", type=float, default=15.0)
    p.add_argument("--contingency", default="no_reward", choices=["no_reward", "reward"])
    p.add_argument("--inhibition-fraction", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("report", help="rebuild the HTML report for an experiment directory")
    p.add_argument("exp_dir", type=Path)
    args = ap.parse_args(argv)

    if args.cmd == "train":
        from .train import load_train_config, train
        root = train(load_train_config(yaml.safe_load(args.config.read_text())))
        print(f"Saved ensemble to {root}")
    elif args.cmd == "fit":
        from .fit import fit, load_fit_config
        root = fit(load_fit_config(yaml.safe_load(args.config.read_text())))
        print(f"Saved fitted networks to {root}")
    elif args.cmd == "experiment":
        from .experiment import run_experiment
        from .report import build_report
        exp_dir = run_experiment(yaml.safe_load(args.config.read_text()))
        print(f"Report: {build_report(exp_dir)}")
    elif args.cmd == "mock-events":
        from .data import write_mock_events
        rows = write_mock_events(args.out, tuple(args.frames_per_run), args.fs, contingency=args.contingency,
                                 inhibition_fraction=args.inhibition_fraction, seed=args.seed)
        print(f"Wrote {len(rows)} mock trials to {args.out}")
    elif args.cmd == "report":
        from .report import build_report
        print(f"Report: {build_report(args.exp_dir)}")


if __name__ == "__main__":
    sys.exit(main())
