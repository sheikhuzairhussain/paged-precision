from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path

from paged_precision.experiment import ExperimentConfig, experiment_matrix, run_on_modal, write_results


ROOT = Path(__file__).resolve().parents[2]


def _run_id(value: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", value):
        raise argparse.ArgumentTypeError(
            "use 1-63 lowercase letters, digits, or hyphens, starting with a letter or digit"
        )
    return value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Paged Precision experiment.")
    parser.add_argument("--config", type=Path, default=ROOT / "experiment.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = ExperimentConfig.load(args.config)
    matrix = experiment_matrix(config)
    arms = len(matrix) // (len(config.models) * config.windows)
    print(f"{len(config.models)} models, {config.windows} windows, {arms} arms, {len(matrix)} rows")
    if args.dry_run:
        for row in matrix:
            fraction = "" if row["residual_residency"] is None else row["residual_residency"]
            policy = row["policy"] or ""
            print(f"{row['model']},{row['window']},{row['method']},{policy},{fraction}")
        return
    rows = run_on_modal(config)
    destination = ROOT / "results" / "experiment.csv"
    write_results(destination, rows)
    print(destination)


def full_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Launch the full Paged Precision suite on 10 Modal B200 workers.")
    parser.add_argument("--run-id", required=True, type=_run_id, help="unique name used for the Modal app and volumes")
    parser.add_argument("--foreground", action="store_true", help="wait for the run instead of launching it detached")
    args = parser.parse_args(argv)

    environment = os.environ.copy()
    environment["PAGED_PRECISION_RUN_ID"] = args.run_id
    command = ["modal", "run"]
    if not args.foreground:
        command.append("--detach")
    command.append(str(ROOT / "scripts" / "run_full_b200.py"))
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
