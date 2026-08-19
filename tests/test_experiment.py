from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess

from paged_precision.cli import full_main
from paged_precision.experiment import (
    ExperimentConfig,
    ExperimentRow,
    experiment_matrix,
    read_results,
    run_experiment,
    write_results,
)


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_experiment_matrix_has_840_rows() -> None:
    config = ExperimentConfig.load(ROOT / "experiment.yaml")
    matrix = experiment_matrix(config)

    assert len(matrix) == 840
    assert {row["method"] for row in matrix} == {"bf16", "tq2", "paged_precision", "tq3", "tq4"}
    assert {row["policy"] for row in matrix} == {None, "sink", "recent", "attention_ema"}


def test_experiment_interface_validates_and_round_trips_results(tmp_path) -> None:
    config = ExperimentConfig.load(ROOT / "experiment.yaml")

    def fake_model(_config, model):
        rows = []
        for row in experiment_matrix(config):
            if row["model"] != model.name:
                continue
            rows.append(
                ExperimentRow(
                    model=model.name,
                    window=row["window"],
                    method=row["method"],
                    policy=row["policy"],
                    residual_residency=row["residual_residency"],
                    nll=1.0,
                    hbm_mib=100.0,
                    dram_mib=10.0,
                )
            )
        return rows

    rows = run_experiment(config, fake_model)
    output = tmp_path / "experiment.csv"
    write_results(output, rows)

    assert read_results(output) == rows
    assert b"\r" not in output.read_bytes()


def test_full_command_launches_detached_run(monkeypatch) -> None:
    captured = {}

    def fake_run(command, *, cwd, env, check):
        captured.update(command=command, cwd=cwd, env=env, check=check)
        return CompletedProcess(command, 0)

    monkeypatch.setattr("paged_precision.cli.subprocess.run", fake_run)
    full_main(["--run-id", "dissertation-reproduction"])

    assert captured["command"] == [
        "modal",
        "run",
        "--detach",
        str(ROOT / "scripts" / "run_full_b200.py"),
    ]
    assert captured["cwd"] == ROOT
    assert captured["env"]["PAGED_PRECISION_RUN_ID"] == "dissertation-reproduction"
    assert captured["check"] is True
