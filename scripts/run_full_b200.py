from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import modal


RUN_ID = os.environ.get("PAGED_PRECISION_RUN_ID")
if not RUN_ID:
    raise RuntimeError("set PAGED_PRECISION_RUN_ID to a unique run name")
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "paged_precision"
RESULT_VOLUME_NAME = f"paged-precision-{RUN_ID}-results"
CACHE_VOLUME_NAME = f"paged-precision-{RUN_ID}-cache"
RESULT_FIELDS = (
    "model",
    "window",
    "method",
    "policy",
    "residual_residency",
    "nll",
    "hbm_mib",
    "dram_mib",
)
SHARDS = (
    {"id": "mistral-00-07", "model": "mistral", "windows": list(range(0, 8))},
    {"id": "mistral-08-15", "model": "mistral", "windows": list(range(8, 16))},
    {"id": "mistral-16-22", "model": "mistral", "windows": list(range(16, 23))},
    {"id": "mistral-23-29", "model": "mistral", "windows": list(range(23, 30))},
    {"id": "llama-00-04", "model": "llama", "windows": list(range(0, 5))},
    {"id": "llama-05-09", "model": "llama", "windows": list(range(5, 10))},
    {"id": "llama-10-14", "model": "llama", "windows": list(range(10, 15))},
    {"id": "llama-15-19", "model": "llama", "windows": list(range(15, 20))},
    {"id": "llama-20-24", "model": "llama", "windows": list(range(20, 25))},
    {"id": "llama-25-29", "model": "llama", "windows": list(range(25, 30))},
)

app = modal.App(f"paged-precision-{RUN_ID}")
result_volume = modal.Volume.from_name(RESULT_VOLUME_NAME, create_if_missing=True)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)
token = os.environ.get("HF_TOKEN")
secrets = [modal.Secret.from_dict({"HF_TOKEN": token})] if token else []

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("torch==2.7.1", index_url="https://download.pytorch.org/whl/cu128")
    .uv_pip_install(
        "transformers==4.50.0",
        "accelerate==1.1.1",
        "datasets==3.1.0",
        "huggingface-hub==0.26.2",
        "safetensors==0.4.5",
        "sentencepiece==0.2.0",
        "pyyaml==6.0.2",
    )
    .env(
        {
            "PYTHONPATH": "/root/paged_precision/src",
            "HF_HOME": "/cache",
            "PAGED_PRECISION_RUN_ID": RUN_ID,
        }
    )
)
image = base_image.add_local_dir(SOURCE, remote_path="/root/paged_precision/src/paged_precision")
worker_image = base_image.env({"HF_HUB_OFFLINE": "1"}).add_local_dir(
    SOURCE,
    remote_path="/root/paged_precision/src/paged_precision",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            {
                "model": row["model"],
                "window": int(row["window"]),
                "method": row["method"],
                "policy": row["policy"] or None,
                "residual_residency": float(row["residual_residency"]) if row["residual_residency"] else None,
                "nll": float(row["nll"]),
                "hbm_mib": float(row["hbm_mib"]),
                "dram_mib": float(row["dram_mib"]),
            }
            for row in csv.DictReader(handle)
        ]


def _payload(row) -> dict:
    return asdict(row)


@app.function(image=image, volumes={"/results": result_volume}, timeout=5 * 60)
def initialize_results(manifest: dict) -> None:
    root = Path("/results")
    if any(root.iterdir()):
        raise RuntimeError(f"refusing to use non-empty volume {RESULT_VOLUME_NAME}")
    _write_json(root / "manifest.json", manifest)
    result_volume.commit()


@app.function(
    image=image,
    volumes={"/cache": cache_volume},
    secrets=secrets,
    cpu=4,
    memory=16384,
    timeout=2 * 60 * 60,
)
def prepare_cache(config) -> dict:
    import os

    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    from paged_precision.experiment import _token_windows

    root = Path("/cache")
    if any(root.iterdir()):
        raise RuntimeError(f"refusing to use non-empty volume {CACHE_VOLUME_NAME}")
    _write_json(root / "run.json", {"run_id": RUN_ID, "created_at": _now()})
    for model in config.models:
        snapshot_download(
            repo_id=model.model_id,
            cache_dir="/cache/hub",
            token=os.environ["HF_TOKEN"],
            allow_patterns=("*.json", "*.model", "*.safetensors", "tokenizer*"),
            ignore_patterns=("original/*", "consolidated.*", "*.pth", "*.pt"),
        )
    records = load_dataset(config.dataset, config.subset, split=config.split, cache_dir="/cache/datasets")
    for model in config.models:
        tokenizer = AutoTokenizer.from_pretrained(model.model_id, use_fast=True, local_files_only=True)
        _write_json(root / "windows" / f"{model.name}.json", {"windows": _token_windows(tokenizer, records, config)})
    cache_volume.commit()
    return {"volume": CACHE_VOLUME_NAME, "prepared_at": _now()}


@app.function(
    image=worker_image,
    gpu="B200",
    volumes={"/cache": cache_volume.read_only(), "/results": result_volume},
    secrets=secrets,
    timeout=12 * 60 * 60,
    max_containers=10,
)
def evaluate_shard(config, shard: dict, attempt: int) -> dict:
    import torch
    from transformers import AutoModelForCausalLM

    from paged_precision.experiment import _evaluate_bf16, _evaluate_direct, _evaluate_paged_precision
    from paged_precision.quantization import DirectKVQuantizerPair, KVQuantizerPair

    shard_id = shard["id"]
    started = time.perf_counter()
    _write_json(
        Path(f"/results/status/{shard_id}.attempt-{attempt}.started.json"),
        {"run_id": RUN_ID, "shard": shard_id, "attempt": attempt, "started_at": _now()},
    )
    result_volume.commit()
    try:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        model_spec = next(model for model in config.models if model.name == shard["model"])
        model = AutoModelForCausalLM.from_pretrained(
            model_spec.model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="sdpa",
            local_files_only=True,
        ).eval()
        windows = json.loads(Path(f"/cache/windows/{model_spec.name}.json").read_text(encoding="utf-8"))["windows"]
        layers = int(model.config.num_hidden_layers)
        head_dim = int(model.config.hidden_size) // int(model.config.num_attention_heads)
        nested = [KVQuantizerPair.for_head_dim(head_dim, seed=config.seed + 97 * layer) for layer in range(layers)]
        direct = {
            bits: [
                DirectKVQuantizerPair.for_head_dim(head_dim, bits=bits, seed=config.seed + 97 * layer)
                for layer in range(layers)
            ]
            for bits in (2, 3, 4)
        }

        all_rows = []
        for window_index in shard["windows"]:
            path = Path(f"/results/windows/{model_spec.name}/{window_index:02d}.csv")
            if path.exists():
                rows = _read_rows(path)
            else:
                token_ids = windows[window_index]
                evaluated = [
                    _evaluate_bf16(model, model_spec.name, window_index, token_ids, config),
                    _evaluate_direct(model, model_spec.name, window_index, token_ids, config, direct[2], 2),
                    _evaluate_paged_precision(model, model_spec.name, window_index, token_ids, config, nested, 0.0, None),
                ]
                for fraction in config.residual_residency[1:-1]:
                    for policy in config.policies:
                        evaluated.append(
                            _evaluate_paged_precision(
                                model,
                                model_spec.name,
                                window_index,
                                token_ids,
                                config,
                                nested,
                                fraction,
                                policy,
                            )
                        )
                evaluated.extend(
                    (
                        _evaluate_direct(model, model_spec.name, window_index, token_ids, config, direct[3], 3),
                        _evaluate_direct(model, model_spec.name, window_index, token_ids, config, direct[4], 4),
                    )
                )
                rows = [_payload(row) for row in evaluated]
                if len(rows) != 14:
                    raise RuntimeError(f"window {window_index} produced {len(rows)} rows")
                _write_rows(path, rows)
                _write_json(
                    Path(f"/results/progress/{shard_id}/{window_index:02d}.json"),
                    {
                        "run_id": RUN_ID,
                        "shard": shard_id,
                        "window": window_index,
                        "rows": len(rows),
                        "completed_at": _now(),
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                )
                result_volume.commit()
                print(f"{shard_id}: completed window {window_index}", flush=True)
            all_rows.extend(rows)

        summary = {
            "run_id": RUN_ID,
            "shard": shard_id,
            "attempt": attempt,
            "windows": shard["windows"],
            "rows": len(all_rows),
            "completed_at": _now(),
            "elapsed_seconds": time.perf_counter() - started,
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        }
        _write_json(Path(f"/results/status/{shard_id}.attempt-{attempt}.complete.json"), summary)
        result_volume.commit()
        return {"summary": summary, "rows": all_rows}
    except Exception as error:
        _write_json(
            Path(f"/results/status/{shard_id}.attempt-{attempt}.failed.json"),
            {
                "run_id": RUN_ID,
                "shard": shard_id,
                "attempt": attempt,
                "failed_at": _now(),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        result_volume.commit()
        raise


def _validate(rows: list[dict], config) -> None:
    from paged_precision.experiment import experiment_matrix

    expected = {
        (row["model"], row["window"], row["method"], row["policy"], row["residual_residency"])
        for row in experiment_matrix(config)
    }
    actual = {
        (row["model"], row["window"], row["method"], row["policy"], row["residual_residency"])
        for row in rows
    }
    if len(rows) != 840 or actual != expected:
        raise RuntimeError(f"expected 840 exact result coordinates, received {len(rows)} rows and {len(actual)} coordinates")
    if any(not math.isfinite(float(row[field])) for row in rows for field in ("nll", "hbm_mib", "dram_mib")):
        raise RuntimeError("results contain a non-finite measurement")


@app.function(image=image, volumes={"/results": result_volume}, timeout=24 * 60 * 60)
def supervise(config, manifest: dict) -> dict:
    initialize_results.remote(manifest)
    result_volume.reload()
    cache = prepare_cache.remote(config)
    calls = [(shard, 1, evaluate_shard.spawn(config, shard, 1)) for shard in SHARDS]
    _write_json(
        Path("/results/calls.json"),
        {
            "run_id": RUN_ID,
            "created_at": _now(),
            "calls": [
                {
                    "shard": shard["id"],
                    "attempt": attempt,
                    "call_id": call.object_id,
                    "url": call.get_dashboard_url(),
                }
                for shard, attempt, call in calls
            ],
        },
    )
    result_volume.commit()

    results = []
    failures = []
    for shard, attempt, call in calls:
        try:
            results.append(call.get())
        except Exception as first_error:
            retry = evaluate_shard.spawn(config, shard, 2)
            try:
                results.append(retry.get())
            except Exception as second_error:
                failures.append(
                    {
                        "shard": shard["id"],
                        "first": f"{type(first_error).__name__}: {first_error}",
                        "second": f"{type(second_error).__name__}: {second_error}",
                    }
                )

    rows = [row for result in results for row in result["rows"]]
    result_volume.reload()
    status = "failed"
    if not failures:
        _validate(rows, config)
        _write_rows(Path("/results/experiment.csv"), rows)
        status = "complete"
    summary = {
        "run_id": RUN_ID,
        "status": status,
        "finished_at": _now(),
        "rows": len(rows),
        "completed_shards": [result["summary"] for result in results],
        "failures": failures,
        "cache": cache,
    }
    _write_json(Path("/results/supervisor.complete.json"), summary)
    result_volume.commit()
    return summary


@app.local_entrypoint()
def main() -> None:
    from paged_precision.experiment import ExperimentConfig, experiment_matrix

    config = ExperimentConfig.load(ROOT / "experiment.yaml")
    if not token:
        raise RuntimeError("a Hugging Face token is required for Llama")
    matrix = experiment_matrix(config)
    arms = sorted(
        {
            (row["method"], row["policy"], row["residual_residency"])
            for row in matrix
        },
        key=str,
    )
    manifest = {
        "run_id": RUN_ID,
        "created_at": _now(),
        "gpu": "B200",
        "torch": "2.7.1+cu128",
        "cuda": "12.8",
        "seed": config.seed,
        "models": [asdict(model) for model in config.models],
        "dataset": {"name": config.dataset, "subset": config.subset, "split": config.split},
        "context_tokens": config.context_tokens,
        "score_tokens": config.score_tokens,
        "windows_per_model": config.windows,
        "arms": arms,
        "expected_rows": len(matrix),
        "result_volume": RESULT_VOLUME_NAME,
        "cache_volume": CACHE_VOLUME_NAME,
        "shards": list(SHARDS),
    }
    supervisor = supervise.spawn(config, manifest)
    print(
        json.dumps(
            {
                "run_id": RUN_ID,
                "result_volume": RESULT_VOLUME_NAME,
                "cache_volume": CACHE_VOLUME_NAME,
                "supervisor_call_id": supervisor.object_id,
                "supervisor_url": supervisor.get_dashboard_url(),
            },
            indent=2,
            sort_keys=True,
        )
    )
