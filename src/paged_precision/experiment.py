from __future__ import annotations

import csv
import gc
import math
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
import yaml

from paged_precision.policies import AttentionEMAPolicy, RecentPolicy, SinkPolicy
from paged_precision.quantization import DirectKVQuantizerPair, KVQuantizerPair
from paged_precision.runtime import DirectKVModelAdapter, PagedPrecisionModelAdapter, extract_legacy_cache


MIB = 1024 * 1024
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


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_id: str


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    models: tuple[ModelSpec, ...]
    dataset: str
    subset: str
    split: str
    context_tokens: int
    score_tokens: int
    windows: int
    gpu: str
    residual_residency: tuple[float, ...]
    policies: tuple[str, ...]

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        models = tuple(ModelSpec(name=str(row["name"]), model_id=str(row["model_id"])) for row in raw["models"])
        config = cls(
            seed=int(raw["seed"]),
            models=models,
            dataset=str(raw["dataset"]),
            subset=str(raw["subset"]),
            split=str(raw["split"]),
            context_tokens=int(raw["context_tokens"]),
            score_tokens=int(raw["score_tokens"]),
            windows=int(raw["windows"]),
            gpu=str(raw["gpu"]),
            residual_residency=tuple(float(value) for value in raw["residual_residency"]),
            policies=tuple(str(value) for value in raw["policies"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if len(self.models) != 2 or {model.name for model in self.models} != {"mistral", "llama"}:
            raise ValueError("the dissertation experiment requires Mistral and Llama")
        if self.context_tokens <= 0 or self.score_tokens <= 0 or self.windows <= 0:
            raise ValueError("token counts and windows must be positive")
        if self.gpu.upper() != "B200":
            raise ValueError("the experiment is defined for a B200")
        if self.residual_residency != (0.0, 0.1, 0.25, 0.5, 1.0):
            raise ValueError("residual_residency must be [0, 0.1, 0.25, 0.5, 1]")
        if self.policies != ("sink", "recent", "attention_ema"):
            raise ValueError("policies must be [sink, recent, attention_ema]")


@dataclass(frozen=True)
class ExperimentRow:
    model: str
    window: int
    method: str
    policy: str | None
    residual_residency: float | None
    nll: float
    hbm_mib: float
    dram_mib: float


def experiment_matrix(config: ExperimentConfig) -> list[dict[str, Any]]:
    arms: list[tuple[str, str | None, float | None]] = [
        ("bf16", None, None),
        ("tq2", None, None),
        ("paged_precision", None, 0.0),
    ]
    arms.extend(
        ("paged_precision", policy, fraction)
        for fraction in config.residual_residency[1:-1]
        for policy in config.policies
    )
    arms.extend((("tq3", None, None), ("tq4", None, 1.0)))
    return [
        {
            "model": model.name,
            "window": window,
            "method": method,
            "policy": policy,
            "residual_residency": fraction,
        }
        for model in config.models
        for window in range(config.windows)
        for method, policy, fraction in arms
    ]


def run_experiment(
    config: ExperimentConfig,
    evaluate_model: Callable[[ExperimentConfig, ModelSpec], list[ExperimentRow]],
) -> list[ExperimentRow]:
    rows = [row for model in config.models for row in evaluate_model(config, model)]
    expected = len(experiment_matrix(config))
    if len(rows) != expected:
        raise ValueError(f"expected {expected} result rows, received {len(rows)}")
    keys = {
        (row.model, row.window, row.method, row.policy, row.residual_residency)
        for row in rows
    }
    expected_keys = {
        (row["model"], row["window"], row["method"], row["policy"], row["residual_residency"])
        for row in experiment_matrix(config)
    }
    if keys != expected_keys:
        raise ValueError("result rows do not match the experiment matrix")
    if any(not math.isfinite(value) for row in rows for value in (row.nll, row.hbm_mib, row.dram_mib)):
        raise ValueError("result rows must contain finite measurements")
    return rows


def write_results(path: str | Path, rows: list[ExperimentRow]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=destination.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            payload = asdict(row)
            payload["residual_residency"] = "" if row.residual_residency is None else row.residual_residency
            writer.writerow(payload)
        temporary = handle.name
    os.replace(temporary, destination)


def read_results(path: str | Path) -> list[ExperimentRow]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [
            ExperimentRow(
                model=row["model"],
                window=int(row["window"]),
                method=row["method"],
                policy=row["policy"] or None,
                residual_residency=(float(row["residual_residency"]) if row["residual_residency"] else None),
                nll=float(row["nll"]),
                hbm_mib=float(row["hbm_mib"]),
                dram_mib=float(row["dram_mib"]),
            )
            for row in csv.DictReader(handle)
        ]


def evaluate_model_on_gpu(config: ExperimentConfig, model_spec: ModelSpec) -> list[ExperimentRow]:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_spec.model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_spec.model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    ).eval()
    records = load_dataset(config.dataset, config.subset, split=config.split)
    windows = _token_windows(tokenizer, records, config)
    layers = int(model.config.num_hidden_layers)
    head_dim = int(model.config.hidden_size) // int(model.config.num_attention_heads)
    quantizers = [
        KVQuantizerPair.for_head_dim(head_dim, seed=config.seed + 97 * layer)
        for layer in range(layers)
    ]
    direct_quantizers = {
        bits: [
            DirectKVQuantizerPair.for_head_dim(head_dim, bits=bits, seed=config.seed + 97 * layer)
            for layer in range(layers)
        ]
        for bits in (2, 3, 4)
    }

    rows: list[ExperimentRow] = []
    for window_index, token_ids in enumerate(windows):
        rows.append(_evaluate_bf16(model, model_spec.name, window_index, token_ids, config))
        rows.append(
            _evaluate_direct(model, model_spec.name, window_index, token_ids, config, direct_quantizers[2], 2)
        )
        rows.append(
            _evaluate_paged_precision(
                model,
                model_spec.name,
                window_index,
                token_ids,
                config,
                quantizers,
                0.0,
                None,
            )
        )
        for fraction in config.residual_residency[1:-1]:
            for policy in config.policies:
                rows.append(
                    _evaluate_paged_precision(
                        model,
                        model_spec.name,
                        window_index,
                        token_ids,
                        config,
                        quantizers,
                        fraction,
                        policy,
                    )
                )
        rows.append(
            _evaluate_direct(model, model_spec.name, window_index, token_ids, config, direct_quantizers[3], 3)
        )
        rows.append(
            _evaluate_direct(model, model_spec.name, window_index, token_ids, config, direct_quantizers[4], 4)
        )
    return rows


def _token_windows(tokenizer, records, config: ExperimentConfig) -> list[list[int]]:
    stream: list[int] = []
    eos = tokenizer.eos_token_id
    for record in records:
        text = str(record.get("text", ""))
        if not text.strip():
            continue
        stream.extend(tokenizer.encode(text, add_special_tokens=False))
        if eos is not None:
            stream.append(eos)
    length = config.context_tokens + config.score_tokens
    required = length * config.windows
    if len(stream) < required:
        raise ValueError(f"dataset produced {len(stream)} tokens, but {required} are required")
    return [stream[index * length : (index + 1) * length] for index in range(config.windows)]


def _evaluate_bf16(model, model_name: str, window: int, token_ids: list[int], config: ExperimentConfig) -> ExperimentRow:
    device = next(model.parameters()).device
    prefix = torch.tensor([token_ids[: config.context_tokens]], dtype=torch.long, device=device)
    targets = token_ids[config.context_tokens :]
    with torch.no_grad():
        outputs = model(input_ids=prefix, use_cache=True, return_dict=True)
    logits = outputs.logits[:, -1]
    cache = outputs.past_key_values
    losses = []
    for target in targets:
        losses.append(float(F.cross_entropy(logits.float(), torch.tensor([target], device=device)).item()))
        with torch.no_grad():
            outputs = model(
                input_ids=torch.tensor([[target]], dtype=torch.long, device=device),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
        logits = outputs.logits[:, -1]
        cache = outputs.past_key_values
    hbm = sum(tensor.numel() * tensor.element_size() for layer in extract_legacy_cache(cache) for tensor in layer)
    row = ExperimentRow(model_name, window, "bf16", None, None, sum(losses) / len(losses), hbm / MIB, 0.0)
    del outputs, logits, cache, prefix
    _release_cuda()
    return row


def _evaluate_direct(
    model,
    model_name: str,
    window: int,
    token_ids: list[int],
    config: ExperimentConfig,
    quantizers: list[DirectKVQuantizerPair],
    bits: int,
) -> ExperimentRow:
    device = next(model.parameters()).device
    prefix = torch.tensor([token_ids[: config.context_tokens]], dtype=torch.long, device=device)
    build = DirectKVModelAdapter.prefill(
        model,
        prefix,
        max_cache_tokens=config.context_tokens + config.score_tokens,
        quantizers=quantizers,
    )
    adapter = build.adapter
    logits = build.last_logits
    losses = []
    for target in token_ids[config.context_tokens :]:
        losses.append(float(F.cross_entropy(logits.float(), torch.tensor([target], device=device)).item()))
        logits = adapter.decode(torch.tensor([[target]], dtype=torch.long, device=device))
    adapter.cache.synchronize_transfers()
    memory = adapter.cache.memory_snapshot()
    row = ExperimentRow(
        model_name,
        window,
        f"tq{bits}",
        None,
        1.0 if bits == 4 else None,
        sum(losses) / len(losses),
        (memory.cache_hbm_bytes + adapter.quantizer_hbm_bytes()) / MIB,
        0.0,
    )
    del adapter, build, logits, prefix
    _release_cuda()
    return row


def _evaluate_paged_precision(
    model,
    model_name: str,
    window: int,
    token_ids: list[int],
    config: ExperimentConfig,
    quantizers: list[KVQuantizerPair],
    fraction: float,
    policy_name: str | None,
) -> ExperimentRow:
    device = next(model.parameters()).device
    prefix = torch.tensor([token_ids[: config.context_tokens]], dtype=torch.long, device=device)
    policy = _policy(policy_name, fraction)
    build = PagedPrecisionModelAdapter.prefill(
        model,
        prefix,
        max_cache_tokens=config.context_tokens + config.score_tokens,
        residual_fraction=fraction,
        quantizers=quantizers,
        policy=policy,
        initial_hot_blocks=_initial_hot_blocks(config, fraction, policy_name),
    )
    adapter = build.adapter
    logits = build.last_logits
    losses = []
    for target in token_ids[config.context_tokens :]:
        losses.append(float(F.cross_entropy(logits.float(), torch.tensor([target], device=device)).item()))
        logits = adapter.decode(torch.tensor([[target]], dtype=torch.long, device=device))
    adapter.cache.synchronize_transfers()
    memory = adapter.cache.memory_snapshot()
    row = ExperimentRow(
        model_name,
        window,
        "paged_precision",
        policy_name,
        fraction,
        sum(losses) / len(losses),
        (memory.cache_hbm_bytes + adapter.quantizer_hbm_bytes()) / MIB,
        memory.pinned_dram_bytes / MIB,
    )
    del adapter, build, logits, prefix
    _release_cuda()
    return row


def _policy(name: str | None, fraction: float):
    if not 0 < fraction < 1:
        return None
    policies = {
        "sink": SinkPolicy,
        "recent": RecentPolicy,
        "attention_ema": AttentionEMAPolicy,
    }
    try:
        return policies[name](fraction)
    except KeyError as error:
        raise ValueError(f"unknown residency policy: {name}") from error


def _initial_hot_blocks(
    config: ExperimentConfig,
    fraction: float,
    policy_name: str | None,
) -> list[int] | None:
    if fraction >= 1:
        return None
    if fraction <= 0:
        return []
    block_size = 32
    max_blocks = math.ceil((config.context_tokens + config.score_tokens) / block_size)
    visible = math.ceil(config.context_tokens / block_size)
    capacity = min(visible, math.ceil(max_blocks * fraction))
    if policy_name == "sink":
        return list(range(capacity))
    if policy_name == "recent":
        return list(range(visible - capacity, visible))
    selected = {0} | set(range(max(0, visible - 4), visible))
    for block in range(visible - 1, -1, -1):
        selected.add(block)
        if len(selected) >= capacity:
            break
    return sorted(selected)


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


try:
    import modal
except ImportError:  # pragma: no cover
    modal = None


if modal is not None:
    try:
        from huggingface_hub import get_token
    except ImportError:  # Local CPU checks do not need a Hugging Face token.
        local_token = None
    else:
        local_token = get_token()

    app = modal.App("paged-precision")
    cache_volume = modal.Volume.from_name("paged-precision-cache", create_if_missing=True)
    secrets = [modal.Secret.from_dict({"HF_TOKEN": local_token})] if local_token else []
    source = Path(__file__).resolve().parent
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .uv_pip_install(
            "torch==2.7.1",
            index_url="https://download.pytorch.org/whl/cu128",
        )
        .uv_pip_install(
            "transformers==4.50.0",
            "accelerate==1.1.1",
            "datasets==3.1.0",
            "huggingface-hub==0.26.2",
            "safetensors==0.4.5",
            "sentencepiece==0.2.0",
            "pyyaml==6.0.2",
        )
        .env({"PYTHONPATH": "/root/paged_precision/src", "HF_HOME": "/cache"})
        .add_local_dir(source, remote_path="/root/paged_precision/src/paged_precision")
    )

    @app.function(
        image=image,
        gpu="B200",
        volumes={"/cache": cache_volume},
        secrets=secrets,
        timeout=12 * 60 * 60,
    )
    def evaluate_model_remote(config: ExperimentConfig, model_spec: ModelSpec) -> list[ExperimentRow]:
        return evaluate_model_on_gpu(config, model_spec)
else:  # pragma: no cover
    app = None


def run_on_modal(config: ExperimentConfig) -> list[ExperimentRow]:
    if app is None:
        raise RuntimeError("Modal is not installed")
    with app.run():
        calls = {model.name: evaluate_model_remote.spawn(config, model) for model in config.models}
        return run_experiment(config, lambda _config, model: calls[model.name].get())
