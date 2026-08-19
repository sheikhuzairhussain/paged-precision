from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from paged_precision.policies import ResidencyObservation, ResidencyPolicy
from paged_precision.quantization import AdditivePrecisionQuantizer
from paged_precision.runtime.cache import PagedPrecisionCache


DECODE_CHUNK_BLOCKS = 512


@dataclass(frozen=True)
class PackedAttentionOutput:
    values: torch.Tensor
    attention_mass: torch.Tensor | None
    peak_scratch_bytes: int


def packed_decode_attention(
    cache: PagedPrecisionCache,
    query: torch.Tensor,
    *,
    layer: int,
    key_quantizer: AdditivePrecisionQuantizer,
    value_quantizer: AdditivePrecisionQuantizer,
    policy: ResidencyPolicy | None = None,
    previous_attention_mass: torch.Tensor | None = None,
    current_key: torch.Tensor | None = None,
    current_value: torch.Tensor | None = None,
) -> PackedAttentionOutput:
    """Decode bounded packed chunks and apply the resident residuals."""

    if query.ndim != 3 or query.shape[0] != cache.layout.batch_size or query.shape[-1] != cache.layout.head_dim:
        raise ValueError("query must have shape [batch, query_heads, head_dim]")
    if (current_key is None) != (current_value is None):
        raise ValueError("current_key and current_value must be provided together")
    visible = cache.available_blocks()
    if visible == 0 and current_key is None:
        raise ValueError("cannot attend to an empty cache")

    if policy is not None:
        transition = policy.observe(
            ResidencyObservation(
                visible_blocks=visible,
                hot_blocks=cache.block_to_slot[:visible] >= 0,
                previous_attention_mass=previous_attention_mass,
                hot_capacity=cache.layout.hot_slots,
            )
        )
        cache.set_hot_blocks(list(transition.selected), retention_mode=policy.retention_mode)

    batch, query_heads, head_dim = query.shape
    running_max = torch.full((batch, query_heads), -torch.inf, dtype=torch.float32, device=query.device)
    denominator = torch.zeros_like(running_max)
    numerator = torch.zeros((batch, query_heads, head_dim), dtype=torch.float32, device=query.device)
    scale = 1 / math.sqrt(head_dim)
    peak_scratch = 0
    block_masses: list[torch.Tensor] = []

    for start_block in range(0, visible, DECODE_CHUNK_BLOCKS):
        end_block = min(start_block + DECODE_CHUNK_BLOCKS, visible)
        keys = cache.decode_rotated_blocks(
            start_block,
            end_block,
            layer,
            "key",
            key_quantizer,
            dtype=query.dtype,
        )
        values = cache.decode_rotated_blocks(
            start_block,
            end_block,
            layer,
            "value",
            value_quantizer,
            dtype=query.dtype,
        )
        rotated_query = query @ key_quantizer.rotation.to(query.device, query.dtype)
        logits = _grouped_logits(rotated_query, keys).float() * scale
        token_counts = [int(value) for value in cache.valid_tokens[start_block:end_block].tolist()]
        block_masses.append(_block_log_mass(logits, token_counts, cache.layout.block_size))
        block_max = logits.amax(dim=-1)
        new_max = torch.maximum(running_max, block_max)
        previous_scale = torch.exp(running_max - new_max)
        block_weights = torch.exp(logits - new_max.unsqueeze(-1))
        denominator = denominator * previous_scale + block_weights.sum(dim=-1)
        rotated_values = _grouped_weighted_values(block_weights, values.float())
        value_rotation = value_quantizer.rotation.to(rotated_values.device, rotated_values.dtype)
        numerator = numerator * previous_scale.unsqueeze(-1) + rotated_values @ value_rotation.T
        running_max = new_max
        peak_scratch = max(
            peak_scratch,
            sum(
                tensor.numel() * tensor.element_size()
                for tensor in (
                    keys,
                    values,
                    logits,
                    block_weights,
                )
                if tensor is not None
            ),
        )

    if current_key is not None and current_value is not None:
        if current_key.ndim == 4:
            current_key = current_key[..., 0, :]
            current_value = current_value[..., 0, :]
        keys = current_key.unsqueeze(-2)
        values = current_value.unsqueeze(-2)
        logits = _grouped_logits(query, keys).float() * scale
        block_max = logits.amax(dim=-1)
        new_max = torch.maximum(running_max, block_max)
        previous_scale = torch.exp(running_max - new_max)
        block_weights = torch.exp(logits - new_max.unsqueeze(-1))
        denominator = denominator * previous_scale + block_weights.sum(dim=-1)
        numerator = numerator * previous_scale.unsqueeze(-1) + _grouped_weighted_values(
            block_weights,
            values.float(),
        )
        peak_scratch = max(
            peak_scratch,
            sum(tensor.numel() * tensor.element_size() for tensor in (keys, values, logits, block_weights)),
        )

    attention_mass = None
    if block_masses:
        attention_mass = torch.softmax(torch.cat(block_masses, dim=-1), dim=-1).mean(dim=(0, 1))
    return PackedAttentionOutput(
        values=(numerator / denominator.clamp_min(1e-30).unsqueeze(-1)).to(query.dtype),
        attention_mass=attention_mass,
        peak_scratch_bytes=peak_scratch,
    )


def _block_log_mass(logits: torch.Tensor, token_counts: list[int], block_size: int) -> torch.Tensor:
    if token_counts[-1] < block_size:
        logits = torch.nn.functional.pad(logits, (0, block_size - token_counts[-1]), value=-torch.inf)
    return torch.logsumexp(
        logits.float().reshape(*logits.shape[:-1], len(token_counts), block_size),
        dim=-1,
    )


def _grouped_logits(query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    batch, query_heads, head_dim = query.shape
    kv_heads = keys.shape[1]
    if query_heads % kv_heads:
        raise ValueError("query head count must be divisible by KV head count")
    groups = query_heads // kv_heads
    grouped_query = query.reshape(batch, kv_heads, groups, head_dim)
    return torch.einsum("bkgd,bktd->bkgt", grouped_query, keys).reshape(batch, query_heads, keys.shape[-2])


def _grouped_weighted_values(weights: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    batch, query_heads, token_count = weights.shape
    kv_heads = values.shape[1]
    groups = query_heads // kv_heads
    grouped_weights = weights.reshape(batch, kv_heads, groups, token_count)
    return torch.einsum("bkgt,bktd->bkgd", grouped_weights, values).reshape(
        batch,
        query_heads,
        values.shape[-1],
    )
