from __future__ import annotations

import gc
import inspect
import types
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import torch

from paged_precision.policies import ResidencyPolicy
from paged_precision.quantization import DirectKVQuantizerPair, KVQuantizerPair
from paged_precision.runtime.attention import packed_decode_attention
from paged_precision.runtime.cache import PackedCacheLayout, PagedPrecisionCache
from paged_precision.runtime.direct_cache import DirectPackedCacheLayout, PackedDirectKVCache


SUPPORTED_MODEL_TYPES = {"llama", "mistral"}


def extract_legacy_cache(past_key_values: Any) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    if past_key_values is None:
        raise ValueError("model did not return a KV cache")
    if hasattr(past_key_values, "to_legacy_cache"):
        past_key_values = past_key_values.to_legacy_cache()
    if isinstance(past_key_values, (tuple, list)):
        return tuple((layer[0], layer[1]) for layer in past_key_values)
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return tuple(zip(past_key_values.key_cache, past_key_values.value_cache))
    layers = getattr(past_key_values, "layers", None)
    if layers is not None:
        result = []
        for layer in layers:
            keys = getattr(layer, "keys", getattr(layer, "key_cache", None))
            values = getattr(layer, "values", getattr(layer, "value_cache", None))
            if keys is None or values is None:
                raise TypeError("unrecognised Hugging Face cache layer")
            result.append((keys, values))
        return tuple(result)
    raise TypeError(f"unsupported Hugging Face cache type: {type(past_key_values).__name__}")


def validate_supported_model(model) -> str:
    model_type = str(getattr(model.config, "model_type", ""))
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"unsupported model type {model_type!r}; expected Llama or Mistral")
    for index, layer in enumerate(_model_layers(model)):
        attention = getattr(layer, "self_attn", None)
        required = ("q_proj", "k_proj", "v_proj", "o_proj", "head_dim")
        if attention is None or any(not hasattr(attention, name) for name in required):
            raise ValueError(f"layer {index} does not expose the expected attention projections")
    return model_type


@dataclass(frozen=True)
class PrefillBuildResult:
    adapter: "PagedPrecisionModelAdapter"
    last_logits: torch.Tensor


class _AttentionPatch:
    def __init__(self, adapter: "PagedPrecisionModelAdapter") -> None:
        self.adapter = adapter
        self.originals: list[tuple[Any, Any]] = []

    def __enter__(self):
        for layer_index, layer in enumerate(_model_layers(self.adapter.model)):
            attention = layer.self_attn
            self.originals.append((attention, attention.forward))
            signature = inspect.signature(attention.forward)
            legacy = "position_ids" in signature.parameters and "position_embeddings" not in signature.parameters

            def patched(module, hidden_states, *args, _layer=layer_index, _signature=signature, **kwargs):
                bound = _signature.bind_partial(hidden_states, *args, **kwargs)
                call_kwargs = dict(bound.arguments)
                call_kwargs.pop("hidden_states", None)
                variadic = call_kwargs.pop("kwargs", None)
                if isinstance(variadic, dict):
                    call_kwargs.update(variadic)
                position_embeddings = call_kwargs.pop("position_embeddings", None)
                output, weights = self.adapter.attention_forward(
                    _layer,
                    module,
                    hidden_states,
                    position_embeddings=position_embeddings,
                    **call_kwargs,
                )
                return (output, weights, None) if legacy else (output, weights)

            attention.forward = types.MethodType(patched, attention)
        return self.adapter

    def __exit__(self, exc_type, exc_value, traceback):
        for attention, original in self.originals:
            attention.forward = original
        self.originals.clear()


class PagedPrecisionModelAdapter:
    """Run Llama or Mistral decode directly from the packed Paged Precision cache."""

    def __init__(
        self,
        model,
        cache: PagedPrecisionCache,
        quantizers: list[KVQuantizerPair],
        *,
        token_position: int,
        policy: ResidencyPolicy | None,
    ) -> None:
        validate_supported_model(model)
        if len(quantizers) != cache.layout.layers:
            raise ValueError("one key/value quantizer pair is required per layer")
        self.model = model
        self.cache = cache
        self.quantizers = quantizers
        self.token_position = token_position
        self.policy = policy
        self.last_attention_mass: list[torch.Tensor | None] = [None] * cache.layout.layers

    @classmethod
    def prefill(
        cls,
        model,
        input_ids: torch.Tensor,
        *,
        max_cache_tokens: int,
        residual_fraction: float,
        quantizers: list[KVQuantizerPair],
        block_size: int = 32,
        policy: ResidencyPolicy | None = None,
        initial_hot_blocks: list[int] | None = None,
    ) -> PrefillBuildResult:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, tokens]")
        if input_ids.device.type == "cuda":
            torch.cuda.synchronize(input_ids.device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=True, return_dict=True)
        legacy = extract_legacy_cache(outputs.past_key_values)
        if not legacy:
            raise ValueError("prefill cache is empty")
        layers = len(legacy)
        batch, kv_heads, token_count, head_dim = legacy[0][0].shape
        device = legacy[0][0].device
        dtype = legacy[0][0].dtype
        resident_quantizers = [pair.to(device, dtype=dtype) for pair in quantizers]
        layout = PackedCacheLayout(
            layers=layers,
            batch_size=batch,
            kv_heads=kv_heads,
            head_dim=head_dim,
            max_tokens=max_cache_tokens,
            hot_fraction=residual_fraction,
            block_size=block_size,
            norm_dtype=dtype,
        )
        cache = PagedPrecisionCache(layout, device=device)
        retention_mode = policy.retention_mode if policy is not None else "recoverable"
        cache.prepare_prefill(
            token_count,
            hot_block_ids=initial_hot_blocks,
            retention_mode=retention_mode,
        )
        for layer, ((keys, values), pair) in enumerate(zip(legacy, resident_quantizers)):
            cache.store_prefill_layer(layer, pair.key.encode(keys), pair.value.encode(values))
        cache.synchronize_transfers()
        last_logits = outputs.logits[:, -1].detach().clone()
        del outputs, legacy
        gc.collect()
        if input_ids.device.type == "cuda":
            torch.cuda.empty_cache()
        return PrefillBuildResult(
            cls(
                model,
                cache,
                resident_quantizers,
                token_position=token_count,
                policy=policy,
            ),
            last_logits,
        )

    @contextmanager
    def patched(self) -> Iterator["PagedPrecisionModelAdapter"]:
        with _AttentionPatch(self):
            yield self

    def attention_forward(
        self,
        layer: int,
        module,
        hidden_states: torch.Tensor,
        *,
        position_embeddings,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        if hidden_states.shape[1] != 1:
            raise ValueError("packed decode supports one token at a time")
        query, key, value = _project_qkv(
            module,
            hidden_states,
            position_embeddings,
            position_ids=kwargs.get("position_ids"),
        )
        pair = self.quantizers[layer]
        decision_layer = layer == 0
        packed = packed_decode_attention(
            self.cache,
            query[:, :, 0],
            layer=layer,
            key_quantizer=pair.key,
            value_quantizer=pair.value,
            policy=self.policy if decision_layer else None,
            previous_attention_mass=self._aggregate_attention_mass() if decision_layer else None,
            current_key=key,
            current_value=value,
        )
        append_kwargs = {}
        if isinstance(self.cache, PagedPrecisionCache):
            append_kwargs = {
                "retention_mode": self.policy.retention_mode if self.policy is not None else "recoverable",
                "admit_hot": self.policy.admit_new_blocks if self.policy is not None else False,
            }
        self.cache.append_layer_token(
            layer,
            self.token_position,
            pair.key.encode(key),
            pair.value.encode(value),
            **append_kwargs,
        )
        self.last_attention_mass[layer] = packed.attention_mass
        output = packed.values.reshape(hidden_states.shape[0], 1, -1).contiguous()
        return module.o_proj(output), None

    def _aggregate_attention_mass(self) -> torch.Tensor | None:
        masses = [mass for mass in self.last_attention_mass if mass is not None]
        if not masses:
            return None
        visible = self.cache.available_blocks()
        aligned = []
        for mass in masses:
            values = mass.detach().float().cpu().flatten()
            if values.numel() < visible:
                values = torch.nn.functional.pad(values, (0, visible - values.numel()))
            aligned.append(values[:visible])
        aggregate = torch.stack(aligned).mean(dim=0).clamp_min(0)
        return aggregate / aggregate.sum().clamp_min(1e-30)

    def decode(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.shape[1] != 1:
            raise ValueError("decode expects one token per batch")
        position_ids = torch.full_like(input_ids, self.token_position)
        cache_position = torch.tensor([self.token_position], dtype=torch.long, device=input_ids.device)
        with self.patched(), torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                use_cache=False,
                position_ids=position_ids,
                cache_position=cache_position,
                return_dict=True,
            )
        self.token_position += 1
        return outputs.logits[:, -1]

    def quantizer_hbm_bytes(self) -> int:
        seen: set[int] = set()
        total = 0
        for pair in self.quantizers:
            for quantizer in (pair.key, pair.value):
                for tensor in (
                    quantizer.rotation,
                    quantizer.base_codebook,
                    quantizer.residual_codebook,
                    quantizer.direct_codebook,
                ):
                    pointer = tensor.untyped_storage().data_ptr()
                    if pointer not in seen:
                        seen.add(pointer)
                        total += tensor.numel() * tensor.element_size()
        return total


class DirectKVModelAdapter(PagedPrecisionModelAdapter):
    """Run decode from a packed direct TQ2, TQ3, or TQ4 cache."""

    @classmethod
    def prefill(
        cls,
        model,
        input_ids: torch.Tensor,
        *,
        max_cache_tokens: int,
        quantizers: list[DirectKVQuantizerPair],
        block_size: int = 32,
    ) -> PrefillBuildResult:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, tokens]")
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=True, return_dict=True)
        legacy = extract_legacy_cache(outputs.past_key_values)
        layers = len(legacy)
        batch, kv_heads, token_count, head_dim = legacy[0][0].shape
        device = legacy[0][0].device
        dtype = legacy[0][0].dtype
        resident_quantizers = [pair.to(device, dtype) for pair in quantizers]
        bits = resident_quantizers[0].key.bits
        if any(pair.key.bits != bits or pair.value.bits != bits for pair in resident_quantizers):
            raise ValueError("all direct quantizers must use the same precision")
        cache = PackedDirectKVCache(
            DirectPackedCacheLayout(
                layers=layers,
                batch_size=batch,
                kv_heads=kv_heads,
                head_dim=head_dim,
                max_tokens=max_cache_tokens,
                bits=bits,
                block_size=block_size,
                norm_dtype=dtype,
            ),
            device=device,
        )
        cache.prepare_prefill(token_count)
        for layer, ((keys, values), pair) in enumerate(zip(legacy, resident_quantizers)):
            cache.store_prefill_layer(layer, pair.key.encode(keys), pair.value.encode(values))
        cache.synchronize_transfers()
        last_logits = outputs.logits[:, -1].detach().clone()
        del outputs, legacy
        gc.collect()
        if input_ids.device.type == "cuda":
            torch.cuda.empty_cache()
        return PrefillBuildResult(
            cls(model, cache, resident_quantizers, token_position=token_count, policy=None),
            last_logits,
        )

    def quantizer_hbm_bytes(self) -> int:
        seen: set[int] = set()
        total = 0
        for pair in self.quantizers:
            for tensor in (pair.key.rotation, pair.key.codebook, pair.value.rotation, pair.value.codebook):
                pointer = tensor.untyped_storage().data_ptr()
                if pointer not in seen:
                    seen.add(pointer)
                    total += tensor.numel() * tensor.element_size()
        return total


def _project_qkv(
    module,
    hidden_states: torch.Tensor,
    position_embeddings,
    *,
    position_ids: torch.Tensor | None,
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    query = module.q_proj(hidden_states)
    key = module.k_proj(hidden_states)
    query = query.view(hidden_shape).transpose(1, 2)
    key = key.view(hidden_shape).transpose(1, 2)
    value = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    if position_embeddings is None:
        rotary = getattr(module, "rotary_emb", None)
        if rotary is None or position_ids is None:
            raise ValueError("model attention did not provide rotary position embeddings")
        cos, sin = rotary(value, position_ids)
    else:
        cos, sin = position_embeddings
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return query * cos + _rotate_half(query) * sin, key * cos + _rotate_half(key) * sin, value


def _rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    half = tensor.shape[-1] // 2
    return torch.cat((-tensor[..., half:], tensor[..., :half]), dim=-1)


def _model_layers(model) -> list[Any]:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise ValueError("expected a Hugging Face causal LM with model.layers")
    return list(layers)
