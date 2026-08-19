from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from paged_precision.quantization import DirectEncoding, DirectPrecisionQuantizer
from paged_precision.runtime.cache import RuntimeMemorySnapshot
from paged_precision.runtime.packing import pack_indices, packed_nbytes, unpack_indices


@dataclass(frozen=True)
class DirectPackedCacheLayout:
    layers: int
    batch_size: int
    kv_heads: int
    head_dim: int
    max_tokens: int
    bits: int
    block_size: int = 32
    norm_dtype: torch.dtype = torch.bfloat16

    @property
    def max_blocks(self) -> int:
        return math.ceil(self.max_tokens / self.block_size)

    @property
    def packed_block_bytes(self) -> int:
        return packed_nbytes(self.block_size * self.head_dim, self.bits)


class PackedDirectKVCache:
    """A packed fixed-precision cache for direct TQ2, TQ3, or TQ4."""

    def __init__(self, layout: DirectPackedCacheLayout, *, device: torch.device | str) -> None:
        self.layout = layout
        self.device = torch.device(device)
        self.indices = torch.zeros(
            (
                layout.max_blocks,
                layout.layers,
                2,
                layout.batch_size,
                layout.kv_heads,
                layout.packed_block_bytes,
            ),
            dtype=torch.uint8,
            device=self.device,
        )
        self.norms = torch.zeros(
            (layout.layers, 2, layout.batch_size, layout.kv_heads, layout.max_tokens),
            dtype=layout.norm_dtype,
            device=self.device,
        )
        self.valid_tokens = torch.zeros(layout.max_blocks, dtype=torch.int32)

    def prepare_prefill(self, token_count: int) -> None:
        if self.available_blocks() or not 0 < token_count <= self.layout.max_tokens:
            raise ValueError("prefill requires an empty cache and a valid token count")
        for block in range(math.ceil(token_count / self.layout.block_size)):
            start = block * self.layout.block_size
            self.valid_tokens[block] = min(self.layout.block_size, token_count - start)

    def store_prefill_layer(self, layer: int, keys: DirectEncoding, values: DirectEncoding) -> None:
        tokens = self.available_tokens()
        blocks = self.available_blocks()
        for side, encoding in enumerate((keys, values)):
            self.indices[:blocks, layer, side].copy_(self._pack_sequence(encoding.indices[..., :tokens, :]))
            self.norms[layer, side, ..., :tokens].copy_(
                encoding.norms[..., :tokens, :].squeeze(-1).to(self.device, self.layout.norm_dtype)
            )

    def append_layer_token(
        self,
        layer: int,
        token_index: int,
        keys: DirectEncoding,
        values: DirectEncoding,
    ) -> None:
        block = token_index // self.layout.block_size
        offset = token_index % self.layout.block_size
        if layer == 0 and int(self.valid_tokens[block].item()) != offset:
            raise RuntimeError("decode tokens must be appended in order")
        for side, encoding in enumerate((keys, values)):
            self._update_token(self.indices[block, layer, side], encoding.indices, offset)
            self.norms[layer, side, ..., token_index].copy_(
                encoding.norms[..., 0, :].squeeze(-1).to(self.device, self.layout.norm_dtype)
            )
        if layer == self.layout.layers - 1:
            self.valid_tokens[block] = offset + 1

    def decode_rotated_blocks(
        self,
        start_block: int,
        end_block: int,
        layer: int,
        side: Literal["key", "value"],
        quantizer: DirectPrecisionQuantizer,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        side_index = 0 if side == "key" else 1
        blocks = end_block - start_block
        elements = self.layout.block_size * self.layout.head_dim
        stored = blocks * self.layout.batch_size * self.layout.kv_heads * elements
        indices = unpack_indices(
            self.indices[start_block:end_block, layer, side_index],
            self.layout.bits,
            stored,
        ).reshape(
            blocks,
            self.layout.batch_size,
            self.layout.kv_heads,
            self.layout.block_size,
            self.layout.head_dim,
        )
        indices = indices.permute(1, 2, 0, 3, 4).reshape(
            self.layout.batch_size,
            self.layout.kv_heads,
            blocks * self.layout.block_size,
            self.layout.head_dim,
        )
        token_count = int(self.valid_tokens[start_block:end_block].sum().item())
        token_start = start_block * self.layout.block_size
        norms = self.norms[layer, side_index, ..., token_start : token_start + token_count].unsqueeze(-1)
        return (quantizer.codebook[indices[..., :token_count, :].long()] * norms).to(dtype)

    def available_blocks(self) -> int:
        valid = (self.valid_tokens > 0).nonzero().flatten()
        return int(valid[-1].item()) + 1 if valid.numel() else 0

    def available_tokens(self) -> int:
        return int(self.valid_tokens.sum().item())

    def memory_snapshot(self) -> RuntimeMemorySnapshot:
        hbm = sum(tensor.numel() * tensor.element_size() for tensor in (self.indices, self.norms))
        return RuntimeMemorySnapshot(hbm, 0)

    def synchronize_transfers(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _pack_sequence(self, values: torch.Tensor) -> torch.Tensor:
        batch, heads, tokens, channels = values.shape
        blocks = self.available_blocks()
        padded = torch.zeros(
            (batch, heads, blocks * self.layout.block_size, channels),
            dtype=values.dtype,
            device=values.device,
        )
        padded[..., :tokens, :] = values
        rows = padded.reshape(batch, heads, blocks, self.layout.block_size, channels).permute(2, 0, 1, 3, 4)
        return pack_indices(rows.contiguous().reshape(-1), self.layout.bits).data.reshape(
            blocks,
            batch,
            heads,
            self.layout.packed_block_bytes,
        )

    def _update_token(self, destination: torch.Tensor, indices: torch.Tensor, offset: int) -> None:
        elements = self.layout.block_size * self.layout.head_dim
        for batch in range(self.layout.batch_size):
            for head in range(self.layout.kv_heads):
                unpacked = unpack_indices(destination[batch, head], self.layout.bits, elements).reshape(
                    self.layout.block_size,
                    self.layout.head_dim,
                )
                unpacked[offset].copy_(indices[batch, head, 0].to(unpacked.device))
                destination[batch, head].copy_(pack_indices(unpacked, self.layout.bits).data)
