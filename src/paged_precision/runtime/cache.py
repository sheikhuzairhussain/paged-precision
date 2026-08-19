from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from paged_precision.policies import RefinementRetention
from paged_precision.quantization import AdditiveEncoding, AdditivePrecisionQuantizer
from paged_precision.runtime.packing import pack_indices, packed_nbytes, unpack_indices


@dataclass(frozen=True)
class PackedCacheLayout:
    layers: int
    batch_size: int
    kv_heads: int
    head_dim: int
    max_tokens: int
    hot_fraction: float
    block_size: int = 32
    norm_dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        dimensions = (self.layers, self.batch_size, self.kv_heads, self.head_dim, self.max_tokens, self.block_size)
        if any(value <= 0 for value in dimensions):
            raise ValueError("cache dimensions must be positive")
        if not 0 <= self.hot_fraction <= 1:
            raise ValueError("hot_fraction must be in [0, 1]")

    @property
    def max_blocks(self) -> int:
        return math.ceil(self.max_tokens / self.block_size)

    @property
    def hot_slots(self) -> int:
        return math.ceil(self.max_blocks * self.hot_fraction)

    @property
    def packed_block_bytes(self) -> int:
        return packed_nbytes(self.block_size * self.head_dim, 2)


@dataclass(frozen=True)
class RuntimeMemorySnapshot:
    cache_hbm_bytes: int
    pinned_dram_bytes: int


class PagedPrecisionCache:
    """A 2-bit base in HBM with fixed 2-bit residual slots."""

    def __init__(self, layout: PackedCacheLayout, *, device: torch.device | str | None = None) -> None:
        self.layout = layout
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        block_shape = (
            layout.max_blocks,
            layout.layers,
            2,
            layout.batch_size,
            layout.kv_heads,
            layout.packed_block_bytes,
        )
        slot_shape = (layout.hot_slots, *block_shape[1:])
        self.base = torch.zeros(block_shape, dtype=torch.uint8, device=self.device)
        self.hot_residual = torch.zeros(slot_shape, dtype=torch.uint8, device=self.device)
        self.norms = torch.zeros(
            (layout.layers, 2, layout.batch_size, layout.kv_heads, layout.max_tokens),
            dtype=layout.norm_dtype,
            device=self.device,
        )
        self.valid_tokens = torch.zeros(layout.max_blocks, dtype=torch.int32)
        self.block_to_slot = torch.full((layout.max_blocks,), -1, dtype=torch.int32)
        self.slot_to_block = torch.full((layout.hot_slots,), -1, dtype=torch.int32)
        self.cold: dict[int, torch.Tensor] = {}

    def prepare_prefill(
        self,
        token_count: int,
        *,
        hot_block_ids: list[int] | None = None,
        retention_mode: RefinementRetention = "recoverable",
    ) -> None:
        if self.available_blocks():
            raise RuntimeError("prepare_prefill requires an empty cache")
        if not 0 < token_count <= self.layout.max_tokens:
            raise ValueError("token_count exceeds cache capacity")
        blocks = math.ceil(token_count / self.layout.block_size)
        if hot_block_ids is None:
            hot_block_ids = list(range(max(0, blocks - self.layout.hot_slots), blocks))
        hot = list(dict.fromkeys(int(block) for block in hot_block_ids))
        if len(hot) > self.layout.hot_slots or any(block < 0 or block >= blocks for block in hot):
            raise ValueError("invalid prefill hot block set")

        for block in range(blocks):
            start = block * self.layout.block_size
            self.valid_tokens[block] = min(self.layout.block_size, token_count - start)
        for slot, block in enumerate(hot):
            self.block_to_slot[block] = slot
            self.slot_to_block[slot] = block
        for block in range(blocks):
            if self.layout.hot_slots and block not in hot and retention_mode == "recoverable":
                self.cold[block] = self._empty_cold_block()

    def store_prefill_layer(
        self,
        layer: int,
        keys: AdditiveEncoding,
        values: AdditiveEncoding,
    ) -> None:
        if not 0 <= layer < self.layout.layers:
            raise IndexError("layer index is out of range")
        tokens = self.available_tokens()
        blocks = self.available_blocks()
        for side, encoding in enumerate((keys, values)):
            base = self._pack_sequence(encoding.base_indices[..., :tokens, :])
            residual = self._pack_sequence(encoding.residual_indices[..., :tokens, :])
            self.base[:blocks, layer, side].copy_(base)
            for block in range(blocks):
                slot = int(self.block_to_slot[block].item())
                if slot >= 0:
                    self.hot_residual[slot, layer, side].copy_(residual[block])
                elif block in self.cold:
                    self.cold[block][layer, side].copy_(residual[block], non_blocking=False)
            self.norms[layer, side, ..., :tokens].copy_(
                encoding.norms[..., :tokens, :].squeeze(-1).to(self.device, self.layout.norm_dtype)
            )

    def append_layer_token(
        self,
        layer: int,
        token_index: int,
        keys: AdditiveEncoding,
        values: AdditiveEncoding,
        *,
        retention_mode: RefinementRetention = "recoverable",
        admit_hot: bool = False,
    ) -> None:
        if not 0 <= token_index < self.layout.max_tokens:
            raise IndexError("token index exceeds cache capacity")
        block = token_index // self.layout.block_size
        offset = token_index % self.layout.block_size
        if layer == 0 and int(self.valid_tokens[block].item()) != offset:
            raise RuntimeError("decode tokens must be appended in order")

        if layer == 0 and offset == 0:
            if self.layout.hot_slots == self.layout.max_blocks:
                slot = self._allocate_slot(exclude={block})
                self.block_to_slot[block] = slot
                self.slot_to_block[slot] = block
            elif self.layout.hot_slots and admit_hot:
                slot = self._allocate_slot(
                    exclude={block},
                    retain_victim=retention_mode == "recoverable",
                )
                self.block_to_slot[block] = slot
                self.slot_to_block[slot] = block
            elif self.layout.hot_slots and retention_mode == "recoverable":
                self.cold[block] = self._empty_cold_block()

        for side, encoding in enumerate((keys, values)):
            self._update_token(self.base[block, layer, side], encoding.base_indices, offset)
            if self.is_hot(block):
                slot = int(self.block_to_slot[block].item())
                self._update_token(self.hot_residual[slot, layer, side], encoding.residual_indices, offset)
            elif block in self.cold:
                self._update_token(self.cold[block][layer, side], encoding.residual_indices, offset)
            self.norms[layer, side, ..., token_index].copy_(
                encoding.norms[..., 0, :].squeeze(-1).to(self.device, self.layout.norm_dtype)
            )
        if layer == self.layout.layers - 1:
            self.valid_tokens[block] = offset + 1

    def demote(self, block_ids: list[int], *, retain_cold: bool = True) -> None:
        for block in _unique(block_ids):
            slot = int(self.block_to_slot[block].item())
            if slot < 0:
                continue
            if retain_cold:
                cold = self._empty_cold_block()
                cold.copy_(self.hot_residual[slot], non_blocking=False)
                self.cold[block] = cold
            else:
                self.cold.pop(block, None)
            self.block_to_slot[block] = -1
            self.slot_to_block[slot] = -1

    def promote(self, block_ids: list[int]) -> None:
        for block in _unique(block_ids):
            if self.is_hot(block) or block not in self.cold:
                continue
            slot = self._allocate_slot(exclude={block})
            self.hot_residual[slot].copy_(self.cold.pop(block), non_blocking=False)
            self.block_to_slot[block] = slot
            self.slot_to_block[slot] = block

    def set_hot_blocks(
        self,
        block_ids: list[int],
        *,
        retention_mode: RefinementRetention = "recoverable",
    ) -> None:
        visible = self.available_blocks()
        target = set(_unique(block_ids))
        if len(target) > self.layout.hot_slots or any(block < 0 or block >= visible for block in target):
            raise ValueError("invalid hot block set")
        current = {block for block in range(visible) if self.is_hot(block)}
        missing = sorted(block for block in target - current if block not in self.cold)
        if missing:
            raise RuntimeError(f"discarded refinements cannot be promoted: {missing}")
        self.demote(sorted(current - target), retain_cold=retention_mode == "recoverable")
        self.promote(sorted(target - current))
        actual = {block for block in range(visible) if self.is_hot(block)}
        if actual != target:
            raise RuntimeError(f"hot block transition produced {sorted(actual)}, expected {sorted(target)}")

    def decode_rotated_blocks(
        self,
        start_block: int,
        end_block: int,
        layer: int,
        side: Literal["key", "value"],
        quantizer: AdditivePrecisionQuantizer,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not 0 <= start_block < end_block <= self.available_blocks():
            raise ValueError("invalid block range")
        side_index = 0 if side == "key" else 1
        blocks = end_block - start_block
        elements = self.layout.block_size * self.layout.head_dim
        stored_elements = blocks * self.layout.batch_size * self.layout.kv_heads * elements
        base_source = self.base[start_block:end_block, layer, side_index]
        residual_source = torch.zeros_like(base_source)
        slots = self.block_to_slot[start_block:end_block]
        hot_blocks = (slots >= 0).nonzero().flatten()
        if hot_blocks.numel():
            resident_slots = slots[hot_blocks].long().to(self.hot_residual.device)
            residual_source[hot_blocks.to(residual_source.device)] = self.hot_residual[
                resident_slots,
                layer,
                side_index,
            ]
        packed_shape = (
            blocks,
            self.layout.batch_size,
            self.layout.kv_heads,
            self.layout.block_size,
            self.layout.head_dim,
        )
        base = unpack_indices(base_source, 2, stored_elements).reshape(packed_shape)
        residual = unpack_indices(residual_source, 2, stored_elements).reshape(packed_shape)
        base = base.permute(1, 2, 0, 3, 4).reshape(
            self.layout.batch_size,
            self.layout.kv_heads,
            blocks * self.layout.block_size,
            self.layout.head_dim,
        )
        residual = residual.permute(1, 2, 0, 3, 4).reshape_as(base)
        token_count = int(self.valid_tokens[start_block:end_block].sum().item())
        token_start = start_block * self.layout.block_size
        norms = self.norms[layer, side_index, ..., token_start : token_start + token_count].unsqueeze(-1)
        hot_tokens = (slots >= 0).repeat_interleave(self.layout.block_size)[:token_count]
        encoding = AdditiveEncoding(base[..., :token_count, :], residual[..., :token_count, :], norms)
        return quantizer.decode_rotated_mixed(encoding, hot_tokens, dtype=dtype)

    def memory_snapshot(self) -> RuntimeMemorySnapshot:
        hbm = sum(_nbytes(tensor) for tensor in (self.base, self.hot_residual, self.norms))
        dram = sum(_nbytes(tensor) for tensor in self.cold.values())
        return RuntimeMemorySnapshot(hbm, dram)

    def synchronize_transfers(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def is_hot(self, block: int) -> bool:
        return int(self.block_to_slot[block].item()) >= 0

    def available_blocks(self) -> int:
        valid = (self.valid_tokens > 0).nonzero().flatten()
        return int(valid[-1].item()) + 1 if valid.numel() else 0

    def available_tokens(self) -> int:
        return int(self.valid_tokens.sum().item())

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
        return pack_indices(rows.contiguous().reshape(-1), 2).data.reshape(
            blocks,
            batch,
            heads,
            self.layout.packed_block_bytes,
        )

    def _update_token(self, destination: torch.Tensor, indices: torch.Tensor, offset: int) -> None:
        elements = self.layout.block_size * self.layout.head_dim
        for batch in range(self.layout.batch_size):
            for head in range(self.layout.kv_heads):
                unpacked = unpack_indices(destination[batch, head], 2, elements).reshape(
                    self.layout.block_size,
                    self.layout.head_dim,
                )
                unpacked[offset].copy_(indices[batch, head, 0].to(unpacked.device))
                destination[batch, head].copy_(pack_indices(unpacked, 2).data)

    def _allocate_slot(self, *, exclude: set[int], retain_victim: bool = True) -> int:
        free = (self.slot_to_block < 0).nonzero().flatten()
        if free.numel():
            return int(free[0].item())
        candidates = [int(block) for block in self.slot_to_block.tolist() if int(block) not in exclude]
        if not candidates:
            raise RuntimeError("no residual slot can be allocated")
        victim = min(candidates)
        slot = int(self.block_to_slot[victim].item())
        self.demote([victim], retain_cold=retain_victim)
        return slot

    def _empty_cold_block(self) -> torch.Tensor:
        return torch.zeros(
            (
                self.layout.layers,
                2,
                self.layout.batch_size,
                self.layout.kv_heads,
                self.layout.packed_block_bytes,
            ),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=self.device.type == "cuda",
        )


def _unique(values: list[int]) -> list[int]:
    return list(dict.fromkeys(int(value) for value in values))


def _nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()
