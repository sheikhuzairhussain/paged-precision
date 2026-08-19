from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on CUDA CI/Modal
    triton = None
    tl = None


def packed_nbytes(element_count: int, bits: int) -> int:
    _validate_bits(bits)
    if element_count < 0:
        raise ValueError("element_count must be non-negative")
    return (element_count * bits + 7) // 8


@dataclass(frozen=True)
class PackedIndices:
    data: torch.Tensor
    element_count: int
    bits: int
    shape: tuple[int, ...]
    padding_bytes: int = 0

    @property
    def logical_bytes(self) -> int:
        return packed_nbytes(self.element_count, self.bits)

    @property
    def physical_bytes(self) -> int:
        return self.data.numel() * self.data.element_size()

    def unpack(self, *, use_triton: bool = True) -> torch.Tensor:
        values = unpack_indices(self.data[: self.logical_bytes], self.bits, self.element_count, use_triton=use_triton)
        return values.reshape(self.shape)


def pack_indices(
    indices: torch.Tensor,
    bits: int,
    *,
    alignment_bytes: int = 1,
    use_triton: bool = True,
) -> PackedIndices:
    _validate_bits(bits)
    if alignment_bytes <= 0:
        raise ValueError("alignment_bytes must be positive")
    values = indices.detach().contiguous().flatten()
    if values.dtype == torch.bool:
        values = values.to(torch.uint8)
    if values.is_floating_point() or values.is_complex():
        raise TypeError("indices must use an integer dtype")
    if values.numel():
        minimum = int(values.min().item())
        maximum = int(values.max().item())
        if minimum < 0 or maximum >= 2**bits:
            raise ValueError(f"indices for {bits} bits must be in [0, {2**bits - 1}]")
    logical_bytes = packed_nbytes(values.numel(), bits)
    physical_bytes = ((logical_bytes + alignment_bytes - 1) // alignment_bytes) * alignment_bytes
    if logical_bytes == 0:
        packed = torch.zeros(physical_bytes, dtype=torch.uint8, device=values.device)
    elif use_triton and _can_use_triton(values):
        packed = _triton_pack(values.to(torch.uint8), bits, logical_bytes)
        if physical_bytes > logical_bytes:
            packed = torch.cat(
                [packed, torch.zeros(physical_bytes - logical_bytes, dtype=torch.uint8, device=packed.device)]
            )
    else:
        packed = _torch_pack(values.to(torch.int64), bits, physical_bytes)
    return PackedIndices(
        data=packed,
        element_count=values.numel(),
        bits=bits,
        shape=tuple(indices.shape),
        padding_bytes=physical_bytes - logical_bytes,
    )


def unpack_indices(
    packed: torch.Tensor,
    bits: int,
    element_count: int,
    *,
    use_triton: bool = True,
) -> torch.Tensor:
    _validate_bits(bits)
    if packed.dtype != torch.uint8:
        raise TypeError("packed storage must have dtype torch.uint8")
    if element_count < 0:
        raise ValueError("element_count must be non-negative")
    required = packed_nbytes(element_count, bits)
    if packed.numel() < required:
        raise ValueError(f"packed storage has {packed.numel()} bytes; {required} required")
    source = packed.contiguous().flatten()
    if element_count == 0:
        return torch.empty(0, dtype=torch.uint8, device=source.device)
    if use_triton and _can_use_triton(source):
        return _triton_unpack(source, bits, element_count)
    return _torch_unpack(source, bits, element_count)


def _validate_bits(bits: int) -> None:
    if not 1 <= bits <= 5:
        raise ValueError("packed index width must be in [1, 5]")


def _torch_pack(values: torch.Tensor, bits: int, physical_bytes: int) -> torch.Tensor:
    if values.numel() == 0:
        return torch.zeros(physical_bytes, dtype=torch.uint8, device=values.device)
    element_offsets = torch.arange(values.numel(), device=values.device, dtype=torch.int64) * bits
    output = torch.zeros(physical_bytes, dtype=torch.int64, device=values.device)
    for bit in range(bits):
        positions = element_offsets + bit
        byte_indices = torch.div(positions, 8, rounding_mode="floor")
        bit_offsets = positions.remainder(8)
        contributions = ((values >> bit) & 1) << bit_offsets
        output.scatter_add_(0, byte_indices, contributions)
    return output.to(torch.uint8)


def _torch_unpack(packed: torch.Tensor, bits: int, element_count: int) -> torch.Tensor:
    element_offsets = torch.arange(element_count, device=packed.device, dtype=torch.int64) * bits
    result = torch.zeros(element_count, dtype=torch.int64, device=packed.device)
    source = packed.to(torch.int64)
    for bit in range(bits):
        positions = element_offsets + bit
        byte_indices = torch.div(positions, 8, rounding_mode="floor")
        bit_offsets = positions.remainder(8)
        result |= ((source[byte_indices] >> bit_offsets) & 1) << bit
    return result.to(torch.uint8)


def _can_use_triton(tensor: torch.Tensor) -> bool:
    return triton is not None and tensor.is_cuda


if triton is not None:  # pragma: no branch

    @triton.jit
    def _pack_kernel(source, output, element_count: tl.constexpr, bits: tl.constexpr, byte_count, BLOCK: tl.constexpr):
        byte_offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        byte_start_bits = byte_offsets * 8
        first_elements = byte_start_bits // bits
        accumulator = tl.zeros((BLOCK,), dtype=tl.int32)
        for relative in tl.static_range(0, 8):
            element_offsets = first_elements + relative
            element_start_bits = element_offsets * bits
            intersects = (
                (element_offsets < element_count)
                & (element_start_bits < byte_start_bits + 8)
                & (element_start_bits + bits > byte_start_bits)
            )
            values = tl.load(source + element_offsets, mask=intersects, other=0).to(tl.int32)
            shifts = element_start_bits - byte_start_bits
            left = values << tl.maximum(shifts, 0)
            right = values >> tl.maximum(-shifts, 0)
            accumulator |= tl.where(shifts >= 0, left, right)
        tl.store(output + byte_offsets, accumulator & 255, mask=byte_offsets < byte_count)

    @triton.jit
    def _unpack_kernel(source, output, element_count, bits: tl.constexpr, byte_count, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        bit_offsets = offsets * bits
        byte_offsets = bit_offsets // 8
        shifts = bit_offsets % 8
        low = tl.load(source + byte_offsets, mask=(offsets < element_count) & (byte_offsets < byte_count), other=0)
        high = tl.load(
            source + byte_offsets + 1,
            mask=(offsets < element_count) & (byte_offsets + 1 < byte_count),
            other=0,
        )
        combined = low.to(tl.int32) | (high.to(tl.int32) << 8)
        values = (combined >> shifts) & ((1 << bits) - 1)
        tl.store(output + offsets, values, mask=offsets < element_count)


def _triton_pack(values: torch.Tensor, bits: int, byte_count: int) -> torch.Tensor:
    output = torch.empty(byte_count, dtype=torch.uint8, device=values.device)
    block = 256
    _pack_kernel[(triton.cdiv(byte_count, block),)](
        values,
        output,
        element_count=values.numel(),
        bits=bits,
        byte_count=byte_count,
        BLOCK=block,
    )
    return output


def _triton_unpack(packed: torch.Tensor, bits: int, element_count: int) -> torch.Tensor:
    output = torch.empty(element_count, dtype=torch.uint8, device=packed.device)
    block = 256
    _unpack_kernel[(triton.cdiv(element_count, block),)](
        packed,
        output,
        element_count=element_count,
        bits=bits,
        byte_count=packed.numel(),
        BLOCK=block,
    )
    return output
