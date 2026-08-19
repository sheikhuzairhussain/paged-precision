from __future__ import annotations

from dataclasses import dataclass

import torch

from paged_precision.quantization.codebooks import quantize_nearest, turboquant_coordinate_grid, weighted_lloyd
from paged_precision.quantization.rotations import make_random_rotation, split_norm_and_direction


@dataclass(frozen=True)
class DirectEncoding:
    indices: torch.Tensor
    norms: torch.Tensor


@dataclass(frozen=True)
class DirectPrecisionQuantizer:
    """A calibration-free direct TurboQuant-MSE scalar quantizer."""

    bits: int
    rotation: torch.Tensor
    codebook: torch.Tensor

    @classmethod
    def for_dimension(cls, dimension: int, *, bits: int, seed: int) -> "DirectPrecisionQuantizer":
        if bits not in (2, 3, 4):
            raise ValueError("direct precision must be 2, 3, or 4 bits")
        grid, weights = turboquant_coordinate_grid(dimension)
        codebook = weighted_lloyd(grid, weights, levels=2**bits).sort().values
        return cls(bits, make_random_rotation(dimension, seed), codebook)

    def to(self, device: torch.device | str, dtype: torch.dtype) -> "DirectPrecisionQuantizer":
        return DirectPrecisionQuantizer(
            self.bits,
            self.rotation.to(device=device, dtype=dtype),
            self.codebook.to(device=device, dtype=dtype),
        )

    def encode(self, vectors: torch.Tensor) -> DirectEncoding:
        directions, norms = split_norm_and_direction(vectors.to(self.rotation.dtype))
        indices, _ = quantize_nearest(directions @ self.rotation, self.codebook)
        return DirectEncoding(indices, norms.to(vectors.dtype))

    def decode(self, encoding: DirectEncoding, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        rotated = self.decode_rotated(encoding, dtype=self.rotation.dtype)
        return (rotated @ self.rotation.T).to(dtype or encoding.norms.dtype)

    def decode_rotated(self, encoding: DirectEncoding, *, dtype: torch.dtype) -> torch.Tensor:
        values = self.codebook[encoding.indices.long()] * encoding.norms.to(self.codebook.dtype)
        return values.to(dtype)


@dataclass(frozen=True)
class DirectKVQuantizerPair:
    key: DirectPrecisionQuantizer
    value: DirectPrecisionQuantizer

    @classmethod
    def for_head_dim(cls, head_dim: int, *, bits: int, seed: int) -> "DirectKVQuantizerPair":
        return cls(
            DirectPrecisionQuantizer.for_dimension(head_dim, bits=bits, seed=seed),
            DirectPrecisionQuantizer.for_dimension(head_dim, bits=bits, seed=seed + 100_000),
        )

    def to(self, device: torch.device | str, dtype: torch.dtype) -> "DirectKVQuantizerPair":
        return DirectKVQuantizerPair(self.key.to(device, dtype), self.value.to(device, dtype))
