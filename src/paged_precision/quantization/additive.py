from __future__ import annotations

from dataclasses import dataclass

import torch

from paged_precision.quantization.codebooks import quantize_nearest, turboquant_coordinate_grid, weighted_lloyd
from paged_precision.quantization.rotations import make_random_rotation, split_norm_and_direction


@dataclass(frozen=True)
class AdditiveEncoding:
    base_indices: torch.Tensor
    residual_indices: torch.Tensor
    norms: torch.Tensor


@dataclass(frozen=True)
class AdditivePrecisionQuantizer:
    """A deterministic scalar quantizer with a 2-bit base and residual."""

    rotation: torch.Tensor
    base_codebook: torch.Tensor
    residual_codebook: torch.Tensor
    direct_codebook: torch.Tensor

    @classmethod
    def for_dimension(cls, dimension: int, *, seed: int, grid_size: int = 4097) -> "AdditivePrecisionQuantizer":
        if dimension < 2:
            raise ValueError("dimension must be at least two")
        rotation = make_random_rotation(dimension, seed)
        grid, weights = turboquant_coordinate_grid(dimension, grid_size)
        direct = weighted_lloyd(grid, weights, levels=16).sort().values
        direct_indices, _ = quantize_nearest(grid, direct)
        base, residual = _split_codebook(direct, grid, weights, direct_indices)
        return cls(rotation, base, residual, direct)

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> "AdditivePrecisionQuantizer":
        dtype = dtype or self.rotation.dtype
        return AdditivePrecisionQuantizer(
            rotation=self.rotation.to(device=device, dtype=dtype),
            base_codebook=self.base_codebook.to(device=device, dtype=dtype),
            residual_codebook=self.residual_codebook.to(device=device, dtype=dtype),
            direct_codebook=self.direct_codebook.to(device=device, dtype=dtype),
        )

    def encode(self, vectors: torch.Tensor) -> AdditiveEncoding:
        original_dtype = vectors.dtype
        directions, norms = split_norm_and_direction(vectors.to(self.rotation.dtype))
        indices, _ = quantize_nearest(directions @ self.rotation, self.direct_codebook)
        return AdditiveEncoding(
            base_indices=torch.div(indices.long(), 4, rounding_mode="floor").to(torch.uint8),
            residual_indices=(indices.long() % 4).to(torch.uint8),
            norms=norms.to(original_dtype),
        )

    def decode_base(self, encoding: AdditiveEncoding, dtype: torch.dtype | None = None) -> torch.Tensor:
        rotated = self.base_codebook[encoding.base_indices.long()]
        return self._reconstruct(rotated, encoding.norms, dtype)

    def decode_full(self, encoding: AdditiveEncoding, dtype: torch.dtype | None = None) -> torch.Tensor:
        return self._reconstruct(self._refined(encoding), encoding.norms, dtype)

    def decode_rotated(
        self,
        encoding: AdditiveEncoding,
        *,
        refined: bool,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        rotated = self._refined(encoding) if refined else self.base_codebook[encoding.base_indices.long()]
        scaled = rotated * encoding.norms.to(rotated.dtype)
        return scaled.to(dtype or encoding.norms.dtype)

    def decode_rotated_mixed(
        self,
        encoding: AdditiveEncoding,
        residual_mask: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        base = self.base_codebook[encoding.base_indices.long()]
        refined = self._refined(encoding)
        mask = residual_mask.to(device=base.device, dtype=torch.bool).view(1, 1, -1, 1)
        scaled = torch.where(mask, refined, base) * encoding.norms.to(base.dtype)
        return scaled.to(dtype)

    def decode_direct(self, vectors: torch.Tensor) -> torch.Tensor:
        directions, norms = split_norm_and_direction(vectors.to(self.rotation.dtype))
        _, rotated = quantize_nearest(directions @ self.rotation, self.direct_codebook)
        return self._reconstruct(rotated, norms, vectors.dtype)

    def codebook_identity_max_abs_error(self) -> float:
        nested = self.base_codebook[:, None] + self.residual_codebook
        return float((nested - self.direct_codebook.reshape_as(nested)).abs().max().item())

    def _refined(self, encoding: AdditiveEncoding) -> torch.Tensor:
        indices = encoding.base_indices.long() * 4 + encoding.residual_indices.long()
        return self.direct_codebook[indices]

    def _reconstruct(
        self,
        rotated: torch.Tensor,
        norms: torch.Tensor,
        dtype: torch.dtype | None,
    ) -> torch.Tensor:
        vectors = (rotated @ self.rotation.T) * norms.to(rotated.dtype)
        return vectors.to(dtype or norms.dtype)


def _split_codebook(
    direct: torch.Tensor,
    samples: torch.Tensor,
    weights: torch.Tensor,
    direct_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    base_values = []
    residual_rows = []
    for parent in range(4):
        start = parent * 4
        mask = (direct_indices >= start) & (direct_indices < start + 4)
        base = (samples[mask] * weights[mask]).sum() / weights[mask].sum().clamp_min(1e-30)
        children = direct[start : start + 4]
        base_values.append(base)
        residual_rows.append(children - base)
    return torch.stack(base_values), torch.stack(residual_rows)
