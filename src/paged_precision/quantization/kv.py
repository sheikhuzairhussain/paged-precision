from __future__ import annotations

from dataclasses import dataclass

import torch

from paged_precision.quantization.additive import AdditiveEncoding, AdditivePrecisionQuantizer


@dataclass(frozen=True)
class KVEncoding:
    key: AdditiveEncoding
    value: AdditiveEncoding


@dataclass(frozen=True)
class KVQuantizerPair:
    """Deterministic 2-bit base plus 2-bit residual quantizers."""

    key: AdditivePrecisionQuantizer
    value: AdditivePrecisionQuantizer

    @classmethod
    def for_head_dim(cls, head_dim: int, *, seed: int) -> "KVQuantizerPair":
        return cls(
            key=AdditivePrecisionQuantizer.for_dimension(head_dim, seed=seed),
            value=AdditivePrecisionQuantizer.for_dimension(head_dim, seed=seed + 100_000),
        )

    def encode(self, keys: torch.Tensor, values: torch.Tensor) -> KVEncoding:
        return KVEncoding(self.key.encode(keys), self.value.encode(values))

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> "KVQuantizerPair":
        return KVQuantizerPair(self.key.to(device, dtype), self.value.to(device, dtype))

    def codebook_identity_max_abs_error(self) -> tuple[float, float]:
        return self.key.codebook_identity_max_abs_error(), self.value.codebook_identity_max_abs_error()
