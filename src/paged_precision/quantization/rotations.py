from __future__ import annotations

import torch


def make_random_rotation(dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn(dim, dim, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(matrix)
    signs = torch.sign(torch.diag(r))
    signs[signs == 0] = 1
    return q * signs


def split_norm_and_direction(vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    norms = vectors.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return vectors / norms, norms
