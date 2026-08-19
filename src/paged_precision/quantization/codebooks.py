from __future__ import annotations

import torch


def turboquant_coordinate_grid(dim: int, grid_size: int = 4097) -> tuple[torch.Tensor, torch.Tensor]:
    grid = torch.linspace(-1.0, 1.0, grid_size, dtype=torch.float64)
    if dim <= 2:
        weights = torch.ones_like(grid)
    else:
        weights = (1.0 - grid.square()).clamp_min(0.0).pow((dim - 3) / 2)
    weights = weights / weights.sum().clamp_min(1e-30)
    return grid, weights


def weighted_lloyd(
    grid: torch.Tensor,
    weights: torch.Tensor,
    levels: int,
    iterations: int = 60,
) -> torch.Tensor:
    cdf = weights.cumsum(dim=0)
    quantiles = torch.linspace(0.0, 1.0, levels + 2, dtype=grid.dtype)[1:-1]
    indices = torch.searchsorted(cdf, quantiles).clamp(max=grid.numel() - 1)
    centroids = grid[indices].clone()
    for _ in range(iterations):
        labels = torch.argmin((grid[:, None] - centroids[None, :]).square(), dim=1)
        updated = centroids.clone()
        for level in range(levels):
            mask = labels == level
            if mask.any():
                level_weights = weights[mask]
                updated[level] = (grid[mask] * level_weights).sum() / level_weights.sum().clamp_min(1e-30)
        if torch.allclose(updated, centroids, rtol=0, atol=1e-12):
            break
        centroids = updated
    return centroids


def quantize_nearest(values: torch.Tensor, codebook: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.argmin((values[..., None] - codebook).square(), dim=-1)
    reconstruction = codebook[indices]
    return indices.to(torch.uint8), reconstruction
