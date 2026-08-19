from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Literal, Protocol

import torch


RefinementRetention = Literal["recoverable", "discardable"]


@dataclass(frozen=True)
class ResidencyObservation:
    visible_blocks: int
    hot_blocks: torch.Tensor
    previous_attention_mass: torch.Tensor | None
    hot_capacity: int | None = None


@dataclass(frozen=True)
class TransitionPlan:
    promote: tuple[int, ...] = ()
    demote: tuple[int, ...] = ()
    selected: tuple[int, ...] = ()


class ResidencyPolicy(Protocol):
    retention_mode: RefinementRetention
    admit_new_blocks: bool

    def observe(self, observation: ResidencyObservation) -> TransitionPlan: ...


def _transition(observation: ResidencyObservation, target: set[int]) -> TransitionPlan:
    visible = observation.visible_blocks
    hot = {
        int(index)
        for index in observation.hot_blocks[:visible].nonzero().flatten().tolist()
    }
    return TransitionPlan(
        promote=tuple(sorted(target - hot)),
        demote=tuple(sorted(hot - target)),
        selected=tuple(sorted(target)),
    )


def _capacity(observation: ResidencyObservation, fraction: float) -> int:
    requested = observation.hot_capacity
    if requested is None:
        requested = math.ceil(observation.visible_blocks * fraction)
    return min(max(1, requested), observation.visible_blocks)


@dataclass(frozen=True)
class SinkPolicy:
    """Keep the earliest residual blocks in HBM."""

    retention_mode: ClassVar[RefinementRetention] = "discardable"
    admit_new_blocks: ClassVar[bool] = False
    hot_fraction: float

    def __post_init__(self) -> None:
        if not 0 < self.hot_fraction < 1:
            raise ValueError("sink residency requires a fraction between zero and one")

    def observe(self, observation: ResidencyObservation) -> TransitionPlan:
        capacity = _capacity(observation, self.hot_fraction)
        return _transition(observation, set(range(min(capacity, observation.visible_blocks))))


@dataclass(frozen=True)
class RecentPolicy:
    """Keep the newest residual blocks in HBM."""

    retention_mode: ClassVar[RefinementRetention] = "discardable"
    admit_new_blocks: ClassVar[bool] = True
    hot_fraction: float

    def __post_init__(self) -> None:
        if not 0 < self.hot_fraction < 1:
            raise ValueError("recent residency requires a fraction between zero and one")

    def observe(self, observation: ResidencyObservation) -> TransitionPlan:
        capacity = _capacity(observation, self.hot_fraction)
        start = max(0, observation.visible_blocks - capacity)
        return _transition(observation, set(range(start, observation.visible_blocks)))


@dataclass
class AttentionEMAPolicy:
    """Keep residuals for blocks with the highest recent attention mass."""

    retention_mode: ClassVar[RefinementRetention] = "recoverable"
    admit_new_blocks: ClassVar[bool] = False
    hot_fraction: float
    sink_blocks: int = 1
    recent_blocks: int = 4
    decay: float = 0.9
    _ema: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not 0 < self.hot_fraction < 1:
            raise ValueError("EMA residency requires a fraction between zero and one")

    def observe(self, observation: ResidencyObservation) -> TransitionPlan:
        visible = observation.visible_blocks
        hot = {
            int(index)
            for index in observation.hot_blocks[:visible].nonzero().flatten().tolist()
        }
        scores = observation.previous_attention_mass
        if scores is None:
            return TransitionPlan(selected=tuple(sorted(hot)))

        values = scores.detach().double().cpu().flatten()
        if values.numel() < visible:
            values = torch.nn.functional.pad(values, (0, visible - values.numel()))
        values = values[:visible]
        if self._ema is None or self._ema.numel() < visible:
            old_ema = self._ema
            self._ema = torch.zeros(visible, dtype=torch.float64)
            if old_ema is not None:
                self._ema[: old_ema.numel()] = old_ema
        self._ema[:visible] = self.decay * self._ema[:visible] + (1 - self.decay) * values

        capacity = _capacity(observation, self.hot_fraction)
        recent_start = max(visible - self.recent_blocks, 0)
        protected = list(dict.fromkeys((*range(min(self.sink_blocks, visible)), *range(recent_start, visible))))
        target = set(protected[:capacity])
        ranked = sorted(
            (block for block in range(visible) if block not in target),
            key=lambda block: (-float(self._ema[block]), block),
        )
        target.update(ranked[: max(0, capacity - len(target))])
        return TransitionPlan(
            promote=tuple(sorted(target - hot)),
            demote=tuple(sorted(hot - target)),
            selected=tuple(sorted(target)),
        )
