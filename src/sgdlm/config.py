"""Configuration objects for SGDLM models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SGDLMConfig:
    """Validated hyperparameters for sequential SGDLM inference."""

    lags: int = 1
    draws: int = 500
    beta: float = 0.95
    delta_state: float = 0.98
    delta_parent: float = 0.98
    prior_df: float = 10.0
    prior_overall: float = 0.2
    prior_cross: float = 0.5
    prior_lag_decay: float = 1.0
    prior_intercept: float = 10.0
    prior_exogenous: float = 1.0
    seed: int | None = None
    store_history: bool = False

    def __post_init__(self) -> None:
        if self.lags < 1:
            raise ValueError("lags must be at least 1")
        if self.draws < 10:
            raise ValueError("draws must be at least 10")
        for name in ("beta", "delta_state", "delta_parent"):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.prior_df <= 2.0:
            raise ValueError("prior_df must be greater than 2")
        for name in (
            "prior_overall",
            "prior_cross",
            "prior_lag_decay",
            "prior_intercept",
            "prior_exogenous",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> SGDLMConfig:
        return cls(**values)
