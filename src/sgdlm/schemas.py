"""Pydantic contracts for the HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .config import SGDLMConfig


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfigSchema(StrictSchema):
    lags: int = Field(default=1, ge=1)
    draws: int = Field(default=500, ge=10)
    beta: float = Field(default=0.95, gt=0, le=1)
    delta_state: float = Field(default=0.98, gt=0, le=1)
    delta_parent: float = Field(default=0.98, gt=0, le=1)
    prior_df: float = Field(default=10.0, gt=2)
    prior_overall: float = Field(default=0.2, gt=0)
    prior_cross: float = Field(default=0.5, gt=0)
    prior_lag_decay: float = Field(default=1.0, gt=0)
    prior_intercept: float = Field(default=10.0, gt=0)
    prior_exogenous: float = Field(default=1.0, gt=0)
    seed: int | None = None
    store_history: bool = False

    def to_domain(self) -> SGDLMConfig:
        return SGDLMConfig(**self.model_dump())


class FitRequest(StrictSchema):
    data: list[list[float]] = Field(min_length=4)
    parents: list[list[bool]] | None = None
    exog: list[list[float]] | None = None
    series_names: list[str] | None = None
    exog_names: list[str] | None = None
    config: ModelConfigSchema = Field(default_factory=ModelConfigSchema)


class ModelSummary(StrictSchema):
    model_id: str
    created_at: datetime
    observations: int
    series: int
    parameters: int
    series_names: list[str]
    exog_names: list[str]
    terminal_ess: float
    minimum_ess: float


class ModelListResponse(StrictSchema):
    models: list[ModelSummary]


class ForecastRequest(StrictSchema):
    horizon: int = Field(ge=1, le=500)
    future_exog: list[list[float]] | None = None
    simulations: int | None = Field(default=None, ge=1, le=100_000)
    credible_level: float = Field(default=0.9, gt=0, lt=1)
    seed: int | None = None
    include_simulations: bool = False


class ForecastResponse(StrictSchema):
    model_id: str
    series_names: list[str]
    mean: list[list[float]]
    lower: list[list[float]]
    upper: list[list[float]]
    simulations: list[list[list[float]]] | None = None


ShockScale = Literal["innovation_sd", "unit", "unit_effect"]
Smoothing = Literal["moving_average", "gaussian", "savgol"]


class IRFRequest(StrictSchema):
    horizon: int = Field(ge=0, le=500)
    impulse: int | str
    mode: Literal["terminal", "dynamic"] = "terminal"
    draws: int | None = Field(default=None, ge=1, le=100_000)
    credible_level: float = Field(default=0.9, gt=0, lt=1)
    seed: int | None = None
    shock_scale: ShockScale = "innovation_sd"
    smoothing: Smoothing | None = None
    smooth_window: int = Field(default=5, ge=3)


class TerminalIRFResponse(StrictSchema):
    model_id: str
    mode: Literal["terminal"] = "terminal"
    series_names: list[str]
    impulse: int
    mean: list[list[float]]
    lower: list[list[float]]
    upper: list[list[float]]


class DynamicIRFResponse(StrictSchema):
    model_id: str
    mode: Literal["dynamic"] = "dynamic"
    series_names: list[str]
    impulse: int
    origins: list[int]
    raw: list[list[list[float]]]
    smoothed: list[list[list[float]]] | None
    smoothing: Smoothing | None


class HealthResponse(StrictSchema):
    status: Literal["ok"]
    version: str


class ReadinessResponse(HealthResponse):
    storage: str


class ErrorResponse(StrictSchema):
    detail: str
