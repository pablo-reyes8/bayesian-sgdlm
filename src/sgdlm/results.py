"""Typed output containers and model persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import SGDLMConfig

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(slots=True)
class ForecastResult:
    mean: FloatArray
    lower: FloatArray
    upper: FloatArray
    simulations: FloatArray

    def to_dict(self, *, include_simulations: bool = False) -> dict[str, Any]:
        output: dict[str, Any] = {
            "mean": self.mean.tolist(),
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
        }
        if include_simulations:
            output["simulations"] = self.simulations.tolist()
        return output


@dataclass(slots=True)
class IRFResult:
    mean: FloatArray
    lower: FloatArray
    upper: FloatArray
    impulse: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
            "impulse": self.impulse,
        }


@dataclass(slots=True)
class DynamicIRFResult:
    raw: FloatArray
    smoothed: FloatArray | None
    origins: NDArray[np.int64]
    impulse: int
    smoothing: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw.tolist(),
            "smoothed": None if self.smoothed is None else self.smoothed.tolist(),
            "origins": self.origins.tolist(),
            "impulse": self.impulse,
            "smoothing": self.smoothing,
        }


@dataclass(slots=True)
class FitResult:
    config: SGDLMConfig
    data: FloatArray
    parents: BoolArray
    pdims: NDArray[np.int64]
    theta: FloatArray
    precision: FloatArray
    weights: FloatArray
    series_names: list[str]
    exog_names: list[str]
    effective_sample_size: FloatArray
    theta_mean_history: FloatArray
    precision_mean_history: FloatArray
    theta_history: FloatArray | None = None
    precision_history: FloatArray | None = None

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = json.dumps(
            {
                "format_version": 1,
                "config": self.config.to_dict(),
                "series_names": self.series_names,
                "exog_names": self.exog_names,
            }
        )
        np.savez_compressed(
            destination,
            metadata=np.asarray(metadata),
            data=self.data,
            parents=self.parents,
            pdims=self.pdims,
            theta=self.theta,
            precision=self.precision,
            weights=self.weights,
            effective_sample_size=self.effective_sample_size,
            theta_mean_history=self.theta_mean_history,
            precision_mean_history=self.precision_mean_history,
            theta_history=np.empty(0) if self.theta_history is None else self.theta_history,
            precision_history=(
                np.empty(0) if self.precision_history is None else self.precision_history
            ),
        )

    @classmethod
    def load(cls, path: str | Path) -> FitResult:
        with np.load(Path(path), allow_pickle=False) as values:
            metadata = json.loads(str(values["metadata"]))
            if metadata.get("format_version") != 1:
                raise ValueError("unsupported model artifact version")
            theta_history = values["theta_history"]
            precision_history = values["precision_history"]
            return cls(
                config=SGDLMConfig.from_dict(metadata["config"]),
                data=values["data"],
                parents=values["parents"],
                pdims=values["pdims"],
                theta=values["theta"],
                precision=values["precision"],
                weights=values["weights"],
                series_names=metadata["series_names"],
                exog_names=metadata["exog_names"],
                effective_sample_size=values["effective_sample_size"],
                theta_mean_history=values["theta_mean_history"],
                precision_mean_history=values["precision_mean_history"],
                theta_history=None if theta_history.size == 0 else theta_history,
                precision_history=None if precision_history.size == 0 else precision_history,
            )
