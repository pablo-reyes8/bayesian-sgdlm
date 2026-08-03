"""Data validation and design matrix construction."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def as_2d_float(data: ArrayLike, *, name: str = "data") -> FloatArray:
    values = np.asarray(data, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if values.shape[0] < 3 or values.shape[1] < 1:
        raise ValueError(f"{name} must contain at least 3 rows and 1 column")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains missing or non-finite values")
    return values


def validate_parents(parents: ArrayLike | None, n_series: int) -> BoolArray:
    if parents is None:
        return np.zeros((n_series, n_series), dtype=bool)
    mask = np.asarray(parents, dtype=bool)
    if mask.shape != (n_series, n_series):
        raise ValueError(f"parents must have shape ({n_series}, {n_series})")
    mask = mask.copy()
    np.fill_diagonal(mask, False)
    return mask


def lag_vector(data: FloatArray, t: int, lags: int) -> FloatArray:
    """Return [y[t-1], ..., y[t-lags]] with each lag in series order."""

    return np.concatenate([data[t - lag] for lag in range(1, lags + 1)])


def equation_design(
    data: FloatArray,
    t: int,
    equation: int,
    lags: int,
    parents: BoolArray,
    exog: FloatArray | None,
) -> FloatArray:
    components = [np.ones(1), lag_vector(data, t, lags)]
    if exog is not None:
        components.append(exog[t])
    components.append(data[t, parents[equation]])
    return np.concatenate(components)


def block_dimensions(
    n_series: int, lags: int, n_exog: int, parents: BoolArray
) -> NDArray[np.int64]:
    sizes = 1 + n_series * lags + n_exog + parents.sum(axis=1)
    return np.concatenate(([0], np.cumsum(sizes))).astype(np.int64)
