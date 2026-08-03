"""Numerical building blocks for decouple/recouple SGDLM inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import digamma

from .config import SGDLMConfig

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(slots=True)
class FilterState:
    mean: list[FloatArray]
    covariance: list[FloatArray]
    degrees: FloatArray
    scale: FloatArray


def ar1_moments(data: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Estimate stable AR(1) anchors and residual variances for each series."""

    n_series = data.shape[1]
    rho = np.zeros(n_series)
    variance = np.zeros(n_series)
    x = np.column_stack((np.ones(data.shape[0] - 1), data[:-1]))
    for j in range(n_series):
        local_x = x[:, [0, j + 1]]
        coef, *_ = np.linalg.lstsq(local_x, data[1:, j], rcond=None)
        rho[j] = np.clip(coef[1], -0.99, 0.99)
        residual = data[1:, j] - local_x @ coef
        variance[j] = max(float(np.mean(residual**2)), np.finfo(float).eps)
    return rho, variance


def initial_state(
    data: FloatArray,
    parents: BoolArray,
    n_exog: int,
    config: SGDLMConfig,
) -> FilterState:
    n_series = data.shape[1]
    rho, residual_variance = ar1_moments(data)
    means: list[FloatArray] = []
    covariances: list[FloatArray] = []
    for j in range(n_series):
        size = 1 + n_series * config.lags + n_exog + int(parents[j].sum())
        mean = np.zeros(size)
        mean[1 + j] = rho[j]
        variances = np.empty(size)
        variances[0] = residual_variance[j] * (config.prior_overall * config.prior_intercept) ** 2
        for lag in range(1, config.lags + 1):
            for source in range(n_series):
                index = 1 + (lag - 1) * n_series + source
                relative_scale = residual_variance[j] / residual_variance[source]
                cross = 1.0 if source == j else config.prior_cross
                variances[index] = (
                    relative_scale
                    * (config.prior_overall * cross / lag**config.prior_lag_decay) ** 2
                )
        exog_start = 1 + n_series * config.lags
        variances[exog_start : exog_start + n_exog] = config.prior_exogenous**2
        variances[exog_start + n_exog :] = config.prior_exogenous**2
        means.append(mean)
        covariances.append(np.diag(np.maximum(variances, 1e-10)))
    return FilterState(
        mean=means,
        covariance=covariances,
        degrees=np.full(n_series, config.prior_df),
        scale=residual_variance,
    )


def evolve_covariance(
    covariance: FloatArray, own_size: int, delta_state: float, delta_parent: float
) -> FloatArray:
    discounts = np.full(covariance.shape[0], delta_parent)
    discounts[:own_size] = delta_state
    multiplier = 1.0 / np.sqrt(discounts)
    evolved = covariance * np.outer(multiplier, multiplier)
    return (evolved + evolved.T) / 2.0


def update_and_sample(
    mean: FloatArray,
    covariance: FloatArray,
    degrees: float,
    scale: float,
    design: FloatArray,
    observation: float,
    beta: float,
    draws: int,
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray, float, float]:
    gain_numerator = covariance @ design
    forecast_factor = max(float(design @ gain_numerator + scale), 1e-12)
    error = float(observation - design @ mean)
    updated_degrees = beta * degrees + 1.0
    adjustment = max((beta * degrees + error**2 / forecast_factor) / updated_degrees, 1e-12)
    updated_scale = max(scale * adjustment, 1e-12)
    updated_mean = mean + gain_numerator * error / forecast_factor
    updated_covariance = covariance - np.outer(gain_numerator, gain_numerator) / forecast_factor
    updated_covariance = _positive_definite(updated_covariance * adjustment / updated_scale)

    precision = rng.gamma(
        shape=updated_degrees / 2.0,
        scale=2.0 / (updated_degrees * updated_scale),
        size=draws,
    )
    standard = rng.standard_normal((draws, mean.size))
    chol = np.linalg.cholesky(updated_covariance)
    theta = updated_mean + (standard @ chol.T) / np.sqrt(precision[:, None])
    return theta, precision, updated_degrees, updated_scale


def importance_weights(
    theta: FloatArray,
    pdims: NDArray[np.int64],
    parents: BoolArray,
    own_size: int,
) -> FloatArray:
    draws, n_series = theta.shape[0], parents.shape[0]
    log_weights = np.empty(draws)
    for r in range(draws):
        gamma = np.zeros((n_series, n_series))
        for j in range(n_series):
            block = theta[r, pdims[j] : pdims[j + 1]]
            gamma[j, parents[j]] = block[own_size:]
        sign, logdet = np.linalg.slogdet(np.eye(n_series) - gamma)
        log_weights[r] = logdet if sign != 0 and np.isfinite(logdet) else -np.inf
    finite = np.isfinite(log_weights)
    if not finite.any():
        return np.full(draws, 1.0 / draws)
    weights = np.zeros(draws)
    weights[finite] = np.exp(log_weights[finite] - np.max(log_weights[finite]))
    total = weights.sum()
    return weights / total if total > 0 else np.full(draws, 1.0 / draws)


def variational_decouple(
    theta: FloatArray,
    precision: FloatArray,
    weights: FloatArray,
    pdims: NDArray[np.int64],
    fallback_degrees: FloatArray,
) -> FilterState:
    n_series = precision.shape[1]
    means: list[FloatArray] = []
    covariances: list[FloatArray] = []
    degrees = np.empty(n_series)
    scales = np.empty(n_series)
    for j in range(n_series):
        block = theta[:, pdims[j] : pdims[j + 1]]
        expected_precision = max(float(weights @ precision[:, j]), 1e-12)
        mean = (weights[:, None] * precision[:, [j]] * block).sum(axis=0)
        mean /= expected_precision
        errors = block - mean
        covariance_kernel = np.einsum(
            "r,r,ri,rj->ij", weights, precision[:, j], errors, errors, optimize=True
        )
        covariance_kernel = _positive_definite(covariance_kernel)
        inv_kernel = np.linalg.pinv(covariance_kernel, hermitian=True)
        distances = np.einsum("ri,ij,rj->r", errors, inv_kernel, errors)
        d_value = float(weights @ (precision[:, j] * distances))
        df = _match_degrees(
            block.shape[1],
            d_value,
            expected_precision,
            float(weights @ np.log(np.maximum(precision[:, j], 1e-300))),
            fallback_degrees[j],
        )
        scale = max((df + block.shape[1] - d_value) / (df * expected_precision), 1e-12)
        means.append(mean)
        covariances.append(_positive_definite(covariance_kernel * scale))
        degrees[j] = df
        scales[j] = scale
    return FilterState(means, covariances, degrees, scales)


def effective_sample_size(weights: FloatArray) -> float:
    return float(1.0 / np.sum(weights**2))


def _match_degrees(
    dimension: int,
    d_value: float,
    expected_precision: float,
    expected_log_precision: float,
    fallback: float,
) -> float:
    def equation(value: float) -> float:
        positive = value + dimension - d_value
        if positive <= 0:
            return np.nan
        return float(
            np.log(positive)
            - digamma(value / 2.0)
            - np.log(2.0 * expected_precision)
            + expected_log_precision
            - (dimension - d_value) / value
        )

    lower = max(2.000001, d_value - dimension + 1e-6)
    grid = np.geomspace(lower, 1e5, 100)
    values = np.asarray([equation(value) for value in grid])
    for left, right, f_left, f_right in zip(
        grid[:-1], grid[1:], values[:-1], values[1:], strict=True
    ):
        if np.isfinite(f_left) and np.isfinite(f_right) and f_left * f_right <= 0:
            return float(brentq(equation, left, right, maxiter=100))
    return max(float(fallback), 2.000001)


def _positive_definite(matrix: FloatArray) -> FloatArray:
    symmetric = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    floor = max(float(np.max(np.abs(eigenvalues))) * 1e-10, 1e-12)
    projected = (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T
    return np.asarray(projected, dtype=np.float64)
