"""Public SGDLM estimator, forecasting, and impulse responses."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

from .config import SGDLMConfig
from .core import (
    effective_sample_size,
    evolve_covariance,
    importance_weights,
    initial_state,
    update_and_sample,
    variational_decouple,
)
from .design import (
    FloatArray,
    as_2d_float,
    block_dimensions,
    equation_design,
    lag_vector,
    validate_parents,
)
from .results import DynamicIRFResult, FitResult, ForecastResult, IRFResult


class SGDLM:
    """Simultaneous Graphical Dynamic Linear Model.

    The estimator performs independent normal-gamma DLM updates, recouples
    posterior draws with the ``|I - Gamma|`` importance correction, and uses
    variational moment matching to continue the sequential filter.
    """

    def __init__(self, config: SGDLMConfig | None = None) -> None:
        self.config = config or SGDLMConfig()
        self.result_: FitResult | None = None

    def fit(
        self,
        data: ArrayLike,
        *,
        parents: ArrayLike | None = None,
        exog: ArrayLike | None = None,
        series_names: Sequence[str] | None = None,
        exog_names: Sequence[str] | None = None,
    ) -> FitResult:
        values = as_2d_float(data)
        n_obs, n_series = values.shape
        if n_obs <= self.config.lags + 2:
            raise ValueError("data has too few rows for the requested lag order")
        parent_mask = validate_parents(parents, n_series)
        exogenous = None if exog is None else as_2d_float(exog, name="exog")
        if exogenous is not None and exogenous.shape[0] != n_obs:
            raise ValueError("exog must contain the same number of rows as data")
        n_exog = 0 if exogenous is None else exogenous.shape[1]
        names = _validate_names(series_names, n_series, "y")
        x_names = _validate_names(exog_names, n_exog, "x")

        pdims = block_dimensions(n_series, self.config.lags, n_exog, parent_mask)
        state = initial_state(values, parent_mask, n_exog, self.config)
        rng = np.random.default_rng(self.config.seed)
        ess_values = np.empty(n_obs - self.config.lags)
        theta_history: list[FloatArray] = []
        precision_history: list[FloatArray] = []
        theta_mean_history: list[FloatArray] = []
        precision_mean_history: list[FloatArray] = []
        own_size = 1 + n_series * self.config.lags + n_exog

        theta = np.empty((self.config.draws, int(pdims[-1])))
        precision = np.empty((self.config.draws, n_series))
        for output_index, t in enumerate(range(self.config.lags, n_obs)):
            for j in range(n_series):
                evolved = evolve_covariance(
                    state.covariance[j],
                    own_size,
                    self.config.delta_state,
                    self.config.delta_parent,
                )
                design = equation_design(values, t, j, self.config.lags, parent_mask, exogenous)
                local_theta, local_precision, _, _ = update_and_sample(
                    state.mean[j],
                    evolved,
                    state.degrees[j],
                    state.scale[j],
                    design,
                    values[t, j],
                    self.config.beta,
                    self.config.draws,
                    rng,
                )
                theta[:, pdims[j] : pdims[j + 1]] = local_theta
                precision[:, j] = local_precision
            weights = importance_weights(theta, pdims, parent_mask, own_size)
            ess_values[output_index] = effective_sample_size(weights)
            theta_mean_history.append(weights @ theta)
            precision_mean_history.append(weights @ precision)
            if self.config.store_history:
                theta_history.append(theta.copy())
                precision_history.append(precision.copy())
            state = variational_decouple(theta, precision, weights, pdims, state.degrees)

        self.result_ = FitResult(
            config=self.config,
            data=values,
            parents=parent_mask,
            pdims=pdims,
            theta=theta.copy(),
            precision=precision.copy(),
            weights=weights.copy(),
            series_names=names,
            exog_names=x_names,
            effective_sample_size=ess_values,
            theta_mean_history=np.asarray(theta_mean_history),
            precision_mean_history=np.asarray(precision_mean_history),
            theta_history=np.asarray(theta_history) if theta_history else None,
            precision_history=(np.asarray(precision_history) if precision_history else None),
        )
        return self.result_

    def forecast(
        self,
        horizon: int,
        *,
        future_exog: ArrayLike | None = None,
        simulations: int | None = None,
        credible_level: float = 0.9,
        seed: int | None = None,
    ) -> ForecastResult:
        result = self._require_result()
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        if not 0.0 < credible_level < 1.0:
            raise ValueError("credible_level must be in (0, 1)")
        count = simulations or result.config.draws
        if count < 1:
            raise ValueError("simulations must be positive")
        future = _validate_future_exog(future_exog, horizon, len(result.exog_names))
        rng = np.random.default_rng(result.config.seed if seed is None else seed)
        selected = rng.choice(result.theta.shape[0], size=count, replace=True, p=result.weights)
        paths = np.empty((count, horizon, result.data.shape[1]))
        for path_index, draw_index in enumerate(selected):
            history = result.data.copy()
            for step in range(horizon):
                mean, covariance = _forecast_moments(
                    result,
                    result.theta[draw_index],
                    result.precision[draw_index],
                    history,
                    None if future is None else future[step],
                )
                observation = mean + _covariance_cholesky(covariance) @ rng.standard_normal(
                    result.data.shape[1]
                )
                paths[path_index, step] = observation
                history = np.vstack((history, observation))
        tail = (1.0 - credible_level) / 2.0
        return ForecastResult(
            mean=paths.mean(axis=0),
            lower=np.quantile(paths, tail, axis=0),
            upper=np.quantile(paths, 1.0 - tail, axis=0),
            simulations=paths,
        )

    def impulse_response(
        self,
        horizon: int,
        impulse: int | str,
        *,
        draws: int | None = None,
        credible_level: float = 0.9,
        seed: int | None = None,
        shock_scale: str = "innovation_sd",
    ) -> IRFResult:
        result = self._require_result()
        if horizon < 0:
            raise ValueError("horizon cannot be negative")
        impulse_index = _series_index(impulse, result.series_names)
        count = draws or result.config.draws
        rng = np.random.default_rng(result.config.seed if seed is None else seed)
        selected = rng.choice(result.theta.shape[0], size=count, replace=True, p=result.weights)
        responses = np.empty((count, horizon + 1, result.data.shape[1]))
        for output_index, draw_index in enumerate(selected):
            responses[output_index] = _draw_irf(
                result,
                result.theta[draw_index],
                result.precision[draw_index],
                horizon,
                impulse_index,
                shock_scale,
            )
        tail = (1.0 - credible_level) / 2.0
        return IRFResult(
            mean=responses.mean(axis=0),
            lower=np.quantile(responses, tail, axis=0),
            upper=np.quantile(responses, 1.0 - tail, axis=0),
            impulse=impulse_index,
        )

    def dynamic_impulse_response(
        self,
        horizon: int,
        impulse: int | str,
        *,
        smoothing: str | None = None,
        smooth_window: int = 5,
        shock_scale: str = "innovation_sd",
    ) -> DynamicIRFResult:
        """Compute an IRF at every origin using future time-varying coefficients.

        Unlike the terminal IRF, parameters are not frozen: the response at
        origin ``t`` and horizon ``h`` uses the filtered SGDLM coefficients at
        time ``t + h``. Smoothing, when requested, only acts across origins.
        """

        result = self._require_result()
        if horizon < 0:
            raise ValueError("horizon cannot be negative")
        available = result.theta_mean_history.shape[0]
        if horizon >= available:
            raise ValueError("horizon must be shorter than the filtered sample")
        impulse_index = _series_index(impulse, result.series_names)
        n_origins = available - horizon
        raw = np.empty((n_origins, horizon + 1, result.data.shape[1]))
        for origin in range(n_origins):
            raw[origin] = _dynamic_draw_irf(result, origin, horizon, impulse_index, shock_scale)
        smoothed = _smooth_dynamic_irf(raw, smoothing, smooth_window)
        origins = np.arange(
            result.config.lags,
            result.config.lags + n_origins,
            dtype=np.int64,
        )
        return DynamicIRFResult(raw, smoothed, origins, impulse_index, smoothing)

    @classmethod
    def load(cls, path: str) -> SGDLM:
        result = FitResult.load(path)
        model = cls(result.config)
        model.result_ = result
        return model

    def _require_result(self) -> FitResult:
        if self.result_ is None:
            raise RuntimeError("fit or load a model before requesting results")
        return self.result_


def _structural_components(
    result: FitResult, theta: FloatArray
) -> tuple[FloatArray, list[FloatArray]]:
    n_series = result.data.shape[1]
    n_exog = len(result.exog_names)
    own_size = 1 + n_series * result.config.lags + n_exog
    gamma = np.zeros((n_series, n_series))
    lag_matrices = [np.zeros((n_series, n_series)) for _ in range(result.config.lags)]
    for j in range(n_series):
        block = theta[result.pdims[j] : result.pdims[j + 1]]
        gamma[j, result.parents[j]] = block[own_size:]
        for lag in range(result.config.lags):
            start = 1 + lag * n_series
            lag_matrices[lag][j] = block[start : start + n_series]
    impact = np.linalg.inv(np.eye(n_series) - gamma)
    return impact, [impact @ matrix for matrix in lag_matrices]


def _forecast_moments(
    result: FitResult,
    theta: FloatArray,
    precision: FloatArray,
    history: FloatArray,
    future_exog: FloatArray | None,
) -> tuple[FloatArray, FloatArray]:
    n_series = history.shape[1]
    n_exog = len(result.exog_names)
    own_size = 1 + n_series * result.config.lags + n_exog
    gamma = np.zeros((n_series, n_series))
    structural_mean = np.empty(n_series)
    regressors = [np.ones(1), lag_vector(history, history.shape[0], result.config.lags)]
    if future_exog is not None:
        regressors.append(future_exog)
    x = np.concatenate(regressors)
    for j in range(n_series):
        block = theta[result.pdims[j] : result.pdims[j + 1]]
        structural_mean[j] = x @ block[:own_size]
        gamma[j, result.parents[j]] = block[own_size:]
    impact = np.linalg.inv(np.eye(n_series) - gamma)
    mean = impact @ structural_mean
    covariance = impact @ np.diag(1.0 / precision) @ impact.T
    return mean, _positive_definite_covariance(covariance)


def _draw_irf(
    result: FitResult,
    theta: FloatArray,
    precision: FloatArray,
    horizon: int,
    impulse: int,
    shock_scale: str,
) -> FloatArray:
    impact, reduced_lags = _structural_components(result, theta)
    response = np.zeros((horizon + 1, result.data.shape[1]))
    response[0] = _shock_vector(impact, precision, impulse, shock_scale)
    for step in range(1, horizon + 1):
        for lag, coefficient in enumerate(reduced_lags, start=1):
            if step >= lag:
                response[step] += coefficient @ response[step - lag]
    return response


def _dynamic_draw_irf(
    result: FitResult,
    origin: int,
    horizon: int,
    impulse: int,
    shock_scale: str,
) -> FloatArray:
    n_series = result.data.shape[1]
    response = np.zeros((horizon + 1, n_series))
    impact, _ = _structural_components(result, result.theta_mean_history[origin])
    response[0] = _shock_vector(
        impact,
        result.precision_mean_history[origin],
        impulse,
        shock_scale,
    )
    for step in range(1, horizon + 1):
        _, reduced_lags = _structural_components(result, result.theta_mean_history[origin + step])
        for lag, coefficient in enumerate(reduced_lags, start=1):
            if step >= lag:
                response[step] += coefficient @ response[step - lag]
    return response


def _shock_vector(
    impact: FloatArray, precision: FloatArray, impulse: int, shock_scale: str
) -> FloatArray:
    column = impact[:, impulse]
    if shock_scale == "innovation_sd":
        return np.asarray(column / np.sqrt(precision[impulse]), dtype=float)
    if shock_scale == "unit":
        return column.copy()
    if shock_scale == "unit_effect":
        norm = np.linalg.norm(column)
        if norm <= np.finfo(float).eps:
            raise ArithmeticError("cannot normalize a zero structural impact")
        return column / norm
    raise ValueError("shock_scale must be one of: innovation_sd, unit, unit_effect")


def _positive_definite_covariance(covariance: FloatArray) -> FloatArray:
    symmetric = (covariance + covariance.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    floor = scale * 1e-12
    projected = (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T
    return (projected + projected.T) / 2.0


def _covariance_cholesky(covariance: FloatArray) -> FloatArray:
    scale = max(float(np.max(np.abs(np.diag(covariance)))), 1.0)
    identity = np.eye(covariance.shape[0])
    for relative_jitter in (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4):
        try:
            return np.linalg.cholesky(covariance + identity * scale * relative_jitter)
        except np.linalg.LinAlgError:
            continue
    raise ArithmeticError("predictive covariance is numerically indefinite")


def _smooth_dynamic_irf(raw: FloatArray, smoothing: str | None, window: int) -> FloatArray | None:
    if smoothing in (None, "none"):
        return None
    if window < 3:
        raise ValueError("smooth_window must be at least 3")
    if smoothing == "moving_average":
        kernel = np.ones(window) / window
        padded = np.pad(raw, ((window // 2, window - 1 - window // 2), (0, 0), (0, 0)), mode="edge")
        return np.apply_along_axis(
            lambda values: np.convolve(values, kernel, mode="valid"), 0, padded
        )
    if smoothing == "gaussian":
        return np.asarray(gaussian_filter1d(raw, sigma=window / 4.0, axis=0, mode="nearest"))
    if smoothing == "savgol":
        actual = min(window if window % 2 else window + 1, raw.shape[0])
        if actual % 2 == 0:
            actual -= 1
        if actual < 3:
            raise ValueError("not enough origins for Savitzky-Golay smoothing")
        return np.asarray(
            savgol_filter(raw, window_length=actual, polyorder=min(2, actual - 1), axis=0)
        )
    raise ValueError("smoothing must be one of: none, moving_average, gaussian, savgol")


def _validate_names(names: Sequence[str] | None, expected: int, prefix: str) -> list[str]:
    output = [f"{prefix}{index + 1}" for index in range(expected)] if names is None else list(names)
    if len(output) != expected or len(set(output)) != expected:
        raise ValueError(f"names for {prefix} must be unique and have length {expected}")
    return output


def _validate_future_exog(values: ArrayLike | None, horizon: int, n_exog: int) -> FloatArray | None:
    if n_exog == 0:
        if values is not None:
            raise ValueError(
                "future_exog was provided but the fitted model has no exogenous variables"
            )
        return None
    if values is None:
        raise ValueError(
            "future_exog is required because the fitted model uses exogenous variables"
        )
    output = np.asarray(values, dtype=float)
    if output.shape != (horizon, n_exog) or not np.isfinite(output).all():
        raise ValueError(f"future_exog must have shape ({horizon}, {n_exog})")
    return output


def _series_index(value: int | str, names: list[str]) -> int:
    if isinstance(value, str):
        if value not in names:
            raise ValueError(f"unknown impulse series: {value}")
        return names.index(value)
    if value < 0 or value >= len(names):
        raise ValueError(f"impulse must be between 0 and {len(names) - 1}")
    return value
