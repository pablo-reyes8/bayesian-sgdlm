from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sgdlm import SGDLM, FitResult, SGDLMConfig


def fitted_model(data: np.ndarray) -> SGDLM:
    model = SGDLM(SGDLMConfig(lags=1, draws=40, seed=7))
    model.fit(
        data,
        parents=[[False, True], [True, False]],
        series_names=["first", "second"],
    )
    return model


def test_fit_is_reproducible_and_has_valid_weights(synthetic_var: np.ndarray) -> None:
    first = fitted_model(synthetic_var).result_
    second = fitted_model(synthetic_var).result_
    assert first is not None and second is not None
    np.testing.assert_allclose(first.theta, second.theta)
    np.testing.assert_allclose(first.weights.sum(), 1.0)
    assert np.all(first.precision > 0)
    assert np.all(first.effective_sample_size >= 1)


def test_forecast_shapes_and_intervals(synthetic_var: np.ndarray) -> None:
    forecast = fitted_model(synthetic_var).forecast(4, simulations=30, seed=99)
    assert forecast.mean.shape == (4, 2)
    assert forecast.simulations.shape == (30, 4, 2)
    assert np.all(forecast.lower <= forecast.mean)
    assert np.all(forecast.mean <= forecast.upper)


def test_static_and_dynamic_irfs(synthetic_var: np.ndarray) -> None:
    model = fitted_model(synthetic_var)
    static = model.impulse_response(5, "first", draws=20, seed=1)
    dynamic = model.dynamic_impulse_response(5, "first", smoothing="gaussian", smooth_window=5)
    assert static.mean.shape == (6, 2)
    assert dynamic.raw.shape == (synthetic_var.shape[0] - 1 - 5, 6, 2)
    assert dynamic.smoothed is not None
    assert dynamic.smoothed.shape == dynamic.raw.shape
    assert not np.shares_memory(dynamic.raw, dynamic.smoothed)


def test_artifact_round_trip(synthetic_var: np.ndarray, tmp_path: Path) -> None:
    model = fitted_model(synthetic_var)
    path = tmp_path / "model.npz"
    assert model.result_ is not None
    model.result_.save(path)
    loaded = FitResult.load(path)
    np.testing.assert_allclose(loaded.theta, model.result_.theta)
    np.testing.assert_allclose(loaded.theta_mean_history, model.result_.theta_mean_history)
    assert loaded.series_names == ["first", "second"]


def test_exogenous_forecast_requires_future_values(synthetic_var: np.ndarray) -> None:
    exog = np.linspace(0, 1, len(synthetic_var))[:, None]
    model = SGDLM(SGDLMConfig(draws=20, seed=2))
    model.fit(synthetic_var, exog=exog, exog_names=["trend"])
    with pytest.raises(ValueError, match="future_exog is required"):
        model.forecast(2)
    output = model.forecast(2, future_exog=[[1.1], [1.2]], simulations=5)
    assert output.mean.shape == (2, 2)


def test_legacy_unit_effect_normalizes_impact(synthetic_var: np.ndarray) -> None:
    response = fitted_model(synthetic_var).impulse_response(
        2, 0, draws=10, seed=4, shock_scale="unit_effect"
    )
    assert np.isfinite(response.mean).all()
