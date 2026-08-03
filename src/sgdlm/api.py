"""FastAPI application for fitted SGDLM resources."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Response, status

from . import __version__
from .model import SGDLM
from .registry import ModelRecord, ModelRegistry
from .schemas import (
    DynamicIRFResponse,
    ErrorResponse,
    FitRequest,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    IRFRequest,
    ModelListResponse,
    ModelSummary,
    ReadinessResponse,
    TerminalIRFResponse,
)

registry = ModelRegistry()

app = FastAPI(
    title="Bayesian SGDLM API",
    summary="Sequential Bayesian multivariate time-series inference",
    version=__version__,
    description=(
        "Fit Simultaneous Graphical Dynamic Linear Models, persist artifacts, "
        "and compute posterior forecasts and impulse responses."
    ),
    contact={"name": "Bayesian SGDLM maintainers"},
    license_info={"name": "Apache-2.0"},
)

ERROR_RESPONSES = {
    404: {"model": ErrorResponse, "description": "Model not found"},
    422: {"model": ErrorResponse, "description": "Invalid model or numerical request"},
}


@app.get("/health", response_model=HealthResponse, tags=["health"])
@app.get("/health/live", response_model=HealthResponse, tags=["health"])
def liveness() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


@app.get("/health/ready", response_model=ReadinessResponse, tags=["health"])
def readiness() -> ReadinessResponse:
    return ReadinessResponse(status="ok", version=__version__, storage=str(registry.root))


@app.post(
    "/v1/models",
    response_model=ModelSummary,
    status_code=status.HTTP_201_CREATED,
    responses={422: ERROR_RESPONSES[422]},
    tags=["models"],
)
def fit_model(payload: FitRequest) -> ModelSummary:
    try:
        model = SGDLM(payload.config.to_domain())
        result = model.fit(
            payload.data,
            parents=payload.parents,
            exog=payload.exog,
            series_names=payload.series_names,
            exog_names=payload.exog_names,
        )
        return registry.summary(registry.create(result))
    except (ValueError, RuntimeError, ArithmeticError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/v1/models", response_model=ModelListResponse, tags=["models"])
def list_models() -> ModelListResponse:
    return ModelListResponse(models=[registry.summary(record) for record in registry.list()])


@app.get(
    "/v1/models/{model_id}",
    response_model=ModelSummary,
    responses=ERROR_RESPONSES,
    tags=["models"],
)
def get_model(model_id: str) -> ModelSummary:
    return registry.summary(_record(model_id))


@app.delete(
    "/v1/models/{model_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: ERROR_RESPONSES[404]},
    tags=["models"],
)
def delete_model(model_id: str) -> Response:
    try:
        registry.delete(model_id)
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=404, detail="model not found") from error
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post(
    "/v1/models/{model_id}/forecast",
    response_model=ForecastResponse,
    responses=ERROR_RESPONSES,
    tags=["analysis"],
)
def forecast(model_id: str, payload: ForecastRequest) -> ForecastResponse:
    record = _record(model_id)
    try:
        output = record.model.forecast(
            payload.horizon,
            future_exog=payload.future_exog,
            simulations=payload.simulations,
            credible_level=payload.credible_level,
            seed=payload.seed,
        )
        result = record.model.result_
        assert result is not None
        return ForecastResponse(
            model_id=model_id,
            series_names=result.series_names,
            mean=output.mean.tolist(),
            lower=output.lower.tolist(),
            upper=output.upper.tolist(),
            simulations=output.simulations.tolist() if payload.include_simulations else None,
        )
    except (ValueError, RuntimeError, ArithmeticError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post(
    "/v1/models/{model_id}/irf",
    response_model=TerminalIRFResponse | DynamicIRFResponse,
    responses=ERROR_RESPONSES,
    tags=["analysis"],
)
def impulse_response(
    model_id: str, payload: IRFRequest
) -> TerminalIRFResponse | DynamicIRFResponse:
    record = _record(model_id)
    result = record.model.result_
    assert result is not None
    try:
        if payload.mode == "dynamic":
            output = record.model.dynamic_impulse_response(
                payload.horizon,
                payload.impulse,
                smoothing=payload.smoothing,
                smooth_window=payload.smooth_window,
                shock_scale=payload.shock_scale,
            )
            return DynamicIRFResponse(
                model_id=model_id,
                series_names=result.series_names,
                impulse=output.impulse,
                origins=output.origins.tolist(),
                raw=output.raw.tolist(),
                smoothed=None if output.smoothed is None else output.smoothed.tolist(),
                smoothing=payload.smoothing,
            )
        output = record.model.impulse_response(
            payload.horizon,
            payload.impulse,
            draws=payload.draws,
            credible_level=payload.credible_level,
            seed=payload.seed,
            shock_scale=payload.shock_scale,
        )
        return TerminalIRFResponse(
            model_id=model_id,
            series_names=result.series_names,
            impulse=output.impulse,
            mean=output.mean.tolist(),
            lower=output.lower.tolist(),
            upper=output.upper.tolist(),
        )
    except (ValueError, RuntimeError, ArithmeticError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


def _record(model_id: str) -> ModelRecord:
    try:
        return registry.get(model_id)
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=404, detail="model not found") from error
