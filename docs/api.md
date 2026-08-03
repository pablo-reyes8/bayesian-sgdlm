# HTTP API

Start with `python scripts/serve_api.py` or `docker compose up --build`. OpenAPI documentation is
available at `/docs` and the machine-readable schema at `/openapi.json`.

## Health

- `GET /health` and `GET /health/live`: process liveness.
- `GET /health/ready`: artifact storage readiness and location.

## Models

- `POST /v1/models`: fit and persist a model; returns HTTP 201 and a UUID.
- `GET /v1/models`: list persisted models.
- `GET /v1/models/{model_id}`: model dimensions and ESS summary.
- `DELETE /v1/models/{model_id}`: remove the artifact and return HTTP 204.

Fit request:

```json
{
  "data": [[0.1, 0.2], [0.2, 0.1], [0.25, 0.12], [0.3, 0.15]],
  "parents": [[false, true], [true, false]],
  "series_names": ["output", "inflation"],
  "config": {"lags": 1, "draws": 100, "seed": 42}
}
```

Artifacts are stored under `SGDLM_MODEL_DIR`; the default is a dedicated system temporary
directory. The registry is thread-safe and lazily reloads artifacts after process restarts.

## Analysis

- `POST /v1/models/{model_id}/forecast`: posterior forecasts and credible intervals.
- `POST /v1/models/{model_id}/irf`: terminal or dynamic impulse responses.

Forecast request:

```json
{"horizon": 12, "simulations": 1000, "credible_level": 0.9}
```

IRF request:

```json
{
  "horizon": 12,
  "impulse": "inflation",
  "mode": "dynamic",
  "shock_scale": "innovation_sd",
  "smoothing": "gaussian",
  "smooth_window": 7
}
```

All schemas reject unknown fields. Invalid dimensions and numerical requests return HTTP 422;
unknown model IDs return HTTP 404. The typed response schemas are separated for terminal and
dynamic IRFs so clients can discriminate on `mode`.
