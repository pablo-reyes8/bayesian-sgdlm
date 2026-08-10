# Bayesian SGDLM

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status: research alpha](https://img.shields.io/badge/status-research%20alpha-orange.svg)](CHANGELOG.md)
[![Docker](https://img.shields.io/badge/container-Docker-2496ED?logo=docker&logoColor=white)](Dockerfile)


A Python implementation of the Simultaneous Graphical Dynamic Linear Model (SGDLM)
decouple/recouple algorithm from Gruber and West (2016). It provides sequential Bayesian
inference, posterior forecasts, static and time-varying impulse responses, a CLI, and an HTTP API.

> **Status:** research software. Validate graph structure, priors, convergence diagnostics, and
> forecast calibration for the intended application before using results in decisions.

## Model

For series `j`, the package estimates

```text
y[j,t] = x[j,t]' phi[j,t] + y[parents(j),t]' gamma[j,t] + error[j,t]
```

where `x` contains an intercept, lags of every endogenous series, and optional general exogenous
regressors. Independent normal-gamma DLM updates are recoupled with the exact
`|I - Gamma_t|` importance weight, then approximated by variational normal-gamma margins for the
next sequential update. The contemporaneous graph is fixed during one fit, while coefficients and
observation precisions evolve over time.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

For library-only use, run `python -m pip install -e .`. Excel, API, frontend, and Parquet support
are exposed as the `excel`, `api`, `frontend`, and `parquet` extras.

## Python API

```python
import numpy as np
from sgdlm import SGDLM, SGDLMConfig

y = np.loadtxt("data.csv", delimiter=",", skiprows=1)
parents = np.array(
    [
        [False, True],  # series 2 is a contemporaneous parent of equation 1
        [True, False],
    ]
)

model = SGDLM(SGDLMConfig(lags=2, draws=1_000, seed=42))
fit = model.fit(y, parents=parents, series_names=["output", "inflation"])

forecast = model.forecast(horizon=12, simulations=2_000, credible_level=0.9)
terminal_irf = model.impulse_response(horizon=24, impulse="output")
dynamic_irf = model.dynamic_impulse_response(
    horizon=12,
    impulse="output",
    smoothing="savgol",
    smooth_window=7,
)

fit.save("model.sgdlm.npz")
restored = SGDLM.load("model.sgdlm.npz")
```

`forecast.mean/lower/upper` have shape `(horizon, series)`. The terminal IRF has shape
`(horizon + 1, response_series)` and includes posterior bands.

### Two IRF definitions

`impulse_response` is the conventional terminal IRF. Each posterior draw freezes the last
estimated `Gamma`, VAR coefficients, and precisions over the requested horizon.

`dynamic_impulse_response` applies a structural innovation at every in-sample origin. At horizon
`h`, it propagates the response with the SGDLM coefficients filtered at calendar time `t + h`.
The result is a surface with shape `(origin, horizon + 1, response_series)`. This is related in
interpretation to time-varying local projections, but it is not an LP estimator: it preserves the
recursive SGDLM law of motion instead of estimating a separate regression at every horizon.

Optional `moving_average`, `gaussian`, and `savgol` smoothers operate only across origins. Both
`raw` and `smoothed` arrays are returned, so smoothing never replaces the estimated response.

## Scripts and YAML

Fit all numeric columns in a CSV, Excel, or Parquet file:

```bash
python scripts/fit.py data/DATA.xlsx \
  --output artifacts/model.npz \
  --lags 3 --draws 1000 --seed 42 \
  --parents parents.json
```

`parents.json` is a `q x q` boolean array; entry `[j][i]` means series `i` is a contemporaneous
parent of equation `j`. Its diagonal is always removed. Use `--columns a,b,c` and
`--exog-columns policy_dummy,trend` to control the data roles.

```bash
python scripts/forecast.py artifacts/model.npz --horizon 12 --output artifacts/forecast.json
python scripts/irf.py artifacts/model.npz --horizon 24 --impulse 0 --output artifacts/irf.json
python scripts/irf.py artifacts/model.npz --horizon 12 --impulse 0 --dynamic \
  --smoothing gaussian --smooth-window 7 --output artifacts/dynamic-irf.json
```

YAML configurations live in `config/`:

```bash
python scripts/fit.py --config config/fit.yml
python scripts/forecast.py --config config/forecast.yml
python scripts/irf.py --config config/irf.yml
python scripts/irf.py --config config/dynamic-irf.yml
```

Run any script with `--help` for all options.

## HTTP API

```bash
docker compose up --build
curl http://localhost:8000/health
curl -X POST http://localhost:8000/v1/models \
  -H "Content-Type: application/json" \
  --data @examples/request.json
```

Interactive OpenAPI documentation is available at `http://localhost:8000/docs`. Model fitting
returns a UUID. Forecasts and IRFs use `/v1/models/{model_id}/forecast` and
`/v1/models/{model_id}/irf`; artifacts are persisted in `SGDLM_MODEL_DIR` or the system temporary
directory.

## Streamlit

```bash
python scripts/serve_frontend.py
```

Open `http://localhost:8501`. The workbench supports tabular uploads, endogenous/exogenous column
selection, contemporaneous graph editing, hyperparameters, fitting, ESS diagnostics, forecasts,
terminal IRFs, dynamic IRFs, smoothing, and result downloads. Docker Compose starts the API and
frontend together on ports `8000` and `8501`.

## Data Contract

- Rows are ordered observations; columns are series.
- Inputs must be finite numeric values with more than `lags + 2` rows.
- Exogenous training data must align row-for-row with endogenous data.
- Future exogenous values are required for every forecast step when the model was fitted with them.
- Scale variables before fitting when their magnitudes differ substantially.
- A sparse, substantively justified parent graph is preferable to a dense graph.

## Development

```bash
pytest --cov=sgdlm
ruff check .
ruff format --check .
docker build -t bayesian-sgdlm .
```


## Reference

Gruber, L. F. and West, M. (2016). GPU-Accelerated Bayesian Learning and Forecasting in
Simultaneous Graphical Dynamic Linear Models. *Bayesian Analysis*, 11(1), 125-149.
[doi:10.1214/15-BA946](https://doi.org/10.1214/15-BA946). A copy is included at
[`paper/15-BA946.pdf`](paper/15-BA946.pdf).

## License

Apache License 2.0. See [LICENSE](LICENSE).
