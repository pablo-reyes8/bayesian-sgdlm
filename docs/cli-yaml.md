# CLI and YAML

Every workflow accepts flags or YAML. Explicit flags override YAML values.

## Fit

```bash
python scripts/fit.py --config config/fit.yml
```

```yaml
data:
  input: data/DATA.xlsx
  columns: [Prod_man, Desempleo_des, Inflacion_month]
  exog_columns: []
  parents: examples/parents.json
model:
  lags: 3
  draws: 1000
  beta: 0.945
  delta_state: 0.975
  delta_parent: 0.98
  seed: 42
output:
  model: artifacts/model.npz
```

All `model` keys map to `SGDLMConfig`; unknown keys fail instead of being silently ignored.

## Forecast

```yaml
forecast:
  model: artifacts/model.npz
  horizon: 12
  simulations: 2000
  credible_level: 0.9
  seed: 42
  output: artifacts/forecast.json
  include_simulations: false
```

Run it with `python scripts/forecast.py --config config/forecast.yml`. Add `future_exog` when
required.

## Dynamic IRF

```yaml
irf:
  model: artifacts/model.npz
  horizon: 12
  impulse: Inflacion_month
  dynamic: true
  smoothing: gaussian
  smooth_window: 7
  shock_scale: innovation_sd
  output: artifacts/dynamic-irf.json
```

Run it with `python scripts/irf.py --config config/dynamic-irf.yml`. Paths are interpreted from the
current working directory. `scripts/irf.py --config config/irf.yml` computes the terminal variant.
