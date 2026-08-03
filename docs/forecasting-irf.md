# Forecasting and Impulse Responses

## Forecast

Forecast paths resample terminal state and precision draws with importance weights. Each horizon
forms the structural mean and covariance, draws an observation, appends it to the lag history, and
continues recursively. Future exogenous values are required when exogenous variables were fitted.

## Terminal IRF

`impulse_response` freezes each terminal posterior draw. Lag matrices are reconstructed by taking
the `q` lag coefficients from every equation row; this fixes the indexing error in the historical
notebook, which reshaped contiguous global-state chunks containing intercepts and dummies.

Three shock scales are supported:

- `innovation_sd`: structural innovation scaled by `1 / sqrt(lambda)` (default).
- `unit`: structural innovation of size one.
- `unit_effect`: normalize each impact-matrix column to Euclidean norm one, matching the old
  notebook's convention.

## Evolving IRF by origin

`dynamic_impulse_response` creates an innovation at every filtered origin `t`. Horizon zero uses
`A[t]`; the response at horizon `h` uses reduced-form lag matrices filtered at `t + h`. Coefficients
therefore evolve along calendar time instead of being frozen. Output axes are:

```text
[origin date, response horizon, response series]
```

This is related in interpretation to a time-varying local projection, but is not an LP estimator:
it preserves the recursive SGDLM law rather than fitting a separate regression at every horizon.

## Smoothing

`moving_average`, `gaussian`, and `savgol` operate only across origins, independently for every
horizon and response. Results always contain `raw`; `smoothed` is separate. Smoothing is a
descriptive choice, not Bayesian inference, and must not be used to manufacture significance.
