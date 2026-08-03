# Streamlit Workbench

Start the workbench with:

```bash
python scripts/serve_frontend.py
```

The application is available at `http://localhost:8501` and uses the public `SGDLM` class directly.
No inference code is duplicated in the frontend.

## Workflow

1. Upload CSV, Excel, or Parquet data.
2. Assign numeric columns as endogenous or exogenous variables.
3. Edit the contemporaneous parent matrix.
4. Set lags, draws, discount factors, and seed.
5. Fit and inspect the sequential effective sample size.
6. Generate forecasts or terminal/dynamic IRFs.
7. Download model artifacts and JSON results.

Dynamic IRFs expose the response series and horizon separately, allowing raw and smoothed paths to
be compared without replacing the underlying estimate.
