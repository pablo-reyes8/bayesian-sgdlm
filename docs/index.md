# Bayesian SGDLM

This package implements the sequential decouple/recouple strategy for Simultaneous Graphical
Dynamic Linear Models described by Gruber and West (2016).

| Layer | Module | Responsibility |
|---|---|---|
| Specification | `config.py`, `design.py` | Validation, lags, exogenous variables, parent graph |
| Inference | `core.py` | Normal-gamma updates, importance weights, VB moment matching |
| User model | `model.py`, `results.py` | Fit, forecast, IRFs, artifacts |
| Interfaces | `cli.py`, `api.py`, `io.py` | YAML/CLI, HTTP, tabular files |

```python
from sgdlm import SGDLM, SGDLMConfig

model = SGDLM(SGDLMConfig(lags=2, draws=1000, seed=42))
result = model.fit(data, parents=parent_mask)
forecast = model.forecast(12)
```

The old notebook is retained as historical provenance. This is a CPU reference implementation;
it follows the paper's inference strategy but does not reproduce its CUDA acceleration.
