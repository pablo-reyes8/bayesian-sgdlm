# Contributing

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
ruff check .
ruff format --check .
```

Add focused tests for numerical changes. Fix random seeds in tests, document changes to the
state ordering, and compare forecast/IRF behavior before changing an inference equation.

## Pull requests

Keep pull requests scoped, explain statistical assumptions, and include a reproducible example
for behavior changes. Never commit private datasets or fitted artifacts.
