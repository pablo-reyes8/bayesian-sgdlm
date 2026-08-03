from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sgdlm.cli import main


def test_cli_fit_forecast_and_dynamic_irf(synthetic_var, tmp_path: Path) -> None:
    data = tmp_path / "data.csv"
    model = tmp_path / "model.npz"
    forecast = tmp_path / "forecast.json"
    irf = tmp_path / "irf.json"
    pd.DataFrame(synthetic_var, columns=["a", "b"]).to_csv(data, index=False)

    assert main(["fit", str(data), "--output", str(model), "--draws", "20", "--seed", "3"]) == 0
    assert main(["forecast", str(model), "--horizon", "2", "--output", str(forecast)]) == 0
    assert (
        main(
            [
                "irf",
                str(model),
                "--horizon",
                "2",
                "--impulse",
                "a",
                "--dynamic",
                "--smoothing",
                "moving_average",
                "--output",
                str(irf),
            ]
        )
        == 0
    )

    assert len(json.loads(forecast.read_text())["mean"]) == 2
    assert json.loads(irf.read_text())["smoothing"] == "moving_average"


def test_cli_yaml_fit(synthetic_var, tmp_path: Path) -> None:
    data = tmp_path / "data.csv"
    artifact = tmp_path / "yaml-model.npz"
    config = tmp_path / "fit.yml"
    pd.DataFrame(synthetic_var, columns=["a", "b"]).to_csv(data, index=False)
    config.write_text(
        f"""data:
  input: {data}
  columns: [a, b]
model:
  lags: 1
  draws: 20
  seed: 9
output:
  model: {artifact}
""",
        encoding="utf-8",
    )
    assert main(["fit", "--config", str(config)]) == 0
    assert artifact.exists()
