"""Tabular input/output helpers shared by the CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray


def read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(source)
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(source)
    elif suffix == ".parquet":
        frame = pd.read_parquet(source)
    else:
        raise ValueError("input must be CSV, Excel, or Parquet")
    if frame.empty:
        raise ValueError("input table is empty")
    return frame


def read_parent_mask(path: str | Path | None, columns: list[str]) -> NDArray[np.bool_] | None:
    if path is None:
        return None
    source = Path(path)
    if source.suffix.lower() == ".json":
        values = np.asarray(json.loads(source.read_text(encoding="utf-8")), dtype=bool)
    else:
        values = read_table(source).to_numpy(dtype=bool)
    expected = (len(columns), len(columns))
    if values.shape != expected:
        raise ValueError(f"parent mask must have shape {expected}")
    return values


def write_json(path: str | Path, values: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(values, indent=2), encoding="utf-8")


def read_yaml(path: str | Path | None) -> dict[str, object]:
    if path is None:
        return {}
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if values is None:
        return {}
    if not isinstance(values, dict):
        raise ValueError("YAML configuration must contain a mapping at its root")
    return values
