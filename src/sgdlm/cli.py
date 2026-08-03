"""Command-line interface for training and using SGDLM models."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from .config import SGDLMConfig
from .io import read_parent_mask, read_table, read_yaml, write_json
from .model import SGDLM
from .results import DynamicIRFResult, IRFResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sgdlm", description="Fit and use SGDLM models")
    parser.add_argument("--version", action="version", version="%(prog)s 0.2.0")
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser("fit", help="fit a model from tabular data")
    fit.add_argument("input", nargs="?", help="CSV, Excel, or Parquet input")
    fit.add_argument("--config", help="YAML configuration file")
    fit.add_argument("--output", help="model artifact path")
    fit.add_argument(
        "--columns", help="comma-separated endogenous columns; defaults to numeric columns"
    )
    fit.add_argument("--exog-columns", help="comma-separated exogenous columns")
    fit.add_argument("--parents", help="JSON/CSV/Excel q by q parent mask")
    _add_model_arguments(fit)

    forecast = commands.add_parser("forecast", help="forecast from a fitted model")
    forecast.add_argument("model", nargs="?", help="saved .npz model artifact")
    forecast.add_argument("--config", help="YAML configuration file")
    forecast.add_argument("--horizon", type=int)
    forecast.add_argument("--future-exog", help="table containing future exogenous values")
    forecast.add_argument("--simulations", type=int)
    forecast.add_argument("--credible-level", type=float)
    forecast.add_argument("--seed", type=int)
    forecast.add_argument("--output")
    forecast.add_argument(
        "--include-simulations", action=argparse.BooleanOptionalAction, default=None
    )

    irf = commands.add_parser("irf", help="compute posterior impulse responses")
    irf.add_argument("model", nargs="?", help="saved .npz model artifact")
    irf.add_argument("--config", help="YAML configuration file")
    irf.add_argument("--horizon", type=int)
    irf.add_argument("--impulse", help="zero-based index or series name")
    irf.add_argument("--draws", type=int)
    irf.add_argument("--credible-level", type=float)
    irf.add_argument("--seed", type=int)
    irf.add_argument(
        "--unit-shock",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="use a unit rather than one-SD shock",
    )
    irf.add_argument(
        "--shock-scale",
        choices=("innovation_sd", "unit", "unit_effect"),
        help="structural shock normalization",
    )
    irf.add_argument(
        "--dynamic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="compute one evolving IRF per time origin",
    )
    irf.add_argument(
        "--smoothing",
        choices=("none", "moving_average", "gaussian", "savgol"),
        default=None,
    )
    irf.add_argument("--smooth-window", type=int)
    irf.add_argument("--output")

    serve = commands.add_parser("serve", help="start the HTTP API")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "fit":
        return _fit(args)
    if args.command == "forecast":
        return _forecast(args)
    if args.command == "irf":
        return _irf(args)
    if args.command == "serve":
        try:
            import uvicorn
        except ImportError as error:
            raise SystemExit("Install the 'api' extra to use serve: pip install .[api]") from error
        uvicorn.run("sgdlm.api:app", host=args.host, port=args.port)
        return 0
    raise AssertionError("unreachable command")


def _fit(args: argparse.Namespace) -> int:
    document = read_yaml(args.config)
    data_config = _section(document, "data")
    model_config = _section(document, "model")
    output_config = _section(document, "output")
    input_path = args.input or data_config.get("input")
    if not input_path:
        raise ValueError("an input path is required as an argument or data.input in YAML")
    frame = read_table(str(input_path))
    exog_columns = _column_value(args.exog_columns, data_config.get("exog_columns"))
    columns = _column_value(args.columns, data_config.get("columns"))
    if not columns:
        columns = [
            column
            for column in frame.select_dtypes(include="number").columns
            if column not in exog_columns
        ]
    if not columns:
        raise ValueError("no endogenous numeric columns were selected")
    parent_path = args.parents or data_config.get("parents")
    parents = read_parent_mask(None if parent_path is None else str(parent_path), columns)
    explicit = {
        key: getattr(args, key)
        for key in ("lags", "draws", "beta", "delta_state", "delta_parent", "seed", "store_history")
        if getattr(args, key) is not None
    }
    config = SGDLMConfig(**{**model_config, **explicit})
    model = SGDLM(config)
    result = model.fit(
        frame[columns].to_numpy(),
        parents=parents,
        exog=frame[exog_columns].to_numpy() if exog_columns else None,
        series_names=columns,
        exog_names=exog_columns,
    )
    output_path = args.output or output_config.get("model") or "model.sgdlm.npz"
    result.save(str(output_path))
    summary = {
        "model": str(Path(str(output_path))),
        "observations": len(frame),
        "series": columns,
        "parameters": int(result.pdims[-1]),
        "terminal_ess": float(result.effective_sample_size[-1]),
    }
    print(json.dumps(summary, indent=2))
    return 0


def _forecast(args: argparse.Namespace) -> int:
    document = read_yaml(args.config)
    settings = _section(document, "forecast")
    model_path = args.model or settings.get("model")
    horizon = args.horizon if args.horizon is not None else settings.get("horizon")
    if not model_path or horizon is None:
        raise ValueError("model and horizon are required as arguments or in forecast YAML")
    model = SGDLM.load(str(model_path))
    result = model.result_
    assert result is not None
    future = None
    future_path = args.future_exog or settings.get("future_exog")
    if future_path:
        frame = read_table(str(future_path))
        future = frame[result.exog_names].to_numpy()
    forecast = model.forecast(
        int(cast(int | str, horizon)),
        future_exog=future,
        simulations=cast(int | None, _value(args.simulations, settings, "simulations", None)),
        credible_level=float(
            cast(float, _value(args.credible_level, settings, "credible_level", 0.9))
        ),
        seed=cast(int | None, _value(args.seed, settings, "seed", None)),
    )
    output_path = args.output or settings.get("output") or "forecast.json"
    include = bool(_value(args.include_simulations, settings, "include_simulations", False))
    write_json(str(output_path), forecast.to_dict(include_simulations=include))
    print(json.dumps({"output": output_path, "horizon": horizon}, indent=2))
    return 0


def _irf(args: argparse.Namespace) -> int:
    document = read_yaml(args.config)
    settings = _section(document, "irf")
    model_path = args.model or settings.get("model")
    horizon = args.horizon if args.horizon is not None else settings.get("horizon")
    impulse_value = args.impulse if args.impulse is not None else settings.get("impulse")
    if not model_path or horizon is None or impulse_value is None:
        raise ValueError("model, horizon, and impulse are required as arguments or in IRF YAML")
    model = SGDLM.load(str(model_path))
    impulse_text = str(impulse_value)
    impulse: int | str = int(impulse_text) if impulse_text.lstrip("-").isdigit() else impulse_text
    dynamic = bool(_value(args.dynamic, settings, "dynamic", False))
    unit_shock = bool(_value(args.unit_shock, settings, "unit_shock", False))
    shock_scale = str(
        _value(
            args.shock_scale,
            settings,
            "shock_scale",
            "unit" if unit_shock else "innovation_sd",
        )
    )
    response: DynamicIRFResult | IRFResult
    if dynamic:
        response = model.dynamic_impulse_response(
            int(cast(int | str, horizon)),
            impulse,
            smoothing=cast(str | None, _value(args.smoothing, settings, "smoothing", None)),
            smooth_window=int(cast(int, _value(args.smooth_window, settings, "smooth_window", 5))),
            shock_scale=shock_scale,
        )
    else:
        response = model.impulse_response(
            int(cast(int | str, horizon)),
            impulse,
            draws=cast(int | None, _value(args.draws, settings, "draws", None)),
            credible_level=float(
                cast(float, _value(args.credible_level, settings, "credible_level", 0.9))
            ),
            seed=cast(int | None, _value(args.seed, settings, "seed", None)),
            shock_scale=shock_scale,
        )
    output_path = args.output or settings.get("output") or "irf.json"
    write_json(str(output_path), response.to_dict())
    print(json.dumps({"output": output_path, "horizon": horizon}, indent=2))
    return 0


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lags", type=int)
    parser.add_argument("--draws", type=int)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--delta-state", type=float)
    parser.add_argument("--delta-parent", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--store-history", action=argparse.BooleanOptionalAction, default=None)


def _columns(value: str | None) -> list[str]:
    return [] if not value else [column.strip() for column in value.split(",") if column.strip()]


def _column_value(cli_value: str | None, yaml_value: object) -> list[str]:
    if cli_value is not None:
        return _columns(cli_value)
    if yaml_value is None:
        return []
    if not isinstance(yaml_value, list) or not all(isinstance(item, str) for item in yaml_value):
        raise ValueError("YAML columns must be a list of strings")
    return yaml_value


def _section(document: dict[str, object], name: str) -> dict[str, object]:
    value = document.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"YAML section '{name}' must be a mapping")
    return value


def _value(cli_value: object, settings: dict[str, object], key: str, default: object) -> object:
    return cli_value if cli_value is not None else settings.get(key, default)


if __name__ == "__main__":
    raise SystemExit(main())
