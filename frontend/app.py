"""Streamlit workbench for SGDLM analysis."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sgdlm import SGDLM, SGDLMConfig

st.set_page_config(page_title="Bayesian SGDLM", page_icon=None, layout="wide")
st.title("Bayesian SGDLM")


def read_upload(uploaded_file) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(uploaded_file)
    if suffix == ".parquet":
        return pd.read_parquet(uploaded_file)
    raise ValueError("Supported formats: CSV, Excel, and Parquet")


def download_artifact(model: SGDLM) -> bytes:
    result = model.result_
    if result is None:
        return b""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.npz"
        result.save(path)
        return path.read_bytes()


with st.sidebar:
    uploaded = st.file_uploader("Data", type=["csv", "xlsx", "xls", "parquet"])

if uploaded is None:
    st.info("Upload a dataset to begin.")
    st.stop()

try:
    frame = read_upload(uploaded)
except (ValueError, OSError) as error:
    st.error(str(error))
    st.stop()

numeric_columns = frame.select_dtypes(include="number").columns.tolist()
if not numeric_columns:
    st.error("The dataset has no numeric columns.")
    st.stop()

signature = (uploaded.name, uploaded.size)
if st.session_state.get("data_signature") != signature:
    st.session_state.data_signature = signature
    st.session_state.pop("model", None)
    st.session_state.pop("forecast", None)
    st.session_state.pop("irf", None)

with st.sidebar:
    endogenous = st.multiselect("Endogenous series", numeric_columns, default=numeric_columns)
    available_exog = [column for column in numeric_columns if column not in endogenous]
    exogenous = st.multiselect("Exogenous variables", available_exog)
    st.subheader("Model")
    lags = st.number_input("Lags", min_value=1, max_value=24, value=1, step=1)
    draws = st.number_input("Posterior draws", min_value=10, max_value=100_000, value=500, step=10)
    beta = st.number_input("Variance discount", min_value=0.01, max_value=1.0, value=0.95)
    delta_state = st.number_input("State discount", min_value=0.01, max_value=1.0, value=0.98)
    delta_parent = st.number_input("Parent discount", min_value=0.01, max_value=1.0, value=0.98)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)

data_tab, diagnostics_tab, forecast_tab, irf_tab = st.tabs(
    ["Data and graph", "Fit and diagnostics", "Forecast", "Impulse responses"]
)

with data_tab:
    st.dataframe(frame, use_container_width=True, height=320)
    if endogenous:
        graph_key = "|".join(endogenous)
        if st.session_state.get("graph_key") != graph_key:
            st.session_state.graph_key = graph_key
            st.session_state.parent_graph = pd.DataFrame(
                False, index=endogenous, columns=endogenous
            )
        edited_graph = st.data_editor(
            st.session_state.parent_graph,
            use_container_width=True,
            key="parent_graph_editor",
        )
        graph = edited_graph.to_numpy(dtype=bool)
        np.fill_diagonal(graph, False)
        st.session_state.parent_graph = pd.DataFrame(graph, index=endogenous, columns=endogenous)
    else:
        graph = np.empty((0, 0), dtype=bool)

with diagnostics_tab:
    fit_requested = st.button("Fit model", type="primary", use_container_width=False)
    if fit_requested:
        if not endogenous:
            st.error("Select at least one endogenous series.")
        else:
            config = SGDLMConfig(
                lags=int(lags),
                draws=int(draws),
                beta=float(beta),
                delta_state=float(delta_state),
                delta_parent=float(delta_parent),
                seed=int(seed),
            )
            try:
                with st.spinner("Fitting model"):
                    model = SGDLM(config)
                    model.fit(
                        frame[endogenous].to_numpy(),
                        parents=graph,
                        exog=frame[exogenous].to_numpy() if exogenous else None,
                        series_names=endogenous,
                        exog_names=exogenous,
                    )
                    st.session_state.model = model
                    st.session_state.pop("forecast", None)
                    st.session_state.pop("irf", None)
            except (ValueError, RuntimeError, ArithmeticError) as error:
                st.error(str(error))

    model: SGDLM | None = st.session_state.get("model")
    if model is not None and model.result_ is not None:
        result = model.result_
        metric_columns = st.columns(4)
        metric_columns[0].metric("Observations", result.data.shape[0])
        metric_columns[1].metric("Series", result.data.shape[1])
        metric_columns[2].metric("Parameters", int(result.pdims[-1]))
        metric_columns[3].metric("Terminal ESS", f"{result.effective_sample_size[-1]:.1f}")
        ess = pd.DataFrame({"ESS": result.effective_sample_size})
        st.line_chart(ess, height=260)
        st.download_button(
            "Download model",
            data=download_artifact(model),
            file_name="model.sgdlm.npz",
            mime="application/octet-stream",
        )

with forecast_tab:
    model = st.session_state.get("model")
    if model is None or model.result_ is None:
        st.info("Fit a model first.")
    else:
        result = model.result_
        forecast_horizon = st.number_input("Forecast horizon", 1, 500, 12)
        simulations = st.number_input("Forecast simulations", 1, 100_000, int(draws))
        credible_level = st.slider("Credible level", 0.5, 0.99, 0.9)
        future_values = None
        if result.exog_names:
            future_upload = st.file_uploader(
                "Future exogenous data", type=["csv", "xlsx", "xls", "parquet"]
            )
            if future_upload is not None:
                future_frame = read_upload(future_upload)
                future_values = future_frame[result.exog_names].to_numpy()
        if st.button("Run forecast", type="primary"):
            try:
                with st.spinner("Generating forecast"):
                    st.session_state.forecast = model.forecast(
                        int(forecast_horizon),
                        future_exog=future_values,
                        simulations=int(simulations),
                        credible_level=float(credible_level),
                        seed=int(seed),
                    )
            except (ValueError, RuntimeError, ArithmeticError) as error:
                st.error(str(error))
        output = st.session_state.get("forecast")
        if output is not None:
            forecast_frame = pd.DataFrame(output.mean, columns=result.series_names)
            forecast_frame.index.name = "horizon"
            st.line_chart(forecast_frame)
            st.dataframe(forecast_frame, use_container_width=True)
            st.download_button(
                "Download forecast",
                data=json.dumps(output.to_dict(), indent=2),
                file_name="forecast.json",
                mime="application/json",
            )

with irf_tab:
    model = st.session_state.get("model")
    if model is None or model.result_ is None:
        st.info("Fit a model first.")
    else:
        result = model.result_
        left, right = st.columns(2)
        mode = left.selectbox("IRF mode", ["terminal", "dynamic"])
        impulse = right.selectbox("Impulse", result.series_names)
        irf_horizon = left.number_input("IRF horizon", 0, 500, 12)
        shock_scale = right.selectbox("Shock scale", ["innovation_sd", "unit", "unit_effect"])
        smoothing = None
        smooth_window = 5
        if mode == "dynamic":
            smoothing_choice = left.selectbox(
                "Smoothing", ["none", "moving_average", "gaussian", "savgol"]
            )
            smoothing = None if smoothing_choice == "none" else smoothing_choice
            smooth_window = right.number_input("Smoothing window", 3, 101, 5, step=2)
        if st.button("Compute IRF", type="primary"):
            try:
                with st.spinner("Computing impulse responses"):
                    if mode == "dynamic":
                        st.session_state.irf = model.dynamic_impulse_response(
                            int(irf_horizon),
                            impulse,
                            smoothing=smoothing,
                            smooth_window=int(smooth_window),
                            shock_scale=shock_scale,
                        )
                    else:
                        st.session_state.irf = model.impulse_response(
                            int(irf_horizon),
                            impulse,
                            credible_level=0.9,
                            seed=int(seed),
                            shock_scale=shock_scale,
                        )
                    st.session_state.irf_mode = mode
            except (ValueError, RuntimeError, ArithmeticError) as error:
                st.error(str(error))
        output = st.session_state.get("irf")
        if output is not None:
            response_name = st.selectbox("Response", result.series_names)
            response_index = result.series_names.index(response_name)
            if st.session_state.get("irf_mode") == "dynamic":
                selected_horizon = st.slider("Response horizon", 0, output.raw.shape[1] - 1, 0)
                chart = pd.DataFrame(
                    {"raw": output.raw[:, selected_horizon, response_index]},
                    index=output.origins,
                )
                if output.smoothed is not None:
                    chart["smoothed"] = output.smoothed[:, selected_horizon, response_index]
            else:
                chart = pd.DataFrame(
                    {
                        "mean": output.mean[:, response_index],
                        "lower": output.lower[:, response_index],
                        "upper": output.upper[:, response_index],
                    }
                )
                chart.index.name = "horizon"
            st.line_chart(chart)
            st.download_button(
                "Download IRF",
                data=json.dumps(output.to_dict(), indent=2),
                file_name="irf.json",
                mime="application/json",
            )
