from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_starts_without_data() -> None:
    app = Path(__file__).parents[1] / "frontend" / "app.py"
    test_app = AppTest.from_file(str(app)).run(timeout=20)
    assert not test_app.exception
    assert test_app.title[0].value == "Bayesian SGDLM"
    assert test_app.info[0].value == "Upload a dataset to begin."
