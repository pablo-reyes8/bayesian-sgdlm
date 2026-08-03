from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import sgdlm.api as api_module
from sgdlm.api import app
from sgdlm.registry import ModelRegistry

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_module, "registry", ModelRegistry(tmp_path))


@pytest.fixture
def model_id(synthetic_var) -> str:
    response = client.post(
        "/v1/models",
        json={
            "data": synthetic_var.tolist(),
            "series_names": ["first", "second"],
            "config": {"draws": 20, "seed": 5},
        },
    )
    assert response.status_code == 201
    return response.json()["model_id"]


def test_health_endpoints() -> None:
    live = client.get("/health/live")
    ready = client.get("/health/ready")
    assert live.status_code == 200
    assert live.json()["status"] == "ok"
    assert ready.status_code == 200
    assert ready.json()["storage"]


def test_model_lifecycle(model_id: str) -> None:
    detail = client.get(f"/v1/models/{model_id}")
    listing = client.get("/v1/models")
    assert detail.status_code == 200
    assert detail.json()["series_names"] == ["first", "second"]
    assert listing.json()["models"][0]["model_id"] == model_id

    deleted = client.delete(f"/v1/models/{model_id}")
    assert deleted.status_code == 204
    assert client.get(f"/v1/models/{model_id}").status_code == 404


def test_forecast_endpoint(model_id: str) -> None:
    response = client.post(
        f"/v1/models/{model_id}/forecast",
        json={"horizon": 2, "simulations": 5},
    )
    assert response.status_code == 200
    assert response.json()["model_id"] == model_id
    assert len(response.json()["mean"]) == 2
    assert response.json()["simulations"] is None


def test_terminal_and_dynamic_irf_endpoints(model_id: str) -> None:
    terminal = client.post(
        f"/v1/models/{model_id}/irf",
        json={"horizon": 2, "impulse": "first", "mode": "terminal", "draws": 10},
    )
    dynamic = client.post(
        f"/v1/models/{model_id}/irf",
        json={
            "horizon": 2,
            "impulse": 0,
            "mode": "dynamic",
            "smoothing": "savgol",
        },
    )
    assert terminal.status_code == 200
    assert terminal.json()["mode"] == "terminal"
    assert dynamic.status_code == 200
    assert dynamic.json()["mode"] == "dynamic"
    assert dynamic.json()["smoothed"] is not None


def test_schema_rejects_unknown_fields() -> None:
    response = client.post(
        "/v1/models/not-a-model/forecast",
        json={"horizon": 2, "unknown": True},
    )
    assert response.status_code == 422
