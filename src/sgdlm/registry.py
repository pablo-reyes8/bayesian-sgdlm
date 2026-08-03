"""Thread-safe artifact registry used by the HTTP service."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from uuid import UUID, uuid4

from .model import SGDLM
from .results import FitResult
from .schemas import ModelSummary


@dataclass(frozen=True, slots=True)
class ModelRecord:
    model_id: str
    created_at: datetime
    model: SGDLM


class ModelRegistry:
    """Persist model artifacts and lazily cache loaded estimators."""

    def __init__(self, root: str | Path | None = None) -> None:
        configured = root or os.getenv("SGDLM_MODEL_DIR")
        self.root = Path(configured) if configured else Path(tempfile.gettempdir()) / "sgdlm-api"
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, ModelRecord] = {}
        self._lock = RLock()

    def create(self, result: FitResult) -> ModelRecord:
        model_id = str(uuid4())
        created_at = datetime.now(timezone.utc)
        result.save(self._path(model_id))
        model = SGDLM(result.config)
        model.result_ = result
        record = ModelRecord(model_id, created_at, model)
        with self._lock:
            self._cache[model_id] = record
        return record

    def get(self, model_id: str) -> ModelRecord:
        normalized = str(UUID(model_id))
        with self._lock:
            cached = self._cache.get(normalized)
            if cached is not None:
                return cached
        path = self._path(normalized)
        if not path.exists():
            raise KeyError(normalized)
        model = SGDLM.load(str(path))
        created_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        record = ModelRecord(normalized, created_at, model)
        with self._lock:
            self._cache[normalized] = record
        return record

    def list(self) -> list[ModelRecord]:
        records: list[ModelRecord] = []
        for path in sorted(
            self.root.glob("*.npz"), key=lambda item: item.stat().st_mtime, reverse=True
        ):
            try:
                records.append(self.get(path.stem))
            except (KeyError, ValueError):
                continue
        return records

    def delete(self, model_id: str) -> None:
        normalized = str(UUID(model_id))
        path = self._path(normalized)
        if not path.exists():
            raise KeyError(normalized)
        path.unlink()
        with self._lock:
            self._cache.pop(normalized, None)

    def summary(self, record: ModelRecord) -> ModelSummary:
        result = record.model.result_
        if result is None:
            raise RuntimeError("registered model has no fit result")
        return ModelSummary(
            model_id=record.model_id,
            created_at=record.created_at,
            observations=result.data.shape[0],
            series=result.data.shape[1],
            parameters=int(result.pdims[-1]),
            series_names=result.series_names,
            exog_names=result.exog_names,
            terminal_ess=float(result.effective_sample_size[-1]),
            minimum_ess=float(result.effective_sample_size.min()),
        )

    def _path(self, model_id: str) -> Path:
        return self.root / f"{model_id}.npz"
