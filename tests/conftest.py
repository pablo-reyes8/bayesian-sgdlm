from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def synthetic_var() -> np.ndarray:
    rng = np.random.default_rng(123)
    values = np.zeros((45, 2))
    coefficient = np.array([[0.55, 0.10], [-0.08, 0.35]])
    for t in range(1, len(values)):
        values[t] = coefficient @ values[t - 1] + rng.normal(scale=0.2, size=2)
    return values
