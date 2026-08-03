from __future__ import annotations

import numpy as np
import pytest

from sgdlm import SGDLMConfig
from sgdlm.design import block_dimensions, equation_design, validate_parents


def test_config_rejects_invalid_discounts() -> None:
    with pytest.raises(ValueError, match="beta"):
        SGDLMConfig(beta=0)


def test_design_orders_lags_exog_and_parents() -> None:
    data = np.arange(15, dtype=float).reshape(5, 3)
    parents = validate_parents([[0, 1, 0], [0, 0, 0], [1, 0, 0]], 3)
    exog = np.arange(5, dtype=float)[:, None]
    design = equation_design(data, 2, 0, 2, parents, exog)
    np.testing.assert_array_equal(
        design,
        np.concatenate(([1.0], data[1], data[0], exog[2], [data[2, 1]])),
    )
    np.testing.assert_array_equal(block_dimensions(3, 2, 1, parents), [0, 9, 17, 26])


def test_parent_diagonal_is_removed() -> None:
    mask = validate_parents(np.ones((2, 2)), 2)
    np.testing.assert_array_equal(mask, [[False, True], [True, False]])
