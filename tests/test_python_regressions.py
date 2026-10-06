import numpy as np
import pytest

import approx_chol


def test_factorize_raw_non_contiguous_arrays_raise_value_error():
    row_ptrs = np.array([0, 99, 2, 99, 4], dtype=np.uint32)[::2]
    col_indices = np.array([0, 99, 1, 99, 0, 99, 1], dtype=np.uint32)[::2]
    values = np.array([2.0, 0.0, -1.0, 0.0, -1.0, 0.0, 2.0], dtype=np.float64)[::2]

    assert not row_ptrs.flags["C_CONTIGUOUS"]
    assert not col_indices.flags["C_CONTIGUOUS"]
    assert not values.flags["C_CONTIGUOUS"]

    with pytest.raises(ValueError):
        approx_chol.factorize_raw(row_ptrs, col_indices, values, 2)
