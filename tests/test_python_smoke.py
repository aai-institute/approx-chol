import numpy as np

import approx_chol


def test_package_surface_and_basic_factorize_raw_roundtrip():
    assert approx_chol.__all__ == ["Config", "Factor", "factorize", "factorize_raw"]

    row_ptrs = np.array([0, 2, 4], dtype=np.uint32)
    col_indices = np.array([0, 1, 0, 1], dtype=np.uint32)
    values = np.array([2.0, -1.0, -1.0, 2.0], dtype=np.float64)

    factor = approx_chol.factorize_raw(row_ptrs, col_indices, values, 2)
    x = factor.solve(np.array([1.0, -1.0], dtype=np.float64))

    assert x.shape == (factor.n,)
    assert np.isfinite(x).all()
