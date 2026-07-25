import numpy as np

import approx_chol


def test_package_surface_and_basic_factorize_raw_roundtrip():
    # Every name the type stub declares at package level must be reachable from the
    # package, not just from the extension module the other tests load directly.
    assert approx_chol.__all__ == [
        "Backend",
        "Config",
        "ExactFailure",
        "Factor",
        "factorize",
        "factorize_raw",
    ]
    assert approx_chol.Config(
        backend=approx_chol.Backend.ExactBelow(
            24, approx_chol.ExactFailure.FallBackToApproximate
        )
    )

    row_ptrs = np.array([0, 2, 4], dtype=np.uint32)
    col_indices = np.array([0, 1, 0, 1], dtype=np.uint32)
    values = np.array([2.0, -1.0, -1.0, 2.0], dtype=np.float64)

    factor = approx_chol.factorize_raw(row_ptrs, col_indices, values, 2)
    x = factor.solve(np.array([1.0, -1.0], dtype=np.float64))

    assert x.shape == (factor.shape[0],)
    assert np.isfinite(x).all()
