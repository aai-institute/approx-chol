import warnings

import numpy as np
import pytest

import approx_chol
from tests._laplacians import grid_laplacian


def cancelling_path():
    """A path whose middle pivot cancels to zero, so the exact backend gives up."""
    big, small = 2.0**500, 2.0**-500
    return (
        np.array([0, 2, 5, 7], dtype=np.uint32),
        np.array([0, 1, 0, 1, 2, 1, 2], dtype=np.uint32),
        np.array(
            [big, -big, -big, big + small, -small, -small, small], dtype=np.float64
        ),
    )


def test_package_surface_and_basic_factorize_raw_roundtrip():
    assert approx_chol.__all__ == [
        "Backend",
        "Config",
        "DenseFailure",
        "ExactFailure",
        "Factor",
        "Fallback",
        "factorize",
        "factorize_raw",
    ]

    row_ptrs = np.array([0, 2, 4], dtype=np.uint32)
    col_indices = np.array([0, 1, 0, 1], dtype=np.uint32)
    values = np.array([2.0, -1.0, -1.0, 2.0], dtype=np.float64)

    factor = approx_chol.factorize_raw(row_ptrs, col_indices, values, 2)
    x = factor.solve(np.array([1.0, -1.0], dtype=np.float64))

    assert x.shape == (factor.shape[0],)
    assert np.isfinite(x).all()


def test_backend_reaches_the_factorization():
    lap = grid_laplacian(4, 4)
    b = np.zeros(lap.shape[0])
    b[0], b[-1] = 1.0, -1.0

    assert approx_chol.Config().backend == approx_chol.Backend.ExactBelow(
        24, approx_chol.ExactFailure.FallBackToApproximate
    )

    def residual(backend):
        config = approx_chol.Config(backend=backend)
        x = approx_chol.factorize(lap, config=config).solve(b)
        return np.linalg.norm(lap @ x - b) / np.linalg.norm(b)

    assert (
        residual(
            approx_chol.Backend.ExactBelow(
                24, approx_chol.ExactFailure.FallBackToApproximate
            )
        )
        < 1e-12
    )
    assert residual(approx_chol.Backend.Approximate()) > 1e-3


def test_exact_failure_error_propagates_as_value_error():
    row_ptrs, col_indices, values = cancelling_path()

    backend = approx_chol.Backend.ExactBelow(24, approx_chol.ExactFailure.Error)
    with pytest.raises(ValueError, match="exact dense Cholesky failed at vertex 1"):
        approx_chol.factorize_raw(
            row_ptrs,
            col_indices,
            values,
            3,
            config=approx_chol.Config(backend=backend),
        )


def test_a_fallback_is_both_warned_about_and_inspectable():
    row_ptrs, col_indices, values = cancelling_path()

    with pytest.warns(RuntimeWarning, match="fell back to approximate elimination"):
        factor = approx_chol.factorize_raw(row_ptrs, col_indices, values, 3)

    (fallback,) = factor.fallbacks
    assert isinstance(fallback, approx_chol.Fallback.InvalidPivot)
    assert fallback.vertex == 1
    assert fallback.failure == approx_chol.DenseFailure.NonPositivePivot


def test_a_clean_factorization_neither_warns_nor_reports():
    lap = grid_laplacian(4, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        factor = approx_chol.factorize(lap)
    assert factor.fallbacks == []
