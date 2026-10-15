import warnings

import numpy as np
import pytest

import approx_chol
from tests._laplacians import grid_laplacian


def _grid_residual(config: approx_chol.Config) -> float:
    lap = grid_laplacian(4, 4)
    b = np.zeros(lap.shape[0])
    b[0], b[-1] = 1.0, -1.0
    x = approx_chol.factorize(lap, config=config).solve(b)
    return float(np.linalg.norm(lap @ x - b) / np.linalg.norm(b))


def _grid_solution(config: approx_chol.Config) -> np.ndarray:
    lap = grid_laplacian(8, 8)
    b = np.zeros(lap.shape[0])
    b[0], b[-1] = 1.0, -1.0
    return approx_chol.factorize(lap, config=config).solve(b)


def test_package_surface_and_basic_factorize_raw_roundtrip():
    assert approx_chol.__all__ == ["Config", "Factor", "factorize", "factorize_raw"]

    row_ptrs = np.array([0, 2, 4], dtype=np.uint32)
    col_indices = np.array([0, 1, 0, 1], dtype=np.uint32)
    values = np.array([2.0, -1.0, -1.0, 2.0], dtype=np.float64)

    factor = approx_chol.factorize_raw(row_ptrs, col_indices, values, 2)
    x = factor.solve(np.array([1.0, -1.0], dtype=np.float64))

    assert x.shape == (factor.shape[0],)
    assert np.isfinite(x).all()


def test_config_defaults_and_getters():
    default = approx_chol.Config()
    assert (default.seed, default.split, default.dense_up_to) == (0, None, 24)

    config = approx_chol.Config(seed=7, split=3, dense_up_to=0)
    assert (config.seed, config.split, config.dense_up_to) == (7, 3, 0)


def test_no_config_and_a_default_config_factor_alike():
    lap = grid_laplacian(8, 8)
    b = np.zeros(lap.shape[0])
    b[0], b[-1] = 1.0, -1.0
    np.testing.assert_array_equal(
        approx_chol.factorize(lap).solve(b),
        approx_chol.factorize(lap, approx_chol.Config()).solve(b),
    )


def test_dense_up_to_reaches_the_factorization():
    assert _grid_residual(approx_chol.Config(dense_up_to=16)) < 1e-12
    assert _grid_residual(approx_chol.Config(dense_up_to=15)) > 1e-3


def test_seed_and_split_reach_the_factorization():
    base = _grid_solution(approx_chol.Config(seed=0, dense_up_to=0))
    np.testing.assert_array_equal(
        base, _grid_solution(approx_chol.Config(seed=0, dense_up_to=0))
    )
    assert not np.array_equal(
        base, _grid_solution(approx_chol.Config(seed=1, dense_up_to=0))
    )
    assert not np.array_equal(
        base, _grid_solution(approx_chol.Config(seed=0, split=2, dense_up_to=0))
    )


def test_a_clamped_deficit_warns():
    # |a_01| = 2 exceeds both diagonals, and no positive scaling repairs that.
    row_ptrs = np.array([0, 2, 4], dtype=np.uint32)
    col_indices = np.array([0, 1, 0, 1], dtype=np.uint32)
    values = np.array([1.0, -2.0, -2.0, 1.0], dtype=np.float64)

    with pytest.warns(RuntimeWarning, match="fell short of diagonal dominance"):
        factor = approx_chol.factorize_raw(row_ptrs, col_indices, values, 2)
    assert factor.shape == (2, 2)


def test_a_dominant_matrix_does_not_warn():
    lap = grid_laplacian(4, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        approx_chol.factorize(lap)
