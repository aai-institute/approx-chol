import warnings

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.linalg import block_diag

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
    assert approx_chol.__all__ == [
        "Backend",
        "Config",
        "Factor",
        "SchurMode",
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


def test_config_defaults_and_getters():
    default = approx_chol.Config()
    assert (default.seed, default.split) == (0, None)
    assert default.backend == approx_chol.Backend.ExactBelow(max_dim=24)
    assert default.complement == approx_chol.SchurMode.Same()

    config = approx_chol.Config(
        seed=7,
        split=3,
        backend=approx_chol.Backend.Approximate(),
        complement=approx_chol.SchurMode.Approximate(split=2),
    )
    assert (config.seed, config.split) == (7, 3)
    assert config.backend == approx_chol.Backend.Approximate()
    assert config.complement == approx_chol.SchurMode.Approximate(split=2)


def test_no_config_and_a_default_config_factor_alike():
    lap = grid_laplacian(8, 8)
    b = np.zeros(lap.shape[0])
    b[0], b[-1] = 1.0, -1.0
    np.testing.assert_array_equal(
        approx_chol.factorize(lap).solve(b),
        approx_chol.factorize(lap, approx_chol.Config()).solve(b),
    )


def test_backend_reaches_the_factorization():
    exact = approx_chol.Backend.ExactBelow
    assert _grid_residual(approx_chol.Config(backend=exact(max_dim=16))) < 1e-12
    assert _grid_residual(approx_chol.Config(backend=exact(max_dim=15))) > 1e-3


def test_seed_and_split_reach_the_factorization():
    def solution(**config):
        return _grid_solution(
            approx_chol.Config(backend=approx_chol.Backend.Approximate(), **config)
        )

    base = solution(seed=0)
    np.testing.assert_array_equal(base, solution(seed=0))
    assert not np.array_equal(base, solution(seed=1))
    assert not np.array_equal(base, solution(seed=0, split=2))


def test_complement_leaves_a_csr_input_unchanged():
    # A CSR reduces to general blocks; only a bipartite block has a Schur complement to sample.
    def solution(complement):
        config = approx_chol.Config(
            backend=approx_chol.Backend.Approximate(), complement=complement
        )
        return _grid_solution(config)

    base = solution(approx_chol.SchurMode.Same())
    np.testing.assert_array_equal(base, solution(approx_chol.SchurMode.Exact()))


def _pair(off_diagonal: float) -> np.ndarray:
    return np.array([[1.0, off_diagonal], [off_diagonal, 1.0]])


def _warning(matrix) -> str:
    with pytest.warns(RuntimeWarning) as record:
        approx_chol.factorize(sp.csr_matrix(matrix))
    (warning,) = record
    return str(warning.message)


def test_a_slight_deficit_warns_only_that_it_was_clamped():
    message = _warning(_pair(-(1.0 + 1e-10)))
    assert message.startswith("1 block(s) fell short of diagonal dominance by up to")
    assert message.endswith("the deficit was clamped")


def test_each_cause_counts_its_own_blocks():
    # The first pair's scaling settles; no positive scaling repairs |a_01| = 2 in the second.
    message = _warning(block_diag(_pair(-(1.0 + 1e-10)), _pair(-2.0)))
    assert message.startswith(
        "2 block(s) fell short of diagonal dominance by up to 1.000e0"
    )
    assert message.endswith(
        "; 1 block(s) did not settle, so a singular one may have been grounded"
    )


def test_a_dominant_matrix_does_not_warn():
    lap = grid_laplacian(4, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        approx_chol.factorize(lap)
