import warnings

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.linalg import block_diag

import approx_chol
from tests._laplacians import grid_laplacian, grid_sddm


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
        "ScalingPolicy",
        "Workspace",
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

    config = approx_chol.Config(
        seed=7, split=3, backend=approx_chol.Backend.Approximate()
    )
    assert (config.seed, config.split) == (7, 3)
    assert config.backend == approx_chol.Backend.Approximate()


def test_scaling_policy_defaults_getters_and_range():
    default = approx_chol.ScalingPolicy()
    assert (default.tolerance, default.max_iterations) == (1e-9, 2048)
    policy = approx_chol.ScalingPolicy(tolerance=0.25, max_iterations=7)
    assert (policy.tolerance, policy.max_iterations) == (0.25, 7)
    assert policy == approx_chol.ScalingPolicy(0.25, 7) != default
    with pytest.raises(ValueError, match=r"tolerance must be in \[0, 1\)"):
        approx_chol.ScalingPolicy(tolerance=1.0)


def _dld_path(n: int) -> sp.csr_matrix:
    # D L D of a path, D within 1/2 of I: short of dominance until scaled.
    d = 1.0 + ((7919 * np.arange(n)) % 1000) / 1000.0 - 0.5
    lap = sp.diags(
        [-np.ones(n - 1), np.r_[1.0, 2.0 * np.ones(n - 2), 1.0], -np.ones(n - 1)],
        [-1, 0, 1],
    )
    return sp.csr_matrix(sp.diags(d) @ lap @ sp.diags(d))


def test_factorize_and_factorize_raw_scale_under_the_given_policy():
    path = _dld_path(10)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        approx_chol.factorize(path)
    no_budget = approx_chol.ScalingPolicy(max_iterations=0)
    with pytest.warns(RuntimeWarning, match="1 block\\(s\\) did not settle"):
        approx_chol.factorize(path, scaling=no_budget)
    with pytest.warns(RuntimeWarning, match="1 block\\(s\\) did not settle"):
        approx_chol.factorize_raw(
            path.indptr, path.indices, path.data, path.shape[0], scaling=no_budget
        )


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


def test_solve_into_with_a_reused_workspace_is_bit_identical():
    factor = approx_chol.factorize(
        grid_sddm(16, 16), approx_chol.Config(backend=approx_chol.Backend.Approximate())
    )
    workspace = factor.workspace()
    rng = np.random.default_rng(0)
    for _ in range(3):
        b = rng.standard_normal(factor.n)
        fresh, reused = np.empty_like(b), np.empty_like(b)
        factor.solve_into(b, fresh)
        factor.solve_into(b, reused, workspace=workspace)
        np.testing.assert_array_equal(reused, fresh)
        np.testing.assert_array_equal(reused, factor.solve(b))
