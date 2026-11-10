import re
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

import approx_chol
from tests._laplacians import MatrixLike, grid_laplacian, light_row_grid


def _base_csr():
    row_ptrs = np.array([0, 2, 4], dtype=np.uint32)
    col_indices = np.array([0, 1, 0, 1], dtype=np.uint32)
    values = np.array([2.0, -1.0, -1.0, 2.0], dtype=np.float64)
    return row_ptrs, col_indices, values


def test_one_split_is_the_default():
    lap = grid_laplacian(8, 8)
    b = np.zeros(lap.shape[0])
    b[0], b[-1] = 1.0, -1.0

    def solve(split):
        config = approx_chol.Config(split=split, exact_max_dim=0)
        return approx_chol.factorize(lap, config=config).apply(b)

    np.testing.assert_array_equal(solve(1), solve(None))
    assert not np.array_equal(solve(2), solve(None))


@pytest.mark.parametrize("split", [0, -1, 2**32])
def test_split_outside_its_range_raises(split):
    with pytest.raises(
        ValueError, match=re.escape("split must be an integer in [1, 2**32 - 1]")
    ):
        approx_chol.Config(split=split)


@pytest.mark.parametrize("split", [np.int64(2), np.uint32(2)])
def test_a_numpy_integer_split_is_its_value(split):
    config = approx_chol.Config(split=split)
    assert config.split == 2
    assert config == approx_chol.Config(split=2)


def test_configs_compare_and_hash_by_value():
    assert approx_chol.Config(seed=1, split=2) == approx_chol.Config(seed=1, split=2)
    assert hash(approx_chol.Config(seed=1)) == hash(approx_chol.Config(seed=1))
    assert approx_chol.Config(seed=1) != approx_chol.Config(seed=2)
    assert len({approx_chol.Config(), approx_chol.Config(split=1)}) == 1


@pytest.mark.parametrize("split", [2.0, "2", np.float64(2.0), True])
def test_a_split_not_an_integer_raises_type_error(split):
    with pytest.raises(TypeError):
        approx_chol.Config(split=split)


def _stub_frustrated_names() -> list[str]:
    stub = Path(__file__).parent.parent / "python" / "approx_chol" / "__init__.pyi"
    literal = re.search(
        r"_Frustrated: TypeAlias = Literal\[(.*?)\]", stub.read_text(), re.DOTALL
    )
    return re.findall(r'"([^"]+)"', literal.group(1))


FRUSTRATED = _stub_frustrated_names()


@pytest.mark.parametrize("frustrated", FRUSTRATED)
def test_frustrated_round_trips(frustrated):
    assert approx_chol.Config(frustrated=frustrated).frustrated == frustrated


def test_an_unknown_frustrated_route_raises_naming_the_stubs_routes():
    with pytest.raises(ValueError, match="frustrated must be one of ") as error:
        approx_chol.Config(frustrated="sheet")
    named = str(error.value).removesuffix(", not 'sheet'")
    assert re.findall(r"'([^']+)'", named) == FRUSTRATED


def test_frustrated_is_keyword_only():
    with pytest.raises(TypeError):
        approx_chol.Config(0, 1, 24, "cover")


@pytest.mark.parametrize(
    ("light", "like", "unlike"),
    [
        (1e-40, "cover_surplus_edges", "cover_ground_vertex"),
        (1.0, "cover_ground_vertex", "cover_surplus_edges"),
    ],
)
def test_each_frustrated_name_factors_as_its_route(light, like, unlike):
    """Inside the twist band ``"cover"`` carries the surplus on edges, past it to the ground vertex, so two names swapped in the table fail one case."""
    matrix = light_row_grid(10, light)
    b = np.sin(np.arange(matrix.shape[0], dtype=np.float64))

    def solve(**route):
        config = approx_chol.Config(exact_max_dim=0, **route)
        return approx_chol.factorize(matrix, config=config).apply(b)

    cover = solve(frustrated="cover")
    np.testing.assert_array_equal(cover, solve())
    np.testing.assert_array_equal(cover, solve(frustrated=like))
    assert not np.array_equal(cover, solve(frustrated=unlike))


def test_signed_factors_a_frustrated_block_as_it_is():
    """``"signed"`` eliminates the block on one sheet: its solve is neither the cover's nor floating."""
    matrix = light_row_grid(10, 1.0)
    b = np.sin(np.arange(matrix.shape[0], dtype=np.float64))
    config = approx_chol.Config(exact_max_dim=0, frustrated="signed")
    signed = approx_chol.factorize(matrix, config=config).apply(b)
    cover = approx_chol.factorize(
        matrix, config=approx_chol.Config(exact_max_dim=0)
    ).apply(b)
    assert np.all(np.isfinite(signed))
    assert not np.array_equal(signed, cover)


def test_signed_solves_a_triangle_grounded_twice_exactly_where_every_cover_samples():
    """Vertex 2 pivots first, so each signed star holds at most two entries and is eliminated exactly; every cover holds a three-entry star, which it samples."""
    m = np.array([[2.5, -1.0, 1.0], [-1.0, 2.75, -1.5], [1.0, -1.5, 2.5]])
    b = np.array([1.0, -2.0, 0.5])
    exact = np.linalg.solve(m, b)

    def error(**route):
        config = approx_chol.Config(exact_max_dim=0, **route)
        x = approx_chol.factorize(sp.csr_array(m), config=config).apply(b)
        return np.linalg.norm(x - exact) / np.linalg.norm(exact)

    assert error(frustrated="signed") < 1e-12
    for route in ["cover", "cover_ground_vertex", "cover_surplus_edges"]:
        assert error(frustrated=route) > 1e-6, route
    assert error() > 1e-6


@pytest.mark.parametrize(
    ("indptr_dtype", "indices_dtype"),
    [
        (np.int32, np.int32),
        (np.int64, np.int64),
        (np.uint32, np.uint32),
        (np.uint64, np.uint64),
        (np.int16, np.int16),
        (np.int32, np.int64),
        (np.int64, np.uint64),
    ],
)
def test_every_integer_index_dtype_factors_alike(indptr_dtype, indices_dtype):
    lap = grid_laplacian(8, 8)
    b = np.zeros(lap.shape[0])
    b[0], b[-1] = 1.0, -1.0
    expected = approx_chol.factorize(lap).apply(b)

    typed = MatrixLike(
        lap.indptr.astype(indptr_dtype),
        lap.indices.astype(indices_dtype),
        lap.data,
        lap.shape,
    )
    np.testing.assert_array_equal(approx_chol.factorize(typed).apply(b), expected)


def test_each_column_rejects_the_dtype_kinds_it_cannot_carry():
    # A float index would truncate silently and a complex value drop its imaginary part.
    row_ptrs, col_indices, values = _base_csr()

    float_index = MatrixLike(row_ptrs.astype(np.float64), col_indices, values, (2, 2))
    with pytest.raises(ValueError, match="indptr must have an integer dtype"):
        approx_chol.factorize(float_index)

    complex_value = MatrixLike(
        row_ptrs, col_indices, values.astype(np.complex128), (2, 2)
    )
    with pytest.raises(
        ValueError, match="data must have an integer or floating-point dtype"
    ):
        approx_chol.factorize(complex_value)

    two_dimensional = MatrixLike(row_ptrs, col_indices.reshape(2, 2), values, (2, 2))
    with pytest.raises(ValueError, match="indices must be a 1-D array"):
        approx_chol.factorize(two_dimensional)


def test_an_all_zero_row_factors_and_solves_to_zero():
    dead = sp.csr_matrix(np.array([[0.0, 0, 0], [0, 2, -1], [0, -1, 2]]))
    x = approx_chol.factorize(dead).apply(np.ones(3))
    np.testing.assert_array_equal(x, [0.0, 1.0, 1.0])


@pytest.mark.parametrize("cls", [approx_chol.Component, approx_chol.Clamp])
def test_component_and_clamp_cannot_be_constructed_directly(cls):
    with pytest.raises(TypeError, match="cannot create"):
        cls()
