import pickle
import re
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

import approx_chol
from tests._laplacians import MatrixLike

QUARTER = np.finfo(np.float64).max / 4
ORDER = "is out of order: the first must be 0, each at most the next"
DIAGONAL = "has a negative diagonal, a zero one beside an edge, or one past MAX/4"


def _duck(indptr, indices, data, shape=None, index_dtype=np.int64):
    n = len(indptr) - 1
    matrix = MatrixLike(
        np.array(indptr, dtype=index_dtype),
        np.array(indices, dtype=index_dtype),
        np.array(data, dtype=np.float64),
        shape or (n, n),
    )
    return lambda: approx_chol.factorize(matrix)


def _u32(indptr, indices, data):
    return _duck(indptr, indices, data, index_dtype=np.uint32)


def _scipy(dense):
    return lambda: approx_chol.factorize(sp.csr_matrix(np.array(dense)))


INPUT_ERRORS = [
    pytest.param(
        _u32([0, 2, 4], [0, 1, 0, 1], [2, -1, -1]),
        ("length_mismatch", "data", None),
        "data has length 3, expected 4",
        id="length_mismatch",
    ),
    pytest.param(
        _duck([0, 2, 4], [0, 1, 0, 1], [2, -1, -1, 2], shape=(2**32 + 1, 2**32 + 1)),
        ("length_mismatch", "indptr", None),
        "indptr has length 3, expected 4294967298",
        id="length_mismatch_oversized_dimension",
    ),
    pytest.param(
        _u32([1, 2, 4], [0, 1, 0, 1], [2, -1, -1, 2]),
        ("row_ptrs", "indptr", 0),
        f"indptr[0] {ORDER}",
        id="row_ptrs_first",
    ),
    pytest.param(
        _u32([0, 3, 2], [0, 1], [2, 2]),
        ("row_ptrs", "indptr", 1),
        f"indptr[1] {ORDER}",
        id="row_ptrs_decreasing",
    ),
    pytest.param(
        _u32([0, 2, 4], [0, 2, 0, 1], [2, -1, -1, 2]),
        ("index_out_of_range", "indices", 1),
        "indices[1] is out of range",
        id="index_out_of_range",
    ),
    pytest.param(
        _duck([0, 2, 4], [0, -1, 0, 1], [2, -1, -1, 2]),
        ("index_out_of_range", "indices", 1),
        "indices[1] is out of range",
        id="index_out_of_range_negative",
    ),
    pytest.param(
        _duck([0, 2, 4], [0, 2**32 + 1, 0, 1], [2, -1, -1, 2]),
        ("index_out_of_range", "indices", 1),
        "indices[1] is out of range",
        id="index_out_of_range_past_u32",
    ),
    pytest.param(
        _duck([0, 2, -4], [0, 1], [2, 2]),
        ("index_out_of_range", "indptr", 2),
        "indptr[2] is out of range",
        id="index_out_of_range_indptr",
    ),
    pytest.param(
        _u32([0, 1], [0], [np.nan]),
        ("out_of_range", "data", 0),
        "data[0] is outside the representable range",
        id="out_of_range",
    ),
    pytest.param(
        _duck([0, 2, 4], [0, 1, 0, 1], [1, -1e-310, -1e-310, 1]),
        ("out_of_range", "data", 1),
        "data[1] is outside the representable range",
        id="out_of_range_subnormal",
    ),
    pytest.param(
        _u32([0, 2, 4], [0, 1, 0, 1], [2, -1, -2, 2]),
        ("not_symmetric", None, 0),
        "row 0 has an entry with no equal mirror",
        id="not_symmetric",
    ),
    pytest.param(
        _scipy([[2.0, -1.0], [0.0, 2.0]]),
        ("not_symmetric", None, 0),
        "row 0 has an entry with no equal mirror",
        id="not_symmetric_scipy",
    ),
    pytest.param(
        _u32([0, 1], [0], [-1]),
        ("diagonal", None, 0),
        f"row 0 {DIAGONAL}",
        id="diagonal",
    ),
    pytest.param(
        _scipy(np.ones((2, 3))),
        ("not_square", None, None),
        "matrix is not square (2x3)",
        id="not_square",
    ),
]

BUILD_ERRORS = [
    # In range and GDD, but the centre's absolute row sum passes MAX/4.
    pytest.param(
        _duck(
            [0, 3, 5, 7],
            [0, 1, 2, 0, 1, 0, 2],
            np.array([1, -0.6, -0.6, -0.6, 1, -0.6, 1]) * QUARTER,
        ),
        ("out_of_range", 0),
        "a value at vertex 0 is outside the representable range",
        id="out_of_range",
    ),
    pytest.param(
        _u32(
            [0, 2, 5, 7],
            [0, 1, 0, 1, 2, 1, 2],
            [1, -1, -1, 2.0**-960, -(2.0**1000), -(2.0**1000), 2.0**-960],
        ),
        ("out_of_range", 1),
        "a value at vertex 1 is outside the representable range",
        id="out_of_range_relaxation",
    ),
]

# Each is raised by a Rust input form or a bound the Python API never reaches.
UNREACHABLE = {
    "InputError": {
        "vertex_count_overflow": "BipartiteFactorWidth2 only",
        "negative": "Laplacian weights and Sddm surpluses only",
        "self_loop": "Laplacian only",
        "repeated_row": "FactorWidth2 only",
        "outside_triangle": "TriangleCsrRef only",
        "row_gap": "the faer adapter only",
        "stored_by_columns": "the sprs adapter only",
    },
    "BuildError": {
        "too_large": "an f64 block of 2**52 vertices",
    },
}


@pytest.mark.parametrize(("call", "expected", "message"), INPUT_ERRORS)
def test_each_input_error_kind_raises_an_input_error_naming_it(call, expected, message):
    with pytest.raises(approx_chol.InputError) as caught:
        call()
    error = caught.value
    assert isinstance(error, ValueError)
    assert (error.kind, error.parameter, error.position) == expected
    assert str(error) == message


@pytest.mark.parametrize(("call", "expected", "message"), BUILD_ERRORS)
def test_each_build_error_kind_raises_a_build_error_naming_it(call, expected, message):
    with pytest.raises(approx_chol.BuildError) as caught:
        call()
    error = caught.value
    assert isinstance(error, ValueError)
    assert (error.kind, error.vertex) == expected
    assert str(error) == message


def _snake(name):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _rust_kinds(path):
    surface = Path(__file__).parent.parent / "crates/approx-chol/public-api.txt"
    text = surface.read_text(encoding="utf-8")
    pattern = rf"^pub approx_chol::{re.escape(path)}::(\w+)$"
    return {_snake(name) for name in re.findall(pattern, text, re.MULTILINE)}


def _kinds(params):
    return {param.values[1][0] for param in params}


@pytest.mark.parametrize(
    ("enum", "path", "params"),
    [
        ("InputError", "input::InputError", INPUT_ERRORS),
        ("BuildError", "BuildError", BUILD_ERRORS),
    ],
)
def test_every_rust_variant_is_raised_here_or_named_unreachable(enum, path, params):
    """A variant added in Rust fails here until it is raised above, which its arm in the binding takes, or listed unreachable."""
    raised, unreachable = _kinds(params), set(UNREACHABLE[enum])
    assert not raised & unreachable
    assert raised | unreachable == _rust_kinds(path)


def test_the_kind_and_message_survive_pickling():
    with pytest.raises(approx_chol.InputError) as caught:
        approx_chol.factorize(sp.csr_array(np.array([[2.0, -1.0], [0.0, 2.0]])))
    error = caught.value
    loaded = pickle.loads(pickle.dumps(error))
    assert type(loaded) is approx_chol.InputError
    assert (str(loaded), loaded.kind, loaded.parameter, loaded.position) == (
        "row 0 has an entry with no equal mirror",
        "not_symmetric",
        None,
        0,
    )

    build = approx_chol.BuildError("a value at vertex 3 is outside", "out_of_range", 3)
    loaded = pickle.loads(pickle.dumps(build))
    assert type(loaded) is approx_chol.BuildError
    assert (str(loaded), loaded.kind, loaded.vertex) == (str(build), "out_of_range", 3)
