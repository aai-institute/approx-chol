import copy
import pickle
import warnings

import numpy as np
import pytest
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.linalg import block_diag

import approx_chol
from tests._laplacians import bits, grid_laplacian

SCIPY_1_18 = tuple(int(part) for part in scipy.__version__.split(".")[:2]) >= (1, 18)
stacks = pytest.mark.skipif(not SCIPY_1_18, reason="scipy passes stacks from 1.18")

LAPLACIAN = grid_laplacian(12, 12)
N = LAPLACIAN.shape[0]


def _factor(**kwargs) -> approx_chol.ApproxInverse:
    return approx_chol.factorize(LAPLACIAN, config=approx_chol.Config(seed=5), **kwargs)


def _sides(k: int) -> np.ndarray:
    b = np.random.default_rng(429).standard_normal((N, k))
    return b - b.mean(axis=0)


def _columns(factor, block: np.ndarray) -> np.ndarray:
    return np.column_stack([factor.apply(np.ascontiguousarray(c)) for c in block.T])


def test_is_a_float64_linear_operator_of_shape_n_by_n():
    factor = _factor()
    assert isinstance(factor, sla.LinearOperator)
    assert sla.aslinearoperator(factor) is factor
    assert (factor.shape, factor.dtype) == ((N, N), np.float64)


def test_is_its_own_adjoint_and_transpose():
    factor = _factor()
    assert factor.H is factor
    assert factor.T is factor


@pytest.mark.parametrize("threads", [None, 1, 3])
def test_threads_is_the_builds_budget(threads):
    assert _factor(threads=threads).threads == threads


@pytest.mark.parametrize("threads", [0, -1, 2**64])
def test_a_budget_out_of_range_raises_value_error(threads):
    with pytest.raises(
        ValueError, match=r"^threads must be an integer in \[1, 2\*\*64 - 1\]$"
    ):
        _factor(threads=threads)
    with pytest.raises(ValueError, match="^threads must be an integer"):
        _factor().with_threads(threads)


def test_a_budget_not_an_integer_raises_type_error():
    with pytest.raises(TypeError):
        _factor(threads=1.5)


def test_with_threads_holds_another_budget_and_leaves_the_original():
    factor = _factor(threads=2)
    other = factor.with_threads(1)
    assert type(other) is approx_chol.ApproxInverse
    assert (other.threads, factor.threads) == (1, 2)
    assert factor.with_threads(None).threads is None


def test_a_bool_budget_raises_type_error():
    for threads in [True, False]:
        with pytest.raises(TypeError, match="^threads must be an integer, not a bool$"):
            _factor(threads=threads)
        with pytest.raises(TypeError, match="^threads must be an integer, not a bool$"):
            _factor().with_threads(threads)


def test_every_budget_builds_and_applies_to_the_samebits():
    b = _sides(9)
    every_core = _factor(threads=None)
    expected = bits(every_core.apply(b))
    for threads in [1, 2, 7]:
        built = _factor(threads=threads)
        np.testing.assert_array_equal(bits(built.apply(b)), expected)
        reapplied = every_core.with_threads(threads)
        np.testing.assert_array_equal(bits(reapplied.apply(b)), expected)


def test_apply_solves_a_vector_into_a_new_array():
    factor = _factor()
    b = _sides(1)[:, 0]
    x = factor.apply(b)
    assert x.shape == b.shape and x.dtype == np.float64
    assert not np.shares_memory(x, b)
    assert np.linalg.norm(LAPLACIAN @ x - b) < np.linalg.norm(b)


@pytest.mark.parametrize("order", ["C", "F"])
def test_apply_solves_each_column_of_a_block_as_alone(order):
    factor = _factor()
    b = np.asarray(_sides(6), order=order)
    x = factor.apply(b)
    assert x.shape == b.shape
    np.testing.assert_array_equal(bits(x), bits(_columns(factor, b)))


def test_apply_converts_a_real_non_float64_input():
    factor = _factor()
    b = np.arange(N) - N // 2
    expected = factor.apply(b.astype(np.float64))
    for converted in [b, b.astype(np.int32), b.astype(np.float32), b.tolist()]:
        np.testing.assert_array_equal(factor.apply(converted), expected)


@pytest.mark.parametrize("dtype", [np.complex128, object, "U3"])
def test_apply_raises_type_error_on_any_other_dtype(dtype):
    with pytest.raises(TypeError, match="^b must have a real dtype"):
        _factor().apply(np.zeros(N, dtype=dtype))


@pytest.mark.parametrize("shape", [(N + 1,), (N - 1, 2), (N, 2, 1), ()])
def test_apply_raises_value_error_on_a_wrong_shape(shape):
    with pytest.raises(ValueError, match=r"^b has shape .*, expected"):
        _factor().apply(np.zeros(shape))


@pytest.mark.parametrize("order", ["C", "F"])
def test_apply_writes_out_in_any_order_and_returns_it(order):
    factor = _factor()
    b = _sides(3)
    out = np.empty(b.shape, order=order)
    assert factor.apply(b, out=out) is out
    np.testing.assert_array_equal(out, factor.apply(b))


def test_apply_writes_a_strided_out():
    factor = _factor()
    b = _sides(1)[:, 0]
    base = np.zeros(2 * N)
    factor.apply(b, out=base[::2])
    np.testing.assert_array_equal(base[::2], factor.apply(b))
    assert not base[1::2].any()


@pytest.mark.parametrize("k", [None, 4])
def test_apply_writes_out_as_b_itself(k):
    factor = _factor()
    b = _sides(k or 1)[:, 0] if k is None else _sides(k)
    expected = factor.apply(b)
    assert factor.apply(b, out=b) is b
    np.testing.assert_array_equal(b, expected)


def test_apply_writes_out_overlapping_b():
    factor = _factor()
    base = np.concatenate([_sides(1)[:, 0], [0.0]])
    expected = factor.apply(base[:-1])
    factor.apply(base[:-1], out=base[1:])
    np.testing.assert_array_equal(base[1:], expected)


@pytest.mark.parametrize("shape", [(N + 1,), (N, 1)])
def test_apply_raises_value_error_on_out_of_another_shape(shape):
    with pytest.raises(ValueError, match=rf"^out has shape .*, expected \({N},\)$"):
        _factor().apply(np.zeros(N), out=np.zeros(shape))


def test_apply_raises_value_error_on_out_whose_entries_overlap():
    out = np.lib.stride_tricks.as_strided(np.zeros(N + 1), (N,), (0,))
    with pytest.raises(ValueError, match="^out's entries overlap or interleave$"):
        _factor().apply(np.zeros(N), out=out)


@pytest.mark.parametrize(
    "strides, overlap",
    [
        pytest.param((8, 8 * 6), False, id="nested"),
        pytest.param((8 * 3, 8 * 16), False, id="just-past-the-span"),
        pytest.param((8 * 3, 8 * 15), True, id="at-the-span"),
        pytest.param((8 * 3, 8 * 5), True, id="interleaved"),
        pytest.param((8 * 2, 0), True, id="repeated-column"),
    ],
)
def test_out_of_two_axes_raises_unless_each_steps_past_the_others_span(
    strides, overlap
):
    factor = approx_chol.factorize(grid_laplacian(2, 3))
    b = np.arange(12.0).reshape(6, 2) - 2.5
    out = np.lib.stride_tricks.as_strided(np.zeros(64), (6, 2), strides)
    if overlap:
        with pytest.raises(ValueError, match="^out's entries overlap or interleave$"):
            factor.apply(b, out=out)
    else:
        factor.apply(b, out=out)
        np.testing.assert_array_equal(out, factor.apply(b))


def test_apply_raises_on_out_not_float64_or_read_only():
    factor = _factor()
    with pytest.raises(TypeError, match="^out must be a float64 array$"):
        factor.apply(np.zeros(N), out=np.zeros(N, dtype=np.float32))
    with pytest.raises(TypeError, match="^out must be a float64 array$"):
        factor.apply(np.zeros(N), out=[0.0] * N)
    read_only = np.zeros(N)
    read_only.flags.writeable = False
    with pytest.raises(ValueError, match="^out must be writeable$"):
        factor.apply(np.zeros(N), out=read_only)


def test_matvec_and_rmatvec_apply_a_vector():
    factor = _factor()
    b = _sides(1)[:, 0]
    np.testing.assert_array_equal(factor.matvec(b), factor.apply(b))
    np.testing.assert_array_equal(factor.rmatvec(b), factor.apply(b))
    np.testing.assert_array_equal(factor @ b, factor.apply(b))


def test_matvec_takes_an_n_by_1_column():
    factor = _factor()
    b = _sides(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        np.testing.assert_array_equal(factor.matvec(b), factor.apply(b))


@stacks
def test_matvec_and_rmatvec_apply_a_stack_of_vectors():
    factor = _factor()
    stack = _sides(6).T.reshape(2, 3, N)
    expected = np.stack([factor.apply(v) for v in stack.reshape(6, N)]).reshape(
        stack.shape
    )
    np.testing.assert_array_equal(factor.matvec(stack), expected)
    np.testing.assert_array_equal(factor.rmatvec(stack), expected)


def test_matmat_and_rmatmat_apply_a_block():
    factor = _factor()
    b = _sides(3)
    np.testing.assert_array_equal(factor.matmat(b), factor.apply(b))
    np.testing.assert_array_equal(factor.rmatmat(b), factor.apply(b))
    np.testing.assert_array_equal(factor @ b, factor.apply(b))


@stacks
def test_matmat_and_rmatmat_apply_a_stack_of_blocks():
    factor = _factor()
    stack = np.stack([_sides(3), 2 * _sides(3)])
    expected = np.stack([factor.apply(block) for block in stack])
    np.testing.assert_array_equal(factor.matmat(stack), expected)
    np.testing.assert_array_equal(factor.rmatmat(stack), expected)


def test_components_list_each_components_vertices():
    two = sp.block_diag([grid_laplacian(2, 2), grid_laplacian(3, 1)], format="csr")
    components = approx_chol.factorize(two).components
    assert type(components) is tuple
    assert all(type(c) is approx_chol.Component for c in components)
    vertices = [c.vertices for c in components]
    assert all(v.dtype == np.intp and not v.flags.writeable for v in vertices)
    assert [sorted(v.tolist()) for v in vertices] == [[0, 1, 2, 3], [4, 5, 6]]
    assert [c.clamp for c in components] == [None, None]


@pytest.mark.parametrize(
    "off_diagonal, scaling",
    [(-(1.0 + 1e-10), "certified"), (-2.0, "refuted")],
)
def test_a_clamped_component_reads_its_clamp(off_diagonal, scaling):
    pair = sp.csr_matrix(np.array([[1.0, off_diagonal], [off_diagonal, 1.0]]))
    dominant = sp.csr_matrix(np.array([[2.0, -1.0], [-1.0, 2.0]]))
    with pytest.warns(RuntimeWarning):
        factor = approx_chol.factorize(
            sp.csr_matrix(block_diag(pair.toarray(), dominant.toarray()))
        )
    clamped, unclamped = factor.components
    assert clamped.vertices.tolist() == [0, 1]
    assert type(clamped.clamp) is approx_chol.Clamp
    assert clamped.clamp.scaling == scaling
    assert clamped.clamp.violation > 0
    assert unclamped.clamp is None


@pytest.mark.parametrize("threads", [None, 1])
def test_a_loaded_or_copied_operator_holds_every_core(threads):
    factor = _factor(threads=threads)
    for restored in [pickle.loads(pickle.dumps(factor)), copy.deepcopy(factor)]:
        assert restored.threads is None


@pytest.mark.parametrize("solver", [sla.cg, sla.minres])
def test_cg_and_minres_take_it_as_a_preconditioner(solver):
    b = _sides(1)[:, 0]
    x, info = solver(LAPLACIAN, b, M=_factor(), rtol=1e-10, maxiter=200)
    assert info == 0
    assert np.linalg.norm(LAPLACIAN @ x - b) < 1e-8 * np.linalg.norm(b)


def test_lsmr_right_preconditions_through_the_composed_operator():
    factor = _factor()
    b = _sides(1)[:, 0]
    y = sla.lsmr(sla.aslinearoperator(LAPLACIAN) @ factor, b, atol=1e-12, btol=1e-12)[0]
    x = factor @ y
    assert np.linalg.norm(LAPLACIAN @ x - b) < 1e-8 * np.linalg.norm(b)


def _strided(b: np.ndarray, stride: int, offset: int = 0) -> np.ndarray:
    """``b`` copied into a float64 view of ``stride`` bytes, ``offset`` bytes into its buffer."""
    view = np.ndarray(
        b.shape, np.float64, bytearray(stride * len(b) + 8), offset, (stride,)
    )
    view[:] = b
    return view


@pytest.mark.parametrize(
    "stride, offset",
    [pytest.param(12, 0, id="stride-12"), pytest.param(8, 1, id="unaligned")],
)
def test_apply_reads_b_of_any_stride_or_alignment(stride, offset):
    factor = _factor()
    b = _sides(1)[:, 0]
    np.testing.assert_array_equal(
        bits(factor.apply(_strided(b, stride, offset))), bits(factor.apply(b))
    )


@pytest.mark.parametrize(
    "stride, offset",
    [pytest.param(12, 0, id="stride-12"), pytest.param(8, 1, id="unaligned")],
)
def test_apply_raises_value_error_on_out_unaligned_or_splitting_entries(stride, offset):
    out = _strided(np.zeros(N), stride, offset)
    with pytest.raises(
        ValueError, match="^out must be aligned, its strides whole float64s$"
    ):
        _factor().apply(np.zeros(N), out=out)


def test_a_structured_arrays_field_is_read_as_b_but_refused_as_out():
    factor = _factor()
    b = _sides(1)[:, 0]
    record = np.zeros(N, dtype=[("tag", np.int32), ("value", np.float64)])
    record["value"] = b
    np.testing.assert_array_equal(
        bits(factor.apply(record["value"])), bits(factor.apply(b))
    )
    with pytest.raises(ValueError, match="^out must be aligned"):
        factor.apply(b, out=record["value"])


@stacks
def test_an_empty_operator_applies_empty_stacks():
    factor = approx_chol.factorize(sp.csr_matrix((0, 0)))
    assert factor.matvec(np.zeros((2, 0))).shape == (2, 0)
    assert factor.matmat(np.zeros((2, 0, 3))).shape == (2, 0, 3)


def test_components_are_built_once_and_shared_across_budgets():
    factor = _factor()
    assert factor.components is factor.components
    assert factor.with_threads(1).components is factor.components
