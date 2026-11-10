import copy
import pickle

import numpy as np
import pytest
import scipy.sparse as sp

import approx_chol
from tests._laplacians import bits, frustrated_grid, grid_laplacian

MATRICES = {
    "laplacian": grid_laplacian(6, 7),
    "sddm": (grid_laplacian(6, 7) + sp.identity(42)).tocsr(),
    "frustrated": frustrated_grid(6, 7),
}

# 0x4143_000F is at least 2**28, so its postcard varint takes 5 bytes.
SCALAR = 5


def _right_hand_sides(n: int) -> list[np.ndarray]:
    rng = np.random.default_rng(431)
    sides = [rng.standard_normal(n) for _ in range(3)]
    for b in sides:
        b -= b.mean()
    return sides


def _assert_solves_alike(original, restored) -> None:
    assert restored.shape == original.shape
    assert restored.n_steps == original.n_steps
    for b in _right_hand_sides(original.n):
        expected = bits(original.apply(b))
        np.testing.assert_array_equal(bits(restored.apply(b)), expected)


def _form() -> bytes:
    _, (form,) = approx_chol.factorize(grid_laplacian(4, 4)).__reduce__()
    return form


@pytest.mark.parametrize("name", MATRICES)
@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_pickle_round_trip_solves_bit_for_bit(name, protocol):
    original = approx_chol.factorize(MATRICES[name], config=approx_chol.Config(seed=3))
    restored = pickle.loads(pickle.dumps(original, protocol=protocol))
    assert type(restored) is approx_chol.ApproxInverse
    _assert_solves_alike(original, restored)


def test_deepcopy_solves_bit_for_bit():
    original = approx_chol.factorize(MATRICES["sddm"])
    _assert_solves_alike(original, copy.deepcopy(original))


def test_reduce_is_the_saved_form_alone():
    original = approx_chol.factorize(MATRICES["laplacian"])
    load, (form,) = original.__reduce__()
    assert load == approx_chol.ApproxInverse._load
    assert type(form) is bytes
    _assert_solves_alike(original, load(form))


def test_another_version_raises_naming_the_version():
    form = _form()
    tampered = bytes([form[0] ^ 0x01]) + form[1:]
    with pytest.raises(
        ValueError, match=r"^format version 0x\w+, expected format version 0x\w+$"
    ):
        approx_chol.ApproxInverse._load(tampered)


def test_another_scalar_raises_naming_the_scalar():
    form = _form()
    assert form[SCALAR] == 1
    tampered = form[:SCALAR] + bytes([0]) + form[SCALAR + 1 :]
    with pytest.raises(ValueError, match=r"^scalar f32, expected scalar f64$"):
        approx_chol.ApproxInverse._load(tampered)


@pytest.mark.parametrize(
    "tamper",
    [
        pytest.param(lambda form: form[: len(form) // 2], id="truncated"),
        pytest.param(lambda form: form[:2], id="truncated-header"),
        pytest.param(lambda form: form + b"\x00", id="trailing-bytes"),
    ],
)
def test_an_undecodable_form_raises_as_invalid(tamper):
    with pytest.raises(ValueError, match="^the saved ApproxInverse is invalid: "):
        approx_chol.ApproxInverse._load(tamper(_form()))
