import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import numpy as np
import pytest
import scipy.sparse as sp

import approx_chol
from tests._laplacians import grid_laplacian


@contextmanager
def writing(write: Callable[[], None]) -> Iterator[None]:
    """Calls ``write`` in a loop on another thread for the duration of the block."""
    stop = threading.Event()

    def loop() -> None:
        while not stop.is_set():
            write()

    thread = threading.Thread(target=loop)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


def test_factorize_reads_a_snapshot_of_arrays_another_thread_writes():
    lap = grid_laplacian(200, 200)
    assert lap.indices.dtype == np.int32
    b = np.zeros(lap.shape[0])
    b[0], b[-1] = 1.0, -1.0
    expected = approx_chol.factorize(lap).solve(b)
    j, column = lap.nnz // 2, lap.indices[lap.nnz // 2]

    # One aligned store at a time, so the copy sees the column either valid or out of range.
    def toggle() -> None:
        lap.indices[j] = lap.shape[0]
        lap.indices[j] = column

    compared = 0
    for _ in range(30):
        with writing(toggle):
            try:
                factor = approx_chol.factorize(lap)
            except ValueError:
                continue
        np.testing.assert_array_equal(factor.solve(b), expected)
        compared += 1
    assert compared > 0


@pytest.mark.parametrize("with_workspace", [False, True])
def test_solve_into_is_not_corrupted_by_a_concurrent_writer(with_workspace: bool):
    grid = grid_laplacian(150, 150)
    # Two blocks, so the solve writes them one after the other.
    factor = approx_chol.factorize(sp.block_diag([grid, grid], format="csr"))
    b = np.random.default_rng(0).standard_normal(factor.n)
    expected = factor.solve(b)
    assert np.all(expected != 1.0)
    out = np.empty_like(b)
    workspace = factor.workspace() if with_workspace else None
    start = time.perf_counter()
    factor.solve_into(b, out, workspace)
    elapsed = time.perf_counter() - start

    for delay in np.linspace(0.0, elapsed, 20):
        writer = threading.Timer(delay, out.fill, (1.0,))
        writer.start()
        factor.solve_into(b, out, workspace)
        writer.join()
        # The write may land after the result; a value that is neither is a corrupted solve.
        assert np.all((out == 1.0) | (out == expected))
