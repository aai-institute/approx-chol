import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import numpy as np
import pytest
import scipy.sparse as sp

import approx_chol


def _grid(k: int) -> sp.csr_matrix:
    path = sp.diags([-1.0, -1.0], [-1, 1], shape=(k, k))
    path = sp.diags(-np.asarray(path.sum(axis=1)).ravel()) + path
    eye = sp.identity(k)
    return (sp.kron(path, eye) + sp.kron(eye, path)).tocsr()


@contextmanager
def _spinning_counter() -> Iterator[list[int]]:
    count = [0]
    stop = threading.Event()

    def spin() -> None:
        while not stop.is_set():
            count[0] += 1

    thread = threading.Thread(target=spin)
    thread.start()
    time.sleep(0.1)
    try:
        yield count
    finally:
        stop.set()
        thread.join()


def _progress_ratio(count: list[int], call: Callable[[], object]) -> float:
    start, t0 = count[0], time.perf_counter()
    call()
    elapsed, busy = time.perf_counter() - t0, count[0] - start
    start = count[0]
    time.sleep(elapsed)
    return busy / (count[0] - start)


@pytest.mark.skipif(
    not getattr(sys, "_is_gil_enabled", lambda: True)(), reason="no GIL to release"
)
def test_other_threads_progress_during_factorize_and_solve():
    lap = _grid(1000)
    b = np.zeros(lap.shape[0])
    b[0], b[-1] = 1.0, -1.0
    out = np.empty_like(b)
    with _spinning_counter() as count:
        factor_box: list[approx_chol.Factor] = []
        factorize = _progress_ratio(
            count, lambda: factor_box.append(approx_chol.factorize(lap))
        )
        factor = factor_box[0]
        solve = _progress_ratio(count, lambda: [factor.solve(b) for _ in range(20)])
        solve_into = _progress_ratio(
            count, lambda: [factor.solve_into(b, out) for _ in range(20)]
        )
    assert factorize > 0.1, (
        f"counter thread ran at {factorize:.3f} of its idle rate during factorize"
    )
    assert solve > 0.5, (
        f"counter thread ran at {solve:.3f} of its idle rate during solve"
    )
    assert solve_into > 0.5, (
        f"counter thread ran at {solve_into:.3f} of its idle rate during solve_into"
    )
