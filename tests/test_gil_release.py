import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import numpy as np
import pytest
import scipy.sparse as sp

import approx_chol
from tests._laplacians import grid_laplacian

pytestmark = pytest.mark.skipif(
    not getattr(sys, "_is_gil_enabled", lambda: True)(), reason="no GIL to release"
)

Share = Callable[[Callable[[], object]], float]


@contextmanager
def _spinner() -> Iterator[Share]:
    count = [0]
    stop = threading.Event()

    def spin() -> None:
        while not stop.is_set():
            count[0] += 1

    def once(call: Callable[[], object], idle_rate: float) -> float:
        time.sleep(0.001)
        start, t0 = count[0], time.perf_counter()
        call()
        return (count[0] - start) / ((time.perf_counter() - t0) * idle_rate)

    # Held, the GIL starves the spinner; released, the spinner runs until the caller reacquires.
    interval = sys.getswitchinterval()
    sys.setswitchinterval(0.2)
    thread = threading.Thread(target=spin)
    thread.start()
    try:
        start, t0 = count[0], time.perf_counter()
        time.sleep(0.05)
        idle_rate = (count[0] - start) / (time.perf_counter() - t0)
        # Under load the spinner may wake after a short call ends: best of three.
        yield lambda call: max(once(call, idle_rate) for _ in range(3))
    finally:
        stop.set()
        thread.join()
        sys.setswitchinterval(interval)


def _calls(lap: sp.csr_matrix) -> dict[str, Callable[[], object]]:
    factor = approx_chol.factorize(lap)
    b = np.zeros(factor.n)
    b[0], b[-1] = 1.0, -1.0
    out = np.empty_like(b)
    workspace = factor.workspace()
    return {
        "factorize": lambda: approx_chol.factorize(lap),
        "solve": lambda: factor.solve(b),
        "solve_into": lambda: factor.solve_into(b, out),
        "solve_into with a workspace": lambda: factor.solve_into(b, out, workspace),
    }


def test_a_matrix_above_32768_entries_lets_other_threads_progress():
    lap = grid_laplacian(100, 100)
    assert lap.nnz > 1 << 15
    calls = _calls(lap)
    with _spinner() as share:
        shares = {name: share(call) for name, call in calls.items()}
    assert all(s > 0.2 for s in shares.values()), shares


def test_a_matrix_of_at_most_32768_entries_keeps_the_gil():
    lap = grid_laplacian(81, 81)
    assert lap.nnz <= 1 << 15
    calls = _calls(lap)
    with _spinner() as share:
        shares = {name: share(call) for name, call in calls.items()}
    assert all(s < 0.1 for s in shares.values()), shares
