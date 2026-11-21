import sys
import threading
import time
from collections.abc import Callable

import numpy as np
import pytest
import scipy.sparse as sp

import approx_chol
from tests._laplacians import grid_laplacian

pytestmark = pytest.mark.skipif(
    not getattr(sys, "_is_gil_enabled", lambda: True)(), reason="no GIL to release"
)


def _steps(lap: sp.csr_matrix) -> dict[str, int]:
    """A spinning thread's steps while each call repeats: none while the GIL is held, how many else is the OS's."""
    factor = approx_chol.factorize(lap)
    b = np.zeros(factor.n)
    b[0], b[-1] = 1.0, -1.0
    out = np.empty_like(b)
    calls: dict[str, Callable[[], object]] = {
        "factorize": lambda: approx_chol.factorize(lap),
        "apply": lambda: factor.apply(b),
        "apply into out": lambda: factor.apply(b, out=out),
    }
    count = [0]
    stop = threading.Event()

    def spin() -> None:
        while not stop.is_set():
            count[0] += 1

    switch = 0.2

    def steps(call: Callable[[], object]) -> int:
        time.sleep(0.001)
        start, t0 = count[0], time.perf_counter()
        # A loaded OS may wake the spinner after a 0.2 ms solve ends; too brief to force a switch.
        while time.perf_counter() - t0 < switch / 10:
            call()
        return count[0] - start

    interval = sys.getswitchinterval()
    sys.setswitchinterval(switch)
    thread = threading.Thread(target=spin)
    thread.start()
    try:
        return {name: steps(call) for name, call in calls.items()}
    finally:
        stop.set()
        thread.join()
        sys.setswitchinterval(interval)


def test_a_matrix_above_32768_entries_lets_other_threads_progress():
    lap = grid_laplacian(100, 100)
    assert lap.nnz > 1 << 15
    steps = _steps(lap)
    assert all(steps.values()), steps


def test_a_matrix_of_at_most_32768_entries_keeps_the_gil():
    lap = grid_laplacian(81, 81)
    assert lap.nnz <= 1 << 15
    steps = _steps(lap)
    assert not any(steps.values()), steps
