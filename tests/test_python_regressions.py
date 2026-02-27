import importlib.machinery
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_extension_module():
    repo_root = Path(__file__).resolve().parents[1]
    candidates = (
        repo_root / "target/debug/lib_approx_chol.dylib",
        repo_root / "target/debug/lib_approx_chol.so",
        repo_root / "target/debug/_approx_chol.pyd",
    )

    for path in candidates:
        if path.exists():
            loader = importlib.machinery.ExtensionFileLoader("_approx_chol", str(path))
            spec = importlib.util.spec_from_loader("_approx_chol", loader)
            assert spec is not None
            module = importlib.util.module_from_spec(spec)
            loader.exec_module(module)
            return module

    pytest.skip(
        "approx-chol Python extension not found; build with `cargo build -p approx-chol-py` first"
    )


def test_factorize_raw_non_contiguous_arrays_raise_value_error():
    ext = _load_extension_module()

    row_ptrs = np.array([0, 99, 2, 99, 4], dtype=np.uint32)[::2]
    col_indices = np.array([0, 99, 1, 99, 0, 99, 1], dtype=np.uint32)[::2]
    values = np.array([2.0, 0.0, -1.0, 0.0, -1.0, 0.0, 2.0], dtype=np.float64)[::2]

    assert not row_ptrs.flags["C_CONTIGUOUS"]
    assert not col_indices.flags["C_CONTIGUOUS"]
    assert not values.flags["C_CONTIGUOUS"]

    with pytest.raises(ValueError):
        ext.factorize_raw(row_ptrs, col_indices, values, 2)
