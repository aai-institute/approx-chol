from importlib import import_module

import pytest


def load_extension_module():
    try:
        return import_module("approx_chol._approx_chol")
    except ModuleNotFoundError as exc:
        if exc.name in {"approx_chol", "approx_chol._approx_chol"}:
            pytest.skip(
                "approx-chol Python extension not available; run `pixi run develop` first"
            )
        raise
