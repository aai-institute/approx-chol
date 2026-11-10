"""Approximate Cholesky factorization for SDDM/Laplacian systems."""

import math

import numpy as np
from scipy.sparse.linalg import LinearOperator

from . import _approx_chol
from ._approx_chol import Clamp, Component, Config, ScalingPolicy

__all__ = [
    "ApproxInverse",
    "Clamp",
    "Component",
    "Config",
    "ScalingPolicy",
    "factorize",
    "factorize_raw",
]


class ApproxInverse(LinearOperator):
    def __init__(self, factor, /):
        super().__init__(np.float64, (factor.n, factor.n))
        self._factor = factor

    def __reduce__(self):
        return (ApproxInverse._load, (self._factor._saved_form(),))

    @classmethod
    def _load(cls, form):
        return cls(_approx_chol._ApproxInverse._load(form))

    @property
    def n(self):
        return self._factor.n

    @property
    def n_steps(self):
        return self._factor.n_steps

    @property
    def components(self):
        return self._factor.components

    @property
    def threads(self):
        return self._factor.threads

    def with_threads(self, threads):
        return ApproxInverse(self._factor.with_threads(threads))

    def apply(self, b, *, out=None):
        return self._factor.apply(b, out)

    def _matvec(self, x):
        if x.ndim == 1:
            return self.apply(x)
        # A stack of vectors, or scipy before 1.18's (n, 1) column, is a block of columns.
        stack = x.reshape(math.prod(x.shape[:-1]), self.n)
        return self.apply(stack.T).T.reshape(x.shape)

    def _matmat(self, X):
        if X.ndim == 2:
            return self.apply(X)
        stacked = np.moveaxis(X, -2, 0)
        block = stacked.reshape(self.n, math.prod(stacked.shape[1:]))
        return np.moveaxis(self.apply(block).reshape(stacked.shape), 0, -2)

    _rmatvec = _matvec
    _rmatmat = _matmat

    def _adjoint(self):
        return self

    _transpose = _adjoint


def factorize(matrix, *, config=None, scaling=None, threads=None):
    return ApproxInverse(_approx_chol.factorize(matrix, config, scaling, threads))


def factorize_raw(row_ptrs, col_indices, values, n, config=None, scaling=None):
    factor = _approx_chol.factorize_raw(
        row_ptrs, col_indices, values, n, config, scaling
    )
    return ApproxInverse(factor)
