# approx-chol

Approximate Cholesky factorization for SDDM and graph Laplacian systems.

This package implements AC and AC(k), porting key algorithmic ideas from [Laplacians.jl](https://github.com/danspielman/Laplacians.jl) to make these algorithms accessible in Python.

AC(k) was introduced and analyzed by Gao, Kyng, and Spielman (2025), "AC(k): Robust Solution of Laplacian Equations by Randomized Approximate Cholesky Factorization" (SIAM Journal on Scientific Computing, November 2025, ISSN 1064-8275).

## Install

```bash
pip install approx-chol
```

## Usage

```python
import numpy as np
from scipy.sparse import csr_array
from approx_chol import factorize

# 4-node path graph Laplacian (0-1-2-3)
row_ptrs = np.array([0, 2, 5, 8, 10], dtype=np.uint32)
col_indices = np.array([0, 1, 0, 1, 2, 1, 2, 3, 2, 3], dtype=np.uint32)
values = np.array([1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0])

laplacian = csr_array((values, col_indices, row_ptrs), shape=(4, 4))

factor = factorize(laplacian)

# RHS must lie in the range of the Laplacian (sum to zero)
b = np.array([1.0, -1.0, 1.0, -1.0])
x = factor.solve(b)
```

### Configuration

```python
from approx_chol import Config, factorize

config = Config(seed=42, split=3)
factor = factorize(laplacian, config=config)
```

### Low-level API

For direct control over CSR arrays without constructing a scipy sparse matrix:

```python
from approx_chol import factorize_raw

factor = factorize_raw(row_ptrs, col_indices, values, n=4)
x = factor.solve(b)
```

## API

- `factorize(matrix, config=None)` — Factorize a `scipy.sparse.csr_array` or `csr_matrix`.
- `factorize_raw(row_ptrs, col_indices, values, n, config=None)` — Factorize from raw CSR arrays.
- `Factor.solve(b)` — Solve LDL^T x = b, returning a new array.
- `Factor.solve_into(b, out)` — Solve in-place into a pre-allocated array.
- `Config(seed=0, split=None)` — Configuration for the factorization.

## Attribution

This package ports key algorithmic ideas from [Laplacians.jl](https://github.com/danspielman/Laplacians.jl) to make AC and AC(k) accessible in Python.

Laplacians.jl is licensed under the MIT License.

## License

MIT — see [LICENSE](https://github.com/aai-institute/approx-chol/blob/main/LICENSE).

## References

- Gao, Yuan; Kyng, Rasmus; Spielman, Daniel (2025). *AC(k): Robust Solution of Laplacian Equations by Randomized Approximate Cholesky Factorization.* SIAM Journal on Scientific Computing.
- Gao, Kyng & Spielman (2023). *Robust and Practical Solution of Laplacian Equations by Approximate Elimination.* <https://arxiv.org/abs/2303.00709>
- Kyng & Sachdeva (2016). *Approximate Gaussian Elimination for Laplacians — Fast, Sparse, and Simple.* <https://arxiv.org/abs/1605.02353>
