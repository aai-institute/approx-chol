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
x = factor.apply(b)
```

### Preconditioning

`ApproxInverse` is a scipy `LinearOperator`, so it passes directly as `M` to iterative solvers:

```python
from scipy.sparse.linalg import cg

x, info = cg(laplacian, b, M=factor)
```

It applies at a thread budget, the build's `threads` unless `with_threads` names another:

```python
factor = factorize(laplacian, threads=4)
serial = factor.with_threads(1)  # shares the factor, same bits
X = serial.apply(B, out=X)  # an n × k block, into a preallocated array
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
x = factor.apply(b)
```

## API

- `factorize(matrix, *, config=None, scaling=None, threads=None)` — Factorize a scipy sparse array or matrix of any format, read through `tocsr()`, or an object with a CSR's `indptr`, `indices`, `data` and `shape`, at a thread budget, `None` every core.
- `factorize_raw(row_ptrs, col_indices, values, n, config=None, scaling=None)` — Factorize from raw CSR arrays.
- `ApproxInverse` — a `LinearOperator`; `apply(b, *, out=None)` solves a vector or an n × k block, into `out` when given; `threads` and `with_threads(threads)` read and change its budget; `components` lists each `Component`'s `vertices` and `clamp`.
- `Config(seed=0, split=1, exact_max_dim=24, *, frustrated="cover")` — Configuration for the factorization; `split` is the number of clique-tree copies each edge is sampled as, `1` standard AC and more AC2; `frustrated` carries a frustrated block's surplus on its cover to a ground vertex (`"cover_ground_vertex"`), onto edges between each vertex's copies (`"cover_surplus_edges"`), or onto those edges only inside the twist band of `CoverSurplus::Auto` (`"cover"`), or eliminates a frustrated block as it is, on one sheet, a bipartite block's complement formed signed (`"signed"`); blocks of at most `exact_max_dim` vertices, a bipartite block's kept side counted, are factored exactly by dense Cholesky, and `0` factors every block approximately.

## Attribution

This package ports key algorithmic ideas from [Laplacians.jl](https://github.com/danspielman/Laplacians.jl) to make AC and AC(k) accessible in Python.

Laplacians.jl is licensed under the MIT License.

## License

MIT — see [LICENSE](https://github.com/aai-institute/approx-chol/blob/main/LICENSE).

## References

- Gao, Yuan; Kyng, Rasmus; Spielman, Daniel (2025). *AC(k): Robust Solution of Laplacian Equations by Randomized Approximate Cholesky Factorization.* SIAM Journal on Scientific Computing.
- Gao, Kyng & Spielman (2023). *Robust and Practical Solution of Laplacian Equations by Approximate Elimination.* <https://arxiv.org/abs/2303.00709>
- Kyng & Sachdeva (2016). *Approximate Gaussian Elimination for Laplacians — Fast, Sparse, and Simple.* <https://arxiv.org/abs/1605.02353>
