# approx-chol

Approximate Cholesky factorization for SDDM and graph Laplacian systems.

Provides a robust Rust implementation of approximate Cholesky (AC) factorization,
suitable as a preconditioner for iterative solvers on symmetric diagonally dominant
(SDDM) linear systems. SDDM matrices arise naturally in graph Laplacians,
finite-element discretizations, and fixed-effects normal equations. Every graph
Laplacian is SDDM; every SDDM matrix can be converted to a Laplacian via
Gremban's reduction (1996).

## Install

```toml
[dependencies]
approx-chol = "0.1"
```

Or with Cargo:

```
cargo add approx-chol
```

## Example

```rust
use approx_chol::{factorize, CsrRef};

// 4-node path graph Laplacian (0-1-2-3)
let row_ptrs    = [0u32, 2, 5, 8, 10];
let col_indices = [0u32, 1, 0, 1, 2, 1, 2, 3, 2, 3];
let values      = [1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0];

let csr = CsrRef::new(&row_ptrs, &col_indices, &values, 4)?;
let decomp = factorize(csr)?;

// RHS must lie in the range of the Laplacian (sum to zero)
let b = [1.0, -1.0, 1.0, -1.0];
let x = decomp.solve(&b);
assert!(x.iter().all(|v| v.is_finite()));
```

For a larger example with a grid Laplacian, see [`examples/basic_solve.rs`](examples/basic_solve.rs).

## Feature flags

| Feature | Effect |
|---------|--------|
| `sprs`  | Zero-copy `CsrRef` conversion from `sprs` matrices (`From` and fallible `try_from_sprs*`). |
| `faer`  | Zero-copy `CsrRef` conversion from `faer` matrices (`From` and fallible `try_from_faer*`). |

## License

MIT

## References

- Gao, Kyng & Spielman (2023). *Robust and Practical Solution of Laplacian Equations by Approximate Elimination.* <https://arxiv.org/abs/2303.00709>
- Kyng & Sachdeva (2016). *Approximate Gaussian Elimination for Laplacians — Fast, Sparse, and Simple.* <https://arxiv.org/abs/1605.02353>
- Gremban (1996). *Combinatorial Preconditioners for Sparse, Symmetric, Diagonally Dominant Linear Systems.* Ph.D. thesis, CMU.
