# approx-chol

Approximate Cholesky factorization for SDDM and graph Laplacian systems.

This crate implements AC and AC(k), porting key algorithmic ideas from [Laplacians.jl](https://github.com/danspielman/Laplacians.jl) to make these algorithms accessible in the Rust ecosystem and through Python bindings.

AC(k) was introduced and analyzed by Gao, Kyng, and Spielman (2025), "AC(k): Robust Solution of Laplacian Equations by Randomized Approximate Cholesky Factorization" (SIAM Journal on Scientific Computing, November 2025, ISSN 1064-8275).

Provides a robust Rust implementation of approximate Cholesky factorization,
suitable as a preconditioner for iterative solvers on symmetric diagonally dominant
(SDDM) linear systems.

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
let x = decomp.solve(&b).expect("rhs length must be <= factor dimension");
assert!(x.iter().all(|v| v.is_finite()));
```

For a larger example with a grid Laplacian, see [`examples/basic_solve.rs`](examples/basic_solve.rs).

## Feature flags

| Feature | Effect |
|---------|--------|
| `sprs`  | Zero-copy `CsrRef` conversion from `sprs` matrices (`TryFrom` and `try_from_sprs*`). |
| `faer`  | Zero-copy `CsrRef` conversion from `faer` matrices (`TryFrom` and `try_from_faer*`). |

## Attribution

This crate ports key algorithmic ideas from Laplacians.jl to make AC and AC(k) accessible in the Rust ecosystem and through Python bindings.

This implementation is a Rust and Python-facing reimplementation of the AC and AC(k) algorithms developed by Daniel A. Spielman and other Laplacians.jl contributors.

Laplacians.jl is licensed under the MIT License.

## License

This crate is licensed under `Apache-2.0 AND MIT`.

- Apache-2.0: see `LICENSE`.
- MIT (for Laplacians.jl-derived material): see `LICENSE-MIT`.
- Additional attribution and provenance notes: see `NOTICE`.

## References

- Gao, Yuan; Kyng, Rasmus; Spielman, Daniel (2025). *AC(k): Robust Solution of Laplacian Equations by Randomized Approximate Cholesky Factorization.* SIAM Journal on Scientific Computing.
- Gao, Kyng & Spielman (2023). *Robust and Practical Solution of Laplacian Equations by Approximate Elimination.* <https://arxiv.org/abs/2303.00709>
- Kyng & Sachdeva (2016). *Approximate Gaussian Elimination for Laplacians — Fast, Sparse, and Simple.* <https://arxiv.org/abs/1605.02353>
