# approx-chol

Approximate Cholesky factorization for symmetric generalized diagonally dominant systems, graph Laplacians among them.

This crate implements AC and AC(k), porting key algorithmic ideas from [Laplacians.jl](https://github.com/danspielman/Laplacians.jl) to make these algorithms accessible in the Rust ecosystem and through Python bindings.

AC(k) was introduced and analyzed by Gao, Kyng, and Spielman (2025), "AC(k): Robust Solution of Laplacian Equations by Randomized Approximate Cholesky Factorization" (SIAM Journal on Scientific Computing, November 2025, ISSN 1064-8275).

## Install

```toml
[dependencies]
approx-chol = "0.7"
```

Or with Cargo:

```
cargo add approx-chol
```

Upgrading from 0.7: [MIGRATING.md](https://github.com/aai-institute/approx-chol/blob/main/MIGRATING.md).

## Example

```rust
use approx_chol::input::CsrRef;
use approx_chol::{factorize, Threads};

// 4-vertex path graph Laplacian (0-1-2-3)
let row_ptrs    = [0u32, 2, 5, 8, 10];
let col_indices = [0u32, 1, 0, 1, 2, 1, 2, 3, 2, 3];
let values      = [1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0];

let csr = CsrRef::new(4, &row_ptrs, &col_indices, &values)?;
let factor = factorize(csr)?;

// A Laplacian is singular: b sums to zero, and x is orthogonal to the all-ones vector
let mut x = vec![1.0, -1.0, 1.0, -1.0];
factor.apply(&mut x, Threads::All)?;
assert!(x.iter().all(|v| f64::is_finite(*v)));
```

`apply` solves a block of right-hand sides at once: a `Vec<[f64; K]>` holds `K` interleaved columns, solved in place, each bit for bit as it solves alone; `apply_with` takes the caller's `MemStack` sized by `scratch(columns, threads)` and allocates nothing at `Threads::Count(1)`; a budget above one runs on rayon's pool, which builds itself on first use.

A `Laplacian` or `Sddm`, a `FactorWidth2` or a `BipartiteFactorWidth2` also factors from its edges or columns through `approx_chol::input`; see [`examples/basic_solve.rs`](examples/basic_solve.rs). A signed or signless graph factors as a `CsrRef` whose diagonal is `Σ|w|` over each vertex's edges; repeated entries in a row sum.

## Feature flags

| Feature | Effect |
|---------|--------|
| `sprs`  | Zero-copy `CsrRef` conversion from `sprs` matrices via `TryFrom`; enables `ndarray`. |
| `faer`  | Zero-copy `CsrRef` conversion from `faer` matrices via `TryFrom`. |
| `nalgebra` | Zero-copy `CsrRef` conversion from a `nalgebra-sparse` 0.12 `CsrMatrix` or `CscMatrix` via `TryFrom`; every `nalgebra` 0.35 matrix, owned or a view, is a `MatrixMut` block for `apply`, a contiguous one copied less through `ColumnMajor::new(m.as_mut_slice(), m.nrows())`. Needs Rust 1.89, nalgebra's MSRV, above the crate's 1.85. |
| `ndarray` | `ndarray` 0.17's `ArrayRef2` and `ArrayRef1` are `MatrixMut` blocks for `apply`: an `Array2`, an `Array1` or a view of either, passed as `&mut *a`; an `ArrayD` converts with `into_dimensionality::<Ix2>()`. Columns of row stride 1, as in a Fortran-order array, are solved where they are. |
| `serde` | Serialize and deserialize `ApproxInverse`, led by its `Header`: format version and scalar type. |

## Attribution

This crate ports key algorithmic ideas from Laplacians.jl to make AC and AC(k) accessible in the Rust ecosystem and through Python bindings.

This implementation is a Rust and Python-facing reimplementation of the AC and AC(k) algorithms developed by Daniel A. Spielman and other Laplacians.jl contributors.

Laplacians.jl is licensed under the MIT License.

## License

This crate is licensed under `MIT`.

- MIT: see `LICENSE`.
- Additional attribution and provenance notes: see `NOTICE`.

## References

- Gao, Yuan; Kyng, Rasmus; Spielman, Daniel (2025). *AC(k): Robust Solution of Laplacian Equations by Randomized Approximate Cholesky Factorization.* SIAM Journal on Scientific Computing.
- Gao, Kyng & Spielman (2023). *Robust and Practical Solution of Laplacian Equations by Approximate Elimination.* <https://arxiv.org/abs/2303.00709>
- Kyng & Sachdeva (2016). *Approximate Gaussian Elimination for Laplacians — Fast, Sparse, and Simple.* <https://arxiv.org/abs/1605.02353>
