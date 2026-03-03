# approx-chol

Approximate Cholesky factorization for graph Laplacians, implemented in Rust with Python bindings.

This project implements AC and AC(k), porting key algorithmic ideas from [Laplacians.jl](https://github.com/danspielman/Laplacians.jl) to make these algorithms accessible in the Rust and Python ecosystems.

AC(k) was introduced and analyzed by Gao, Kyng, and Spielman (2025), "AC(k): Robust Solution of Laplacian Equations by Randomized Approximate Cholesky Factorization" (SIAM Journal on Scientific Computing, November 2025, ISSN 1064-8275).


## Package READMEs

- `crates/approx-chol/README.md`: Rust crate documentation (Rust API only).
- `crates/approx-chol-py`: Python extension crate (`pyo3`).
- `python/approx_chol`: Python package surface.

## Development

```bash
pixi install
pixi run develop
cargo check --workspace
cargo test --workspace --all-features
pixi run test
```

## Attribution

This project brings the core algorithmic ideas behind the AC and AC(k) implementations of [Laplacians.jl](https://github.com/danspielman/Laplacians.jl) to the Rust and Python ecosystems.

This implementation is a Rust and Python-facing reimplementation of the AC and AC(k) algorithms developed by Daniel A. Spielman and other Laplacians.jl contributors.

Laplacians.jl is licensed under the MIT License.

## License

This project is licensed under `Apache-2.0 AND MIT`.

- Apache-2.0: see `LICENSE`.
- MIT (for Laplacians.jl-derived material): see `LICENSES/LAPLACIANS-MIT.txt`.
- Additional attribution and provenance notes: see `NOTICE`.


## References

- Gao, Yuan; Kyng, Rasmus; Spielman, Daniel (2025). *AC(k): Robust Solution of Laplacian Equations by Randomized Approximate Cholesky Factorization.* SIAM Journal on Scientific Computing.
- Gao, Kyng & Spielman (2023). *Robust and Practical Solution of Laplacian Equations by Approximate Elimination.* <https://arxiv.org/abs/2303.00709>
- Kyng & Sachdeva (2016). *Approximate Gaussian Elimination for Laplacians — Fast, Sparse, and Simple.* <https://arxiv.org/abs/1605.02353>
