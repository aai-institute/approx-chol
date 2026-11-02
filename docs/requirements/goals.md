# Goals

approx-chol computes approximate Cholesky factorizations for the matrices fixed in `input.md`, and
from them an approximate inverse, for example a preconditioner for an iterative solver.

## Context

Exact Cholesky eliminates a pivot by adding a clique on its neighbors. This fill-in is what makes
exact factorization of large sparse matrices expensive in time and memory. Approximate Cholesky
replaces each clique by a sparse random sample whose expectation is the clique. Fill-in then
drives factorization time, solve time and memory, and trades against the quality of the
approximate inverse. Controlling it is the key to performance and memory economy.

## Requirements

Each requirement names its source: math (the context above), a consumer, or the product
(maintainer).

- **Performance:** factorization and solve run in time near-linear in the input's nonzeros.
  Math.
- **Memory economy:** peak memory is proportional to the input and the factorizations, without
  redundant copies. Math.
- **Configurability:** the caller controls fill-in, trading the quality of the approximate inverse
  against cost, and chooses the seed and the thread budget (math). The caller also sets the
  dominance scaling's tolerance and step limit, trading the size of a clamp against cost (within,
  which passes them to its users), and the size up to which a component is factored exactly
  (product).
- **Caller ergonomics:** the API is easy to use correctly, and defaults serve the common case.
  Product.
- **Reproducibility:** the same input, configuration and seed give the same approximate inverse.
  Product.
- **Thread determinism:** every result is bit-identical for every thread budget. within; no
  surveyed library provides it, so the crate cannot inherit it.
- **Persistence:** an approximate inverse can be saved and loaded without refactorizing. Product.
