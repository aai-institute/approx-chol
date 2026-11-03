# Consumers

Who calls approx-chol and what each needs to do. The goals are in `goals.md`.

## Requirements

### Consumers

- **within:** least squares with LSMR and a Schwarz preconditioner. It builds one approximate
  inverse per subdomain, often singular and disconnected, one subdomain dominating in size, and
  solves with them in every iteration from its own parallelism.
- **Python users:** hold scipy sparse matrices and numpy arrays and pass the approximate inverse to
  an iterative solver.
- **Rust users of faer, sprs, nalgebra or ndarray:** hold that library's sparse matrices, where
  it has them, and its dense vectors, and bring their own iterative solver, or faer's.
- **Graph users:** hold a weighted graph, a Laplacian, or a Laplacian with surplus; often singular.
- **Persisting users:** build once and solve in another process.

### Actions

| Action | The caller has | It needs back | How often |
|---|---|---|---|
| Submit a matrix | the matrix in one representation | whether it is admissible, and if not, why | per problem |
| Build | an admissible matrix, fill-in level, seed, thread budget | the approximate inverse, and the facts of the build: whether and where the matrix was clamped, its components, the null vector of each singular component, diagnostics | per problem |
| Build for a Schur complement | a bipartite factor-width-2 form, the same settings | the approximate inverse of the Schur complement on the kept side, and the same facts | per problem |
| Solve | the approximate inverse, one right-hand side, thread budget | the solution | per iteration |
| Solve a block | the approximate inverse, several right-hand sides, thread budget | their solutions, each equal to solving that column alone | per iteration |
| Solve concurrently | one approximate inverse, used from several caller threads | the same results as one at a time | per iteration |
| Save and load | the approximate inverse; later the saved form | the saved form; later the same approximate inverse, or an error | once; per process |

### Across all actions

- **Singular components:** the approximate inverse is symmetric, and its target is the
  pseudoinverse.
- **Caller's parallelism:** the crate is safe nested inside the caller's parallel work, without
  deadlock, and never uses more threads than the budget it is given.
- **Watching and stopping a build:** the caller can follow a build's progress and diagnostics,
  and stop it.

## Sources

- **Null vectors:** domain-decomposition coarse spaces (Nicolaides, GDSW, GenEO, BDDC with
  approximate local solves) are built from them, and only the crate knows them cheaply, since they
  involve the computed scaling. within can check its classification with them. Source: prior art,
  consumer (within).
- **Symmetric, pseudoinverse target:** LSMR needs a symmetric preconditioner, and on a singular
  component the pseudoinverse is the symmetric target. Source: math, consumer (within).
- **Block solve:** within's block LSMR solves several right-hand sides per iteration; equality with
  column solves lets it group them freely. Source: consumer (within).
- **Thread budget per solve:** within sets the budget for each application from its own
  scheduling. Source: consumer (within).
- **Nested, within the budget:** within calls the crate from inside its own parallel work and
  must not deadlock there or oversubscribe its cores. Source: consumer (within).
- **Watching and stopping:** a person watching a long build, within's own control of its
  builds, profiling, and tests; Python users expect Ctrl-C to stop a call. Source: consumer
  (within, Python users), product (maintainer).
