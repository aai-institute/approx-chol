# Solve

How a caller applies $B$ to right-hand sides. The requirements are "Solve", "Solve a block" and
"Solve concurrently" in `../../requirements/consumers.md`.

## Decision

- **One solve for $k \ge 1$ columns:** right-hand sides and solutions as column-major blocks of
  $B$'s size by $k$, the solutions written into the caller's block; a thread budget of the build's
  type, named in Rust, all cores by default in Python. A Schur $B$ is sized and numbered on the
  kept side.
- **Each column equals its solve alone,** bit for bit, for every $k$ and every budget.
- **Target:** $B$ approximates $A_r^+$, or $H^+$ for a Schur $B$, $H$ the Schur complement of
  $A_r$. On a singular component the
  solve projects the right-hand side and the solution onto the range, and reports no defect.
- **Solver:** a handle borrowed from $B$ for at most some $k$ columns, owning the scratch; one per
  concurrent solve. Solving only reads $B$, so any number of handles solve with one $B$ at once.
- **Threads:** the caller's pool, never a new one, at most the budget, safe nested in the caller's
  parallel work.
- **One `SolveError`:** a block of the wrong shape.
- **Python:** $B$ is a scipy `LinearOperator`, its adjoint itself: `matvec` and `matmat`, with
  `rmatvec` and `rmatmat` the same, at the default budget and with their own scratch. An explicit
  solve takes a budget and a solver.

## Why

- **One solve:** a single right-hand side is a block of one column; one contract.
- **Bit-equal columns:** within groups right-hand sides freely only if grouping changes nothing.
- **Projection on both sides:** makes $B$ symmetric and the pseudoinverse in the limit, which LSMR
  needs; the defect is the caller's to compute, with the null vectors on $B$.
- **Solver borrowed from $B$:** a handle cannot meet another $B$, so no mismatch is checked; sized
  once for its columns, a solve per iteration allocates nothing.
- **`LinearOperator`:** it is what scipy's iterative solvers take; `lsmr` calls `rmatvec`.
