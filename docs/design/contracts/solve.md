# Solve

How a caller applies $B$ to right-hand sides. The requirements are "Solve", "Solve a block" and
"Solve concurrently" in `../../requirements/consumers.md`.

## Decision

- **One solve for $k \ge 0$ columns:** right-hand sides and solutions as blocks of $B$'s size by
  $k$ in the caller's storage, the solutions written into the caller's block; a thread budget of
  the build's type, named in Rust, all cores by default in Python. A Schur $B$ is sized and
  numbered on the kept side. A block without columns is left as it is.
- **Each column equals its solve alone,** bit for bit, for every $k$ and every budget.
- **Target:** $B$ approximates $A_r^+$, or $H^+$ for a Schur $B$, $H$ the Schur complement of
  $A_r$. On a singular component the
  solve projects the right-hand side and the solution onto the range, and reports no defect.
- **Solver:** $B$ with a budget, named once; it sizes the scratch for every block and solves
  with the caller's scratch, or with its own, allocated per solve. Solving only reads $B$, so any
  number of solves run with one $B$ at once, each with its own scratch.
- **Threads:** the caller's pool, never a new one, at most the budget, safe nested in the caller's
  parallel work.
- **One `SolveError`:** a block of the wrong row count, or scratch that does not meet the
  solver's requirement. A block's own
  layout is checked where it is formed.
- **Python:** $B$ is a scipy `LinearOperator`, its adjoint itself: `matvec` and `matmat`, with
  `rmatvec` and `rmatmat` the same, at the default budget. An explicit solve takes a budget.

## Why

- **One solve:** a single right-hand side is a block of one column; one contract.
- **Bit-equal columns:** within groups right-hand sides freely only if grouping changes nothing;
  a block solve must therefore run each column's arithmetic exactly as a solve alone does.
- **Projection on both sides:** makes $B$ symmetric and the pseudoinverse in the limit, which LSMR
  needs; the defect is the caller's to compute, with the null vectors on $B$.
- **Caller's scratch:** within's local solvers take `&self` and the scratch of a faer `MemStack`;
  a solve per iteration allocates nothing. The allocating form serves a caller who solves once.
- **Budget named once:** a scratch sized for one budget and a solve at another cannot disagree.
- **$k = 0$:** faer's operators may apply to a block without columns.
- **`LinearOperator`:** it is what scipy's iterative solvers take; `lsmr` calls `rmatvec`.
