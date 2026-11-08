# Solve

How a caller applies $B$ to right-hand sides. The requirements are "Solve", "Solve a block" and
"Solve concurrently" in `../../requirements/consumers.md`.

## Decision

- **One solve for $k \ge 0$ columns:** right-hand sides and solutions as blocks of $B$'s size by
  $k$ in the caller's storage, the solutions written into the caller's block; a thread budget of
  the build's type, named in Rust, the held one by default in Python. A Schur $B$ is sized and
  numbered on the kept side. A block without columns is left as it is, whatever its rows.
- **Each column equals its solve alone,** bit for bit, for every $k$ and every budget.
- **Target:** $B$ approximates $A_r^+$, or $H^+$ for a Schur $B$, $H$ the Schur complement of
  $A_r$. On a singular component the
  solve projects the right-hand side and the solution onto the range, and reports no defect.
- **Per solve:** a budget and the caller's scratch, or scratch allocated for that solve; the
  scratch is sized by the block's column count and the budget, a region per thread for the
  components solved at once, and serves every block of fewer columns; one thread's serves every
  budget, solving fewer components at once, never other bits. Solving only reads $B$, so any
  number of solves run with one $B$ at once, each with its own scratch.
- **Threads:** as Execution, `../architecture.md`. A solve spreads its budget over $B$'s
  components, by size; the solution does not depend on how.
- **One error:** a block with columns of the wrong row count, or scratch that does not meet
  the block's requirement. A block's own layout is checked where it is formed.
- **Python:** $B$ is a scipy `LinearOperator`, its adjoint itself: `matvec` and `matmat`, with
  `rmatvec` and `rmatmat` the same, at a budget it holds, the build's unless
  another is named.

## Why

- **One solve:** a single right-hand side is a block of one column; one contract.
- **Bit-equal columns:** within groups right-hand sides freely only if grouping changes nothing;
  a block solve must therefore run each column's arithmetic exactly as a solve alone does.
- **Projection on both sides:** makes $B$ symmetric and the pseudoinverse in the limit, which LSMR
  needs; the defect is the caller's to compute, with the null vectors on $B$.
- **Caller's scratch:** within's local solvers take `&self` and the scratch of a faer `MemStack`;
  a solve per iteration allocates nothing. The allocating form serves a caller who solves once.
- **Budget per solve:** it changes no bit of a solution, so it is per-use state like the scratch.
- **Scratch by columns and budget:** a single right-hand side and a sequential solve are not
  charged for a block or a parallel one; a scratch that does not fit is an error, never a wrong
  solution.
- **Budget over components:** within holds one $B$ per channel pair, and a pair can split into
  many components; applied one after another, they leave the budget idle, 48–74% more apply
  time at 12 threads on pairs of 100 to 100,000 equal components, none on single-component
  pairs.
- **$k = 0$:** faer's operators may apply to a block without columns.
- **`LinearOperator`:** it is what scipy's iterative solvers take, as `M`, or composed with
  $A$ for `lsmr`, which takes no preconditioner and calls `rmatvec`.
