# Input

The matrices approx-chol accepts and the forms callers give them in.

## Requirements

- **Class.** Real symmetric GDD matrices, weak: singular and disconnected matrices are admitted.
  Single and double precision.
- **Representations.** An admissible matrix can be given in each of these forms:

  | Representation | Carries exactly |
  |---|---|
  | CSR, full or upper triangle | the entries |
  | Laplacian of a weighted graph | zero row sums |
  | SDDM: Laplacian plus surplus | each row's surplus |
  | Factor-width-2 form | `A = VWVᵀ`, hence a dominance scaling and any null vector |
  | Bipartite factor-width-2 form | as factor-width-2, and a split into two sides |

- **Scope.** No other class and no other representation.

## Sources

- **Class:** within and graph users hold symmetric GDD matrices, often singular and disconnected.
  GDD is also the widest class approximate Cholesky reaches: it factors SDDM matrices, and a
  matrix reduces exactly to SDDM iff it is GDD. Source: consumer, math: Boman, Chen, Parekh,
  Toledo, "On factor width and symmetric H-matrices", Linear Algebra Appl. (2005): symmetric GDD
  with nonnegative diagonal ⇔ factor width ≤ 2. The reduction: a dominance scaling (Daitch,
  Spielman, "Faster approximate lossy generalized flow via interior point algorithms", STOC 2008,
  arXiv:0803.0988, §4), then the Gremban cover (Gremban, "Combinatorial preconditioners for
  sparse, symmetric, diagonally dominant linear systems", PhD thesis, CMU-CS-96-123, 1996).
- **Representations:** each form beyond CSR states a fact that CSR leaves the crate to recover
  numerically, up to rounding: a zero row sum, a surplus, a scaling, a split. Python, faer, sprs
  and nalgebra-sparse users hold CSR or CSC; graph users hold Laplacians, with or without
  surplus; within forms Gram matrices of columns with at most two nonzeros, the factor-width-2
  form, and eliminates one side for a Schur complement. Source: math, consumer.
- **Scope:** Source: product (maintainer).
