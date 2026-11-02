# Submit

How a caller turns a matrix in one representation into an admissible matrix, or learns why it is
not one. The requirement is "Submit a matrix" in `../../requirements/consumers.md`; the
representations are those of `../../requirements/input.md`.

## Decision

- **One type per representation, validated on construction.**

  | Type | Given as | Accepted when, besides the common checks |
  |---|---|---|
  | `CsrRef` | CSR storage of the whole matrix | symmetric up to rounding; no inadmissible row |
  | `UpperCsrRef` | CSR storage of the upper triangle | no entry below the diagonal; no row of the whole matrix inadmissible |
  | `Laplacian` | edge list, optional weights | weights $\ge 0$; no self-loop |
  | `Sddm` | a `Laplacian` and one surplus per vertex | every surplus $\ge 0$, not all $0$ |
  | `FactorWidth2` | columns of $V$, each two (row, value) pairs, the values absorbing $W$ | the two rows of a column differ |
  | `BipartiteFactorWidth2` | two width-1 factors, the two sides, each column one (row, value) pair per side; optional column weights | both sides have the same column count; weights $\ge 0$ |

  Common checks: well-formed storage, indices in range, lengths that agree, finite values,
  magnitudes and a size the build handles. A value may be $0$.

- **`Input`** is the closed enumeration of these types and has no other constructor, so every
  `Input` is admissible. The Schur-complement build takes only a `BipartiteFactorWidth2`.
- **Admissible is structural:** Submit decides only properties of the entries, not whether the
  matrix is GDD.
- **One `InputError`:** the defect and its position in the caller's terms. Which of several
  defects is reported is unspecified but stable.
- **Borrowed, no accessors;** values `f32` or `f64`, indices any primitive integer.

## Why

- **A type per representation:** a caller looks for what they hold by its name. Beyond the two CSR
  forms, a representation earns a type only if it carries a fact CSR leaves to numerical recovery.
- **Validation in the constructor:** otherwise Build would repeat the checks and the caller would
  have two places to learn why.
- **Inadmissible rows only in CSR:** every other form is a sum of PSD terms.
- **Self-loops and all-zero surpluses rejected:** a self-loop contributes nothing, so it is a defect
  in the caller's data; an `Sddm` with no surplus is a `Laplacian`, and each matrix has one
  representation.
- **Structural admissibility:** deciding GDD means running the dominance scaling, which is the
  build's; a non-GDD input is clamped and reported (`../../algorithm.md`, Clamp).
- **Borrowing:** memory economy; the build copies what $B$ needs.
- **`BipartiteFactorWidth2` from width-1 factors:** that is how within forms it.
- **No accessors:** no consumer reads back what it submitted.
