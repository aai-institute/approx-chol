# Architecture

The parts of approx-chol, the responsibility of each, and what crosses each boundary. The
requirements are in `../requirements/`.

## Decision

```
5  Adapters              Python binding; faer, sprs, nalgebra, ndarray adapters
4  Approximate inverse   B: build, solve, report, save and load
3a Reduction             GDD → SDDM problems, lift, build facts
3b Cholesky              SDDM → factorization; solve with it
2  Input                 the closed admissible-matrix type and its facts
1a Storage               scalars, sparse storage
1b Execution             deterministic parallel execution
```

- A part uses only parts below it. Reduction and Cholesky never use each other. Adapters use
  only the public surface.
- The Rust API is the public face of Input and the approximate inverse, with their configuration
  and reports; it is not a part of its own.
- Each part owns its configuration, errors and encoding. The approximate inverse
  composes them and reaches into none. The adapters report through Input's error for a matrix,
  since a conversion answers the same question; a foreign block needs no check.
- **Interoperability:** one feature per foreign crate, named after it: `faer` 0.24 with faer's
  `rayon`, `sprs` 0.11, `nalgebra` 0.35 with nalgebra-sparse 0.12, `ndarray` 0.17; `sprs` enables
  `ndarray`. dyn-stack 0.13 is a required dependency, re-exported. A Rust adapter borrows, never
  copies: a matrix converts into Input and fails as Input validates; a dense type implements the
  solve's block trait, which the solve reads and writes column by column through its scratch. Moving
  a feature to another version is a breaking release. Python converts a form it cannot borrow, as
  documented.

| Part | Package | Responsible for |
|---|---|---|
| Adapters | crate `approx-chol-py`; modules `faer`, `sprs`, `nalgebra`, `ndarray`, each behind its feature | Translating another language's or library's matrices into Input, implementing the solve's block trait for their dense types, and the approximate inverse as faer's operator traits; no mathematics. |
| Approximate inverse | module `approximate_inverse`, re-exported at the crate root | B as one object: building it per component, solving with it, the composed configuration and facts, the versioned envelope for persistence. Solving reads B and never changes it. |
| Reduction | module `reduction` | Everything that exists because the input is GDD and not SDDM: scaling, switching, the cover, components, the clamp, the classification into grounded and floating. The lift maps right-hand sides in and solutions out: projection, scaling, switching, the cover, and regauge in A's coordinates. |
| Cholesky | module `cholesky` | The approximate Cholesky factorization of an SDDM matrix, augmentation by the ground vertex, fill-in control, randomness, and the solve with the factorization, including the ground's lift and descent. Eliminating the side F of a bipartite factor-width-2 form yields a factorization of the Schur complement, numbered on K. |
| Input | module `input` | The closed admissible-matrix type: its representations, their validation, and the facts each carries. |
| Storage | module `storage` | Scalars and sparse storage. |
| Execution | module `execution` | Running independent work within a thread budget, nested in the caller's parallel work without deadlock, with results that do not depend on the budget. |

| Part | Public, at the crate root | Crate-internal |
|---|---|---|
| Adapters, Rust | impls only: `TryFrom` faer, sprs and nalgebra-sparse matrices for `CsrRef`; `MatrixMut` for faer, nalgebra and ndarray blocks; faer's `LinOp`, `BiLinOp`, `Precond`, `BiPrecond` for `ApproxInverse`; `From<faer::Par>` for `Threads` | none |
| Adapters, Python | module `approx_chol`: `factorize`, `ApproxInverse`, `Component`, `Clamp`, `Config`, `ScalingPolicy`, `InputError`, `BuildError` | none |
| Approximate inverse | `ApproxInverse`, `Component`, `Sgdd`, `Complement`, `factorize`, `factorize_with`, `factorize_complement`, `MatrixMut` with its impls for slices, `Vec` and arrays, `ColumnMajor`, `Columns`, `ApplyError`, `LayoutError`, `dyn_stack`; serde for `ApproxInverse` | none |
| Reduction | `ScalingPolicy`, `Clamp`, `ScalingOutcome`, `BuildError` | `Reduced` (the SDDM matrices with the vertices each keeps, the lift, the facts), `Lift` with its scratch requirement |
| Cholesky | `Config`, `SchurMode` | `Factorization` with its scratch requirement |
| Input | module `input`: `CsrRef`, `UpperCsrRef`, `Laplacian`, `Sddm`, `FactorWidth2`, `FactorWidth1`, `BipartiteFactorWidth2`, `Input`, `InputError`, `Side` | the representation an `Input` holds |
| Storage | `Real`, `PrimInt` (num-traits') | `Csr`, `SddmCsr`, the storage index |
| Execution | `Threads` | `join`, `fork` |

- A part offers only its exports. Visibility enforces it: a part's other items are at most
  `pub(in crate::part)`, its exports `pub(crate)` or public. A test checks that every such item is
  listed here and that every path between parts follows the layer order, and counts the distinct
  part-to-part edges and the exports. The public surface over all features is a committed
  snapshot whose every change CI shows; Python's is the stub that stubtest checks.
- Each part's exports serialize themselves; on load, the approximate inverse checks that the
  parts agree, as their owners' checks state, without reading their representations.

| From → to | Crosses |
|---|---|
| Caller or adapter → Input | a matrix in one representation; back: an admissible matrix or an input error |
| Caller or adapter → approximate inverse | admissible matrix, configuration, thread budget; back: B with its facts, or a build error |
| Caller or adapter → approximate inverse (solve) | B; per solve one or several right-hand sides, a thread budget, scratch; back: the solutions, or a solve error |
| Caller or adapter → approximate inverse (persistence) | B; back: its saved form. Later the saved form; back: B or an error |
| Approximate inverse → Reduction | admissible matrix; back: SDDM matrices with the vertices each keeps, the lift with its scratch requirement, and the build's facts: clamp, components, null vector of each singular component. During a solve: right-hand sides; back: their lifts, then solutions |
| Approximate inverse → Cholesky | one SDDM matrix, the vertices it keeps, fill-in control, seed and the component's stream; back: factorization with its scratch requirement. During a solve: right-hand sides; back: solutions |
| Any part → Execution | independent work, thread budget |

## Why

- **Reduction and Cholesky as siblings:** each is tested against the mathematics alone, the
  first by exact identities, the second by its approximation of an SDDM inverse.
- **Input as a part:** the type is closed and callers, adapters and Reduction all need it.
- **Adapters as a part:** each depends on a foreign library and changes with it; Input stays free
  of those dependencies.
- **dyn-stack required:** within and faer size and pass scratch with it; one protocol composes
  their requirement with ours.
- **`sprs` enables `ndarray`, at 0.17:** sprs users hold their dense vectors as ndarray arrays;
  sprs 0.11 accepts ndarray 0.15 to 0.17, and one pinned version keeps them one type.
- **Projection and regauge in the lift:** they are coordinate semantics of A, which only Reduction
  knows; Cholesky sees at most a zero pivot.
- **Determinism in Execution:** proved once, not in every part that runs in parallel.
- **One core crate plus the Python crate:** a part becomes a crate only when something outside the
  crate forces it; pyo3 does for the binding, nothing does for the others.
- **Exports named, signatures not:** a name list is the vocabulary of a boundary and changes
  with the design; signatures written here would be a second copy of the code. The snapshot and
  the edge test check the code itself, so an addition such as `BlockFactor` cannot pass unseen:
  semver-checks allows additions.
- **`Sgdd` and `Complement` in the approximate inverse:** their `factor` returns B, which
  Reduction may not name; they hold Reduction's `Reduced`.
- **`SddmCsr` in Storage:** Reduction makes it and Cholesky takes it, and neither uses the
  other; the kept vertices travel beside it as indices, so Storage holds no elimination.
- **`BuildError` in Reduction:** both its cases come from the reduction, the scaling and the
  components; Cholesky does not fail.
- **Visibility, then a test:** the compiler rejects any use of a part's private item, trait
  methods and macros included; the test adds what visibility cannot say, the layer order and the
  list.
- **Encodings per part:** the persistence format does not couple the approximate inverse to the
  internals of Reduction and Cholesky.
