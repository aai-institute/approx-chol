# Architecture

The parts of approx-chol, the responsibility of each, and what crosses each boundary. The
requirements are in `../requirements/`.

## Decision

```
5  Adapters              Python binding, faer/sprs adapters
4  Approximate inverse   B: build, solve, report, save and load
3a Reduction             GDD → SDDM problems, lift, build facts, diagnostics
3b Cholesky              SDDM → factorization, diagnostics; solve with it
2  Input                 the closed admissible-matrix type and its facts
1a Storage               scalars, sparse storage
1b Execution             deterministic parallel execution
```

- A part uses only parts below it. Reduction and Cholesky never use each other. Adapters use
  only the approximate inverse and Input.
- The Rust API is the public face of Input and the approximate inverse, with their configuration
  and reports; it is not a part of its own.
- Each part owns its configuration, diagnostics, errors and encoding. The approximate inverse
  composes them and reaches into none. The adapters report through Input's error, since a
  conversion answers the same question, why a matrix is not admissible.

| Part | Package | Responsible for |
|---|---|---|
| Adapters | crate `approx-chol-py`; modules `faer`, `sprs`, each behind its feature | Translating another language's or library's types into Input and the approximate inverse; no mathematics. |
| Approximate inverse | module `approximate_inverse`, re-exported at the crate root | B as one object: building it per component, solving with it, the composed configuration and facts, the versioned envelope for persistence. Solving reads B and never changes it. |
| Reduction | module `reduction` | Everything that exists because the input is GDD and not SDDM: scaling, switching, the cover, components, the clamp, the classification into grounded and floating. The lift maps right-hand sides in and solutions out: projection, scaling, switching, the cover, and regauge in A's coordinates. |
| Cholesky | module `cholesky` | The approximate Cholesky factorization of an SDDM matrix, augmentation by the ground vertex, fill-in control, randomness, and the solve with the factorization, including the ground's lift and descent. Eliminating the side F of a bipartite factor-width-2 form yields a factorization of the Schur complement, numbered on K. |
| Input | module `input` | The closed admissible-matrix type: its representations, their validation, and the facts each carries. |
| Storage | module `storage` | Scalars and sparse storage. |
| Execution | module `execution` | Running independent work within a thread budget, nested in the caller's parallel work without deadlock, with results that do not depend on the budget. |

| From → to | Crosses |
|---|---|
| Caller or adapter → Input | a matrix in one representation; back: an admissible matrix or an input error |
| Caller or adapter → approximate inverse | admissible matrix, configuration, thread budget, observer; back: B with its facts, stopped, or a build error; to the observer: progress and diagnostics |
| Caller or adapter → approximate inverse (solve) | B, one or several right-hand sides, thread budget; back: the solutions |
| Caller or adapter → approximate inverse (persistence) | B; back: its saved form. Later the saved form; back: B or an error |
| Approximate inverse → Reduction | admissible matrix; back: SDDM problems with the vertices to eliminate, lift, and the build's facts: clamp, components, null vector of each singular component, diagnostics. During a solve: right-hand sides; back: their lifts, then solutions |
| Approximate inverse → Cholesky | one SDDM problem, vertices to eliminate, fill-in control, seed; back: factorization, diagnostics. During a solve: right-hand sides; back: solutions |
| Any part → Execution | independent work, thread budget |

## Why

- **Reduction and Cholesky as siblings:** each is tested against the mathematics alone, the
  first by exact identities, the second by its approximation of an SDDM inverse.
- **Input as a part:** the type is closed and callers, adapters and Reduction all need it.
- **Adapters as a part:** each depends on a foreign library and changes with it; Input stays free
  of those dependencies.
- **Projection and regauge in the lift:** they are coordinate semantics of A, which only Reduction
  knows; Cholesky sees at most a zero pivot.
- **Determinism in Execution:** proved once, not in every part that runs in parallel.
- **One core crate plus the Python crate:** a part becomes a crate only when something outside the
  crate forces it; pyo3 does for the binding, nothing does for the others.
- **Encodings per part:** the persistence format does not couple the approximate inverse to the
  internals of Reduction and Cholesky.
