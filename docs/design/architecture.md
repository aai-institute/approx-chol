# Architecture

The parts of approx-chol, the responsibility of each, and what crosses each boundary. The
requirements are in `../requirements/`.

## Decision

```
5  Adapters              Python binding; faer, sprs, nalgebra, ndarray adapters
4  Approximate inverse   B: build, solve, report, save and load
3a Reduction             GDD → WDD problems, lift, build facts
3b Cholesky              WDD → factorization; solve with it
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
| Approximate inverse | module `approximate_inverse`, re-exported at the crate root | B as one object: building it per component, solving with it, the composed configuration and facts, the versioned envelope for persistence. A vertex no entry couples is a component of its own, solved as $1/\sigma$ here. |
| Reduction | module `reduction` | Everything that exists because the input is GDD and not WDD: scaling, switching, components, the clamp, the classification into balanced or frustrated and grounded or floating. The lift maps right-hand sides in and solutions out: projection, scaling, switching, and regauge in A's coordinates. |
| Cholesky | module `cholesky` | The approximate Cholesky factorization of a WDD matrix: a frustrated one factored by its Gremban cover or by signed elimination, as configured; augmentation by the ground vertex, pinned, never a pivot; the clique sample's rule and fill-in level; randomness; and the solve with the factorization, including the cover's and the ground's lift and descent. Eliminating the side F of a bipartite factor-width-2 form yields a factorization of the Schur complement, numbered on K. |
| Input | module `input` | The closed admissible-matrix type: its representations, their validation, and the facts each carries. |
| Storage | module `storage` | Scalars and sparse storage. |
| Execution | module `execution` | Running independent work in the caller's pool, never a new one, within a thread budget, nested in the caller's parallel work without deadlock, with results that do not depend on the budget. |

| Part | Public, at the crate root | Crate-internal |
|---|---|---|
| Adapters, Rust | impls only: `TryFrom` faer, sprs and nalgebra-sparse matrices for `CsrRef`; `MatrixMut` for faer, nalgebra and ndarray blocks; faer's `LinOp`, `BiLinOp`, `Precond`, `BiPrecond` for `ApproxInverse`; `From<faer::Par>` for `Threads` | none |
| Adapters, Python | module `approx_chol`: `factorize`, `ApproxInverse`, `Component`, `Clamp`, `Config`, `ScalingPolicy`, `InputError`, `BuildError` | none |
| Approximate inverse | `ApproxInverse`, `Component`, `Sgdd`, `Complement`, `Clamped`, `factorize`, `factorize_with`, `factorize_complement`, `MatrixMut` with its impls for slices, `Vec` and arrays, `ColumnMajor`, `Columns`, `ApplyError`, `LayoutError`, `dyn_stack`, `Header`, `Scalar`, `HeaderMismatch`; serde for `ApproxInverse` | none |
| Reduction | `ScalingPolicy`, `Clamp`, `ScalingOutcome`, `BuildError` | `Reduced` (per component the WDD matrix with its grounding, its balance, gauge and key, the lift, its clamp), `Lift` with its scratch requirement, the bound on a lone pivot |
| Cholesky | `Config`, `SchurMode`, `Sampling`, `FrustratedRoute`, `CoverSurplus` | `factor`, `Factorization` with its scratch requirement, `fits_u32` |
| Input | module `input`: `CsrRef`, `TriangleCsrRef`, `Laplacian`, `Sddm`, `FactorWidth2`, `FactorWidth1`, `BipartiteFactorWidth2`, `Input`, `InputError`, `Side` | the representation an `Input` holds |
| Storage | `Real`, `PrimInt` (num-traits') | `Csr`, `WddCsr`, `Bipartite`, `Adjacency` (Reduction's view of either shape), `Grounding`, `Grounded`, `Balance`, `Gauge`, the storage index, the compensated sums and range screens |
| Execution | `Threads` | `join`, `fork` |

- A part offers only its exports. Visibility enforces it: a part's other items are at most
  `pub(in crate::part)`, its exports `pub(crate)` or public. The code owns the export list: a
  part's exports are its root's re-exports, and the public ones are also in the snapshot; the
  table above is the target until P2 makes the parts. A test checks that every path between parts
  follows the layer order, counts the distinct part-to-part edges and the exports, and writes them
  to a generated `parts.txt`; doc links are not layer edges. The public surface over all features
  is a committed snapshot whose every change CI shows; Python's is the stub that stubtest checks.
- Each part's exports serialize themselves, under the envelope's one format version; no part
  versions its own. On load, the approximate inverse checks that the parts agree, as their
  owners' checks state, without reading their representations.

| From → to | Crosses |
|---|---|
| Caller or adapter → Input | a matrix in one representation; back: an admissible matrix or an input error |
| Caller or adapter → approximate inverse | admissible matrix, configuration, thread budget; back: B with its facts, or a build error |
| Caller or adapter → approximate inverse (solve) | B; per solve one or several right-hand sides, a thread budget, scratch; back: the solutions, or a solve error |
| Caller or adapter → approximate inverse (persistence) | B; back: its saved form. Later the saved form; back: B or an error |
| Approximate inverse → Reduction | admissible matrix; back: WDD matrices, each a Z-matrix if balanced, with the vertices each keeps, the lift with its scratch requirement, and the build's facts: clamp, components, null vector of each singular component. During a solve: right-hand sides; back: their lifts, then solutions |
| Approximate inverse → Cholesky | one WDD matrix with its grounding, whether it is balanced, its gauge, the configuration and the component's key; back: factorization with its scratch requirement. During a solve: right-hand sides; back: solutions |
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
- **The code owns the export list:** a list kept here would be a second copy of the part roots'
  re-exports, as signatures would be of the code. The snapshot and `parts.txt` are generated from
  the code and committed, so an addition such as `BlockFactor` cannot pass unseen: semver-checks
  allows additions.
- **Doc links not layer edges:** a rustdoc link names an item for the reader and compiles to
  nothing; counted, it would forbid a part's docs from pointing at the part that uses it.
- **`Sgdd` and `Complement` in the approximate inverse:** their `factor` returns B, which
  Reduction may not name; they hold Reduction's `Reduced`.
- **`WddCsr` in Storage:** Reduction makes it and Cholesky takes it, and neither uses the
  other; the kept vertices travel beside it as indices, so Storage holds no elimination.
- **The boundary at WDD, the cover in Cholesky:** the Gremban cover, its surplus encoding and
  signed elimination are alternative ways to factor one frustrated WDD matrix, chosen by
  Cholesky's configuration; the matrix, its scaling and its switching do not depend on the choice.
  Reduction states the fact, balanced or frustrated, and Cholesky decides how to factor it
  (#347).
- **`BuildError` in Reduction:** both its cases come from the reduction, the scaling and the
  components; Cholesky does not fail.
- **Visibility, then a test:** the compiler rejects any use of a part's private item, trait
  methods and macros included; the test adds what visibility cannot say, the layer order and the
  counts.
- **The snapshot before the parts:** the public surface exists on `main` today and changes in
  every 0.8.0 step, so P9a shows those changes; visibility per part needs the parts, P2.
- **Encodings per part:** the persistence format does not couple the approximate inverse to the
  internals of Reduction and Cholesky.


## Plan

From `main` to this architecture, beside Cholesky's steps R1–R6, G1 and F0–F2 in `cholesky.md`.
Refactors are bit-identical on the solve fixtures, the u32/u64 storage test and the wire-format
payloads, apart from the format version where stated, and are committed apart from behavior
changes.

0.8.0 is scope O2: the public surface of P5–P8 and P9a on `main`'s current modules, with D6, R6
and F1, and P1 as far as P5 takes it. P2's module moves, P9b, R1, R1b and R3 follow 0.8.0.

### Inside the parts

| Step | Part | Change | Check | After |
|---|---|---|---|---|
| D1 | Reduction | Named records where a fact crosses a function: a block's `BlockAdjacency`, `Relaxation`, a lone vertex, the null-vector certificate, Perron's best iterate; the singularity pair a record owning `SINGULAR_BAND`; `Congruent` formed with its row sums; Perron's `Shortfall` its own type, not the public report, settled, refuted or unsettled; one name per band; Perron's `tolerance` named `slack`. Reverted by #371 (was #353): `main`'s Reduction has none of these records | bit-identical | – |
| D2 | Storage | `Grounding`, `Balance`, `Sheets` beside the sparse storage, `flushed` given its band by Reduction's `ratio_band`; the scalar traits, compensated sums and range screens out of the crate root; one row-tally pass where `tally_abs` and `scaled_abs_sums` repeated it. `Grounded`, `Gauge` and the `WddCsr`, `Bipartite` names land in Storage with R1 and R1b. Reverted by #371 (was #357): on `main`, `storage.rs` holds the sparse storage alone, `Grounding` and `Balance` sit in `sgdd/block.rs`, the scalar traits at the crate root | bit-identical | – |
| D3 | Input | The representation an `Input` holds apart from the public `FactorWidth2`: a Gram form by layout, whole or bipartite, and by values, incidence or valued, and a CSR form, whole or a triangle; `Lowered`, `Columns` and the `n_left()`, `is_incidence()`, `surplus()` probes removed; `UpperCsrRef::lower` renamed. Reverted by #371 (was #360): `main`'s `Input` holds its forms as `input/csr.rs` and `input/width2` define them | bit-identical | – |
| D4 | Approximate inverse | The index width picked by Cholesky's `fits_u32`; the `store` function removed. Reverted by #371 (was #364): `main` picks the width by `fits_u32` in `lib.rs`, around `Sgdd`'s `Indexed` | bit-identical | R1 |
| D6 | Input, Reduction | Validation on construction: column range, in every constructor values finite and 0 or above the range floor, else `InputError::OutOfRange { parameter, entry }`, symmetry, nothing outside a triangle, magnitude of `n`, a bipartite form whose `n_left + n_right` overflows `VertexCountOverflow`; a Gram form's derived magnitude checked at build, not at construction; one `InputError` for `CsrError` and `ViewError`, an offset row pointer `RowPtrs { row: 0 }`; `Input` opaque, a borrowed `Copy` value, every view it wraps borrowed; `FactorWidth1::new(n, rows)` and `valued(n, rows, values)`; `BuildError` with `OutOfRange` and `TooLarge`, `OutOfRange`'s `vertex` the offending row, `TooLarge`'s its component's lowest vertex as on `main`, `Unscalable` folded into `OutOfRange`, a nonpositive diagonal an input error; `Sddm` admits an all-zero surplus; no `T = f64` default; `factorize` takes `impl Into<Input>`; `OwnedCsr`, `IndexKind`, the CSR accessors, the `&CsMatI` and `&SparseRowMat` conversions and the root re-exports of Input's types removed. After 0.8.0: `TriangleCsrRef::lower`; `ScalingOutcome::Certified` removed, refuted or unsettled as Perron's `Shortfall` records it | integration tests per error; stub, changelog; breaking; Python's error classes come with P8 | – |
| D7 | Reduction | A derived surplus below the range floor is dropped while within its row's rounding, $16\varepsilon\, d_i$ (`RATIO_BAND`); past it, it is a real ground the floor cannot represent: `OutOfRange` at its row, as is a balanced block whose grounds would all drop. One rule for derived surpluses; a stated surplus below the floor is rejected at input (D6). Done in #350 | regression tests: at scale $2^{-964}$ a frustrated triangle with a one-ulp surplus factors, as it does at scale 1; a real ground below the floor is `OutOfRange` at its row; solves that succeed today unchanged | – |
| D8 | Reduction | One refutation test for the scaling, with the rounding floor the Rayleigh check uses: Collatz and Rayleigh read one ceiling, `1 + max(slack, settle_band)`. Done in #355, reverted with D1 and ported onto `main`'s code by #371; no probed case differed | A/B of iterations and clamps | – |

### The public interface

| Step | Change | Check | After |
|---|---|---|---|
| P1 | Execution: `Threads`, `join`, `fork`, the task grain; no other part calls rayon. `main` has `Threads` and the apply's spread under it; P5 routes the build's rayon sites through `Threads` | every result bit-identical for every budget and in a nested pool | – |
| P2 | Modules moved to the tree below; after 0.8.0 | bit-identical | R1 |
| P3 | `ApproxInverse` and `Component` with `vertices`, `null_vector`, `clamp`, the clamp owned by its block alone; the per-block solver removed, so one width branch sits at the top, around the components, and a component's kind is apart from its width; `Indexed` removed; within migrated | breaking; format version bumped; within's tests | D4, P1 |
| P4 | `MatrixMut`, `ColumnMajor`, `Columns`, `scratch`, `apply`, `apply_with`, `ApplyError`, `LayoutError`; the budget spread over components by size. On `main` by #382, #395, #398, #400 | each column equals its solve alone, bit for bit, at every budget; `apply_with` allocates nothing at `Count(1)` | P3 |
| P5 | Build surface, in `src/factor/build.rs` until P2 renames it: `Sgdd` and `Complement` with `Threads` and `clamps`, an iterator of `Clamped` handles over the blocks' clamps, no clamp list and no `ClampId`; no other fact before factoring; the build's six direct rayon sites take the budget through `Threads`, part of P1 | interface tests; `Threads::All` keeps today's schedule, bit-identical | P3, D6, R6 |
| P6 | Persistence: the envelope `(Header, B)`, `Header`, `Scalar` and `HeaderMismatch` public, replacing `FACTOR_FORMAT_VERSION`; one version, none per part; format `0x4143_000E` | round trip; postcard bytes equal the flat `(version, scalar, B)`; version and scalar rejected by name; corrupt payloads rejected | P3 |
| P7 | Adapters: faer (row and column CSR, `MatrixMut`, `LinOp`, `BiLinOp`, `Precond`, `BiPrecond`, `From<Par>`), sprs, nalgebra, ndarray; the unimplemented-trait message | per-feature tests | P4, D6 |
| P8 | Python: `ApproxInverse` a `LinearOperator`, `threads`, `with_threads`, `apply` with `out`; `components`, each with its `vertices`, `null_vector` and `clamp`, no clamp list and no clamp `vertices`; `Config`'s `sampling` and `frustrated`; `InputError` and `BuildError` with their kinds; pickle through postcard, its header read as P6's `Header`; any scipy format through `tocsr()`, a CSC no longer read as its transpose; the checks Rust makes no longer repeated; `factorize_raw` removed | pytest, stubtest, changelog | P3–P6 |
| P9a | The public-surface snapshot: cargo-public-api over all features on a nightly pinned through rustup in `scripts/public-api.sh`, a CI job beside `semver`; every PR that changes the surface regenerates it. Lands first | CI | – |
| P9b | Visibility per part; the layer and export test, with syn, writing the generated `parts.txt`; after 0.8.0 | CI | P2 |

### Module tree

P2's target. Until P2, 0.8.0 included, the code keeps `main`'s modules: `factor`, `sgdd`,
`input`, `storage`, `threads`, `order`, `batch`.

```
src/lib.rs                     re-exports only
src/storage.rs, storage/       real.rs (Real, Index, sums, range screens), csr.rs (Csr, WddCsr, Bipartite), grounded.rs
src/execution.rs               Threads, join, fork, task grain
src/input.rs, input/           error.rs, csr.rs, triangle.rs, laplacian.rs, factor_width.rs, repr.rs, repr/{canonical,gram}.rs
src/reduction.rs, reduction/   components.rs, components/assembly.rs, components/assembly/cells.rs,
                               block.rs, switching.rs, classify.rs, classify/gram.rs, gram.rs, lift.rs,
                               scaling.rs, scaling/{perron,normalized,certificate}.rs
src/cholesky.rs, cholesky/     config.rs, streams.rs, representation.rs, representation/cover.rs, dense.rs,
                               elimination.rs, elimination/{graph,ordering}.rs, star.rs, fill.rs,
                               fill/{clique_tree,cdf}.rs, bipartite.rs, bipartite/block_elimination.rs
src/approximate_inverse.rs, approximate_inverse/
                               build.rs (factorize*, Sgdd, Complement), component.rs (lone vertices too),
                               apply.rs (MatrixMut, ColumnMajor, Columns), wire.rs
src/adapters/                  faer.rs, sprs.rs, nalgebra.rs, ndarray.rs
crates/approx-chol-py/src/     lib.rs, input.rs, build.rs, inverse.rs
```

- One file per input representation; the representation an `Input` holds apart from the public
  types.
- Reduction by stage: components, switching, scaling with its certificate, classification, the
  lift.
- `star` and `fill` beside `elimination` and `bipartite`, which both use them; everything that
  knows the cover under `representation`.
- Adapters under one folder: a root module named `faer` would shadow the crate in paths.

### Open questions

1. **Components of a Schur complement.** Cancellation can split a frustrated component's
   complement: $A$ connected, $H = I$ of two components. The contract promises $H$'s components,
   but they are found on $A$ before $H$ is formed. Find them after formation, which an exact
   formation allows and a sampled one does not, or promise $A$'s restricted to $K$.
