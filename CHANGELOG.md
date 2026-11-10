# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `Config::frustrated`, a `FrustratedRoute`, states how a frustrated block is factored approximately: `Cover(CoverSurplus)` factors its Gremban cover, whose surplus goes to a ground vertex (`GroundVertex`), onto an edge between each vertex's two copies (`SurplusEdges`), or onto those edges only inside the twist band of `CoverSurplus::Auto`, else to the ground vertex (`Auto`, the default, which every build took before). The dense arm factors a frustrated block as itself under any route. In Python, `Config(frustrated=...)`, keyword-only, takes `"cover"`, `"cover_ground_vertex"` or `"cover_surplus_edges"`, and the `frustrated` getter reports it.
- `FrustratedRoute::Signed` eliminates a frustrated general block as it is, on one sheet: fill is sampled on magnitudes and signed by its ends, and what opposite parallel entries cancel grounds both their ends, so the block never floats and its factor solves in place. A frustrated bipartite block's complement is formed signed, exact or sampled, each fill signed by its ends and what opposite fill cancels grounded, and eliminated so, where it was covered as under `Cover(Auto)` ([#433]). In Python, `Config(frustrated="signed")`.
- With `nalgebra`, a nalgebra-sparse 0.12 `&CsrMatrix<T>` converts by `TryFrom` into a `CsrRef<T, usize>` that borrows its arrays, a `&CscMatrix<T>` as the CSR of its transpose, and every nalgebra 0.35 `Matrix`, owned or a view of any strides, is a `MatrixMut` that `apply` stages through scratch, packed like every other block. The feature needs Rust 1.89, nalgebra's MSRV; the crate's stays 1.85. ([#427])
- With `ndarray`, ndarray 0.17's `ArrayRef2` and `ArrayRef1` are `MatrixMut` blocks at any strides: an `Array2`, an `Array1` or a view of either, passed as `&mut *a`, the one-dimensional one a single column; an `ArrayD` converts with `into_dimensionality::<Ix2>()`. Columns of row stride 1, a Fortran-order array's or a row slice of one, are solved where they are, as `ColumnMajor`'s are, others staged through scratch. The `sprs` feature enables `ndarray`, so sprs users' dense arrays are blocks; sprs 0.11 accepts ndarray 0.15 to 0.17, and the pin keeps one ndarray. ([#428])
- With `faer`, a `faer::sparse::SparseColMatRef` converts by `TryFrom` into a `CsrRef`, read as the CSR of its transpose, which a symmetric matrix equals, without a copy; a storage defect names its column as the row, and a matrix stored with gaps (`col_nnz` set) is `InputError::RowGap`; a non-square one is `NotSquare` with its own shape ([#426]).
- With `faer`, faer's `Mat`, `MatMut`, `Col` and `ColMut` are `MatrixMut` blocks at any strides, contiguous columns solved where they are; a typed faer view converts with `as_dyn_mut().as_dyn_stride_mut()`. A type that is no `MatrixMut` is told so, with the faer types and version named ([#426]).
- With `faer`, `ApproxInverse<T>` implements faer's `LinOp`, `BiLinOp`, `Precond` and `BiPrecond`, its own conjugate, transpose and adjoint, so faer's CG and LSMR take it, or `&` it, as a preconditioner: faer's `par` is the budget and every scratch query is `scratch(rhs_ncols, par.into())`; the in-place methods apply in place, without faer's temporary, and `apply` copies `rhs` into `out` first. faer's methods return nothing, so a block of other rows or a stack too small panics with `ApplyError`'s message ([#426]).
- The `faer` feature enables faer's `rayon`, so `Par::Rayon` reaches `Threads` and faer's operators without the caller enabling it ([#426]).
- With `faer`, `Threads: From<faer::Par>`: `Par::Seq` is `Count(1)`, `Par::Rayon(n)` is `Count(n)`, so faer's parallelism serves as a budget.
- `ApproxInverse::apply` and `apply_with` solve a block of right-hand sides, each column bit for bit as it solves alone, at a `Threads` budget: `apply` allocates what the block's layout takes, `apply_with` takes it from a `MemStack`, which `scratch(columns, threads)` bounds for every layout. A block is a `MatrixMut`: a slice or `Vec` of `T` is one column, one of `[T; K]` holds `K` interleaved columns, both solved in place; `ColumnMajor::new(data, rows)` reads `data` column-major, `rows` to a column, and `Columns::new(columns)` takes one `&mut [T]`, `Vec<T>` or other `AsMut<[T]>` per column, all of the first's rows; a malformed layout is a `LayoutError` where it is formed. These and any other `MatrixMut` are solved in interleaved packs of their columns copied into the scratch, so they take one pack's scratch whatever their columns. A block without columns is left as it is, whatever its rows.
- In Python, an `ApproxInverse` pickles, and so deep-copies, as its saved form `(Header, B)` encoded with postcard and nothing else, no object state, and unpickles to a factor that solves bit for bit as the original. A format version or scalar type other than this build's raises `ValueError` naming each that differs, found and expected; any other form that fails to decode or validate, or has trailing bytes, raises `ValueError` as invalid ([#431]).

### Changed (breaking)

- `Header { version, scalar }`, `Scalar { F32, F64 }` and `#[non_exhaustive] HeaderMismatch { found, expected }` replace `FACTOR_FORMAT_VERSION`: `Header::of::<T>()` is the header a persisted `ApproxInverse<T>` is written with and the only one it loads, format version `0x4143_000F` and `T`'s scalar type, and `expect::<T>()` checks a header read apart. A persisted `ApproxInverse` is the pair `(Header, B)`: postcard writes the flat `(version, scalar, B)`, a self-describing format `{"version", "scalar"}` with `"f32"` or `"f64"`. A load reads the version first and rejects any other, earlier payloads included, then the scalar type, before it reads `B`; the error carries `HeaderMismatch`'s message, which names what differs. A payload older than the header fails to decode in self-describing formats, as a type error. ([#433], [#425])
- The build's two stages take a `Threads` budget, and a Schur build has them too: `Sgdd::reduce(input, scaling, threads)` and `Sgdd::factor(config, threads)`, which takes `Config` by value, not `&Config`; `Complement::reduce_onto(input, kept, scaling, threads)` and `Complement::factor(config, threads)` are new. Before factoring, `Sgdd::clamps` and `Complement::clamps` yield a `Clamped` per clamped component of `A`, in the order `ApproxInverse::components` will list them: its `vertices`, numbered and ordered as the component's, a complement's on its kept side, and its `clamp`, the one the component's `Component::clamp` reads; nothing else is known before factoring. `factorize_with` and `factorize_complement` are the two stages at `Threads::All`, which builds to the same bits as before; every budget builds to those bits. In Rust, `Sgdd::reduce(input, scaling)?.factor(&config)` becomes `Sgdd::reduce(input, scaling, Threads::All)?.factor(config, Threads::All)`. ([#424])
- One `#[non_exhaustive] BuildError` with `TooLarge { vertex }` and `OutOfRange { vertex }` replaces `Error` and `ReduceError`: `factorize`, `factorize_with`, `factorize_complement` and `Sgdd::reduce` return it, since every input defect is an `InputError` at construction. `ReduceError::Unscalable` is folded into `OutOfRange`, which names the row whose computed value left the range, in the caller's numbering, where `Unscalable` named the block's first vertex. A relaxation rejects a NaN ratio, which it ignored before; none arises from a valid input, as each ratio divides a sum of positive terms by a finite positive `y`. `OutOfRange` reads "a value at vertex {vertex} is outside the representable range", in Rust and Python, where it read "an entry at vertex …". ([#422])

- `CsrRef::new` and `TriangleCsrRef::upper` check column range, symmetry or triangle membership, coalesced sums and diagonals at construction, so a defect is an `InputError` before the build. A diagonal that is negative, above `MAX/4`, or zero beside a nonzero edge is `InputError::Diagonal { row }`; an isolated zero row is still accepted. `ReduceError::NonpositiveDiagonal` is removed. Unsorted rows are sorted and coalesced to be checked and again when they factor, so the views and `Input` stay borrowed and `Copy`. Computed row sums and component-size limits remain build-time checks. ([#421])

- CSR entries, Gram loadings, column scales, Laplacian weights and stated SDDM surpluses must be finite and either zero or at least `MIN_POSITIVE/EPSILON` in magnitude, checked by their constructors as `InputError::OutOfRange { parameter, entry }`. Each CSR value is checked where it is stored, before repeated columns sum, so duplicates below the floor are rejected even where their sum is in range; a sum outside the range is `OutOfRange` at the column's first stored entry. A stated surplus below that floor is rejected even when the build previously discarded it as rounding; derived surpluses retain their rounding rule. Gram diagonal and cell magnitudes remain build-time checks. ([#421])

- `Sddm::new` accepts an all-zero surplus, which 0.7.0 rejected with `ViewError::ZeroSurplus`, and factors it bit for bit as its `Laplacian`; no `InputError` takes that variant's place. ([#423])
- One `#[non_exhaustive] InputError` replaces `CsrError` and `ViewError` for every input form, the `sprs` and `faer` conversions included, each defect at its position in the caller's terms, an argument named by the constructor's own `parameter`; `Error::InvalidCsr` is `Error::Input`, and `IndexKind` is removed. A length defect is `LengthMismatch { parameter, expected, found }`, `RowPtrsEndMismatchNnz` among them as `col_indices`'s; a first row pointer other than 0, an offset view's included, is `RowPtrs { row: 0 }`, a pointer above the next `RowPtrs { row }`; a row pointer negative or past `usize`, a column outside `0..n` or negative, and an index past the target type are `IndexOutOfRange { parameter, entry }` at that index's own position; `RepeatedVertex` is `SelfLoop { edge }` for a `Laplacian` and `RepeatedRow { column }` for a `FactorWidth2`; `InvalidWeight`, `InvalidScale` and `InvalidSurplus` are `OutOfRange { parameter, entry }` when not finite, `Negative { parameter, entry }` when negative; a CSR entry or repeated column's sum outside the representable range, `CsrError::OutOfRange`, is `OutOfRange { parameter: "values", entry }`, a sum at its column's first stored entry ([#421]); `BelowDiagonal` is `OutsideTriangle`, `ExpectedSquareMatrix` `NotSquare`. What is accepted does not change. Every Python input error's message is `InputError`'s `Display`, a non-square matrix's included. ([#419])
- Every input is built from its parts in the `input` module, size first: `CsrRef::new(n, row_ptrs, col_indices, values)` with `n: usize`; `UpperCsrRef` is `TriangleCsrRef`, built by `TriangleCsrRef::upper(n, row_ptrs, col_indices, values)`; `FactorWidth1::new(n, rows)` and `FactorWidth1::valued(n, rows, values)` replace `FactorWidth1::new(n, rows, values: Option<_>)`. `Input` is opaque, a borrowed `Copy` value built by `From` from any input type, and `factorize`, `factorize_with` and `Sgdd::reduce` take any `Into<Input>`, not `TryInto<Input>` ([#422]'s signature change, made here because no fallible conversion into `Input` remains). No input type defaults `T` to `f64`. Removed: `OwnedCsr`, `CsrRef::to_owned_u32`, `CsrRef`'s `row_ptrs`, `col_indices`, `values` and `n`, every `TryFrom` into `Input`, `From<Infallible> for Error`, and the conversions from `&sprs::CsMatI` and `&faer::sparse::SparseRowMat`: a sprs matrix converts through its `view()`, a faer one through its `as_ref()`, into a `CsrRef`. The crate root no longer re-exports `CsrRef`, `Input`, `InputError` and `Side`; they are in `input`. A dimension past `u32::MAX` is accepted: `CsrError::MatrixDimensionExceedsIndexType` is removed, with no `InputError` in its place; Python no longer rejects one, and an `n` its arrays do not fit is `LengthMismatch` on `row_ptrs`. ([#420])
- `Factor` is renamed `ApproxInverse`, in Rust and Python: it applies an approximate inverse, not a factor.
- `ScalingReport` and `BlockFactor::scaling_report` are removed. `Component::clamp` states how the reduction clamped a component's rows past rounding, if it did: its violation and a `ScalingOutcome`, `Certified`, `Refuted` or `Unsettled`; a Schur build reports its whole block's clamp on the kept vertices. `Component::grounding_unproven` states that a block floated or grounded on a scaling that did not settle.
- `ApproxInverse::solve`, `solve_into` and `solve_in_place` are removed: `apply` solves a right-hand side in place, `apply_with` through a `MemStack` that holds `scratch(columns, threads)`, a `StackReq`; `scratch_len` is removed. `SolveError` is `ApplyError`: `LengthMismatch` is `Rows`, `ScratchTooShort` is `Scratch`. `MemBuffer`, `MemStack` and `StackReq` are dyn-stack 0.13's, faer 0.24's, re-exported.
- `BlockFactor`, `Vertices`, `ApproxInverse::blocks` and `into_blocks` are removed. `ApproxInverse::components` borrows one `Component` per irreducible block: its `vertices`, in the order its factor numbers them, `floats` and `grounding_unproven`. A block is solved only through the whole `ApproxInverse`, so `solve_local`, a block's `scratch` and `splittable_work` are gone with it; a persisted list of `BlockFactor`s no longer decodes. `components` lists the coupled blocks by their component's lowest vertex in the input's numbering, a Gram's frustrated ones among its folded ones, then the isolated vertices.
- A persisted `ApproxInverse` stores every vertex once: each block holds its segment, a run of consecutive vertices or a length of one list, which also lists the isolated vertices, and an isolated vertex is its inverse pivot alone; its dimension and where each list and the isolated and dead vertices start follow from those, so no payload states them apart. A block's factor states a cover's two sheets only where it has them, so a dense factor on two sheets cannot be persisted.
- `Config` gains `frustrated`, so a literal naming every field must name it, and a JSON `Config` without it is rejected.
- `Config::split_merge` is replaced by `Config::sampling`, a `Sampling`: `CliqueTree { copies }` samples each edge as `copies: NonZeroU32` clique-tree copies, `1` standard AC, more AC2; `SchurMode::Approximate` takes a `sampling` in place of its `split_merge`. `None`, `Some(0)` and `Some(1)` are `copies: 1`, `Some(k)` is `copies: k`, to the same bits. A JSON `Config` stating `split_merge`, or `copies` of `0`, is rejected. `Config`, `Sampling` and `SchurMode` are `Eq` and `Hash`. In Python, `Config.split` is the copies, `1` by default, not `None`; a `split` of `0` or outside `u32` raises `ValueError`, not `OverflowError`. A Python `Config` compares and hashes by value.
- A row's surplus within its band is rounding, not a ground, whatever the input: a residue `d_i − Σ_j|a_ij|` within `(k_i + 3)ε·d_i` at degree `k_i`, less the block's clamp, which is rounding too; a stated ground, an `Sddm`'s surplus or a Gram's one-ended column, within the band of no summed term, `16ε·d_i` on an unscaled row, and only where its residue is positive. A balanced block that grounds keeps only its rows past their band, or every residue when none is past it; a frustrated one keeps none of its rounding. An `f32` CSR at weight spans from `1e14` no longer grounds on rounding, which cost a mixed-precision PCG 30–50% more iterations ([#311]); an `Sddm` whose grounds all sit within their band floats as its CSR does, where an `f32` one with grounds of `1` beside weights of `1e14` grounded and failed as a preconditioner even on a consistent system ([#343]). A derived ground below the range floor within its band is rounding, not `OutOfRange`. Factors change bits where a block has rows whose residue is rounding.
- `InnerParallelism` is replaced by `Threads`: `apply` takes `Threads::All`, every thread of the caller's rayon pool, or `Threads::Count(n)`, at most `n` threads at a time whatever the pool's size; `Count(1)` runs on the calling thread, as `Forbid` did. The solution's bits do not depend on it.

### Changed

- `apply` spreads its thread budget over the components by size: a component past the budget's share of the work is solved on the whole budget, one after another, the others at once, each on one thread, once their work outweighs waking the pool. On 1M vertices in 100 to 100,000 components an apply at 12 threads takes 4–8x less time, on 8 equal components 4–5x less; one component, or one thread, as before; the bits do not change. `scratch(columns, threads)` holds a region per thread, at most the pool's, of the most one of the components solved at once takes; `apply_with` needs one thread's scratch and solves as many components at once as its stack holds regions.
- Faster factorization of inputs with many small components or long paths, with fewer allocations per component. ([#312])
- Faster approximate factorization: 4-15% on grids, power-law graphs and fixed-effects inputs, with factors bit for bit as before unless a vertex's AC2 copy degree reaches `u32::MAX - 1`, which takes a `split_merge` near `2^30` or more: there its place in the minimum-degree order now follows its degree back down. ([#386])
- Faster approximate factorization: 3-12% on grids and banded inputs, with factors bit for bit as before except in `f32`, where a star of 21 to 32 entries now sums a neighbor's duplicates in the order they come, as every other star does. ([#388])
- A factor stores no ground, so a definite block of `n` vertices on one sheet, grounded or frustrated, solves in place: it takes no Cholesky slot, not `n + 1`, and a grounded cover takes `2n`, not `2n + 1`.

### Fixed

- `BipartiteFactorWidth2::new` and `scaled` reject side lengths whose sum overflows `usize` with `InputError::VertexCountOverflow { n_left, n_right }`, instead of saturating the matrix dimension. ([#421])
- An approximate step stores each neighbor's share `w/d` and `D⁻¹ = 1/d`, each rounded once. The heaviest neighbor's share was `1 − Σ` of the others, which cancelled to a relative error of 8e-6 at degree 10⁶ ([#306]).
- Decoding rejects an approximate factor with a negative share; such a payload decoded and solved another matrix.
- Decoding rejects a component's clamp whose violation is NaN, infinite, zero or negative; such a payload decoded and reported it as the clamp ([#418]).
- The exact arm factors a frustrated block as itself, not its two-sheet cover, so `Config::exact_max_dim` counts its vertices once; a frustrated block of `exact_max_dim / 2 < n ≤ exact_max_dim` vertices factored approximately before. ([#327])

## [0.7.0] - 2026-10-30

### Changed (breaking)

- `Config` gains `complement: SchurMode`. ([#182])
- `factorize_with` takes a `ScalingPolicy` after its `Config`. ([#267])
- `Backend` and `Config::backend` are removed: `Config::exact_max_dim` is the largest block dimension factored exactly, `0` none; a `Config` serialized by 0.6.0 is rejected from JSON and unsupported from postcard. ([#322])
- `ExactFailure`, `Fallback`, `DenseFailure`, `UnusablePivot` and `Factor::fallbacks` are removed, as the exact arm never fails a pivot. ([#182])
- `Error` is `InvalidCsr(CsrError)` or `Reduce(ReduceError)`; a value defect is `CsrError::{NotSymmetric, BelowDiagonal, OutOfRange}`, naming its row. ([#182])
- `CsrRef::new`, `CsrRef::to_owned_u32`, `OwnedCsr::try_from_usize` and `CsrRef`'s `sprs`/`faer` conversions return `CsrError`, not `Error`.
- A column outside `0..n` is reported by `factorize`, not `CsrRef::new`; `CsrError::InputConversionPanicked` is removed. ([#182])
- `factorize` takes any `TryInto<Input>`; a type converting only into `CsrRef` converts first. ([#182])
- `factorize` and `Factor`'s methods take `T: Real`, implemented only for `f32` and `f64`, and `factorize` an index `I: Sync`. ([#182])
- `low_level` is removed. ([#182])
- `CsrError::ExpectedCsrMatrixGotCsc` is removed: a CSC `sprs` matrix converts into `CsrRef` as its transpose, which a symmetric one equals; a CSC `faer` matrix converts as `matrix.as_ref().transpose()`.
- `Factor::n` is the input dimension; `Factor::original_n` is removed. ([#140])
- `Factor::solve_in_place` takes `x` of length `Factor::n` and caller scratch of at least `Factor::scratch_len`. ([#146])
- `Factor::solve` and `Factor::solve_into` reject a right-hand side or output whose length is not `Factor::n`. ([#146])
- `SolveError` is `LengthMismatch` or `ScratchTooShort`. ([#146])
- An entry below `MIN_POSITIVE / EPSILON` in magnitude is `CsrError::OutOfRange`; 0.6.0 accepted it, and solved scales below about 1e-307 wrongly without an error. ([#163])
- `Factor::solve_in_place` returns `solve`'s `A⁺b` on each floating block; 0.6.0 pinned an elimination-dependent vertex of each floating block to zero instead. ([#135])
- `FACTOR_FORMAT_VERSION` is `0x41430009`, and no other version decodes. ([#148], [#179])
- Python `Backend` is removed: `Config(exact_max_dim=...)` replaces it; `ExactFailure`, `DenseFailure`, `Fallback` and `Factor.fallbacks` are removed. ([#182])
- Python `Factor.n` is the matrix dimension, and `Factor.solve`, `solve_into` and `matvec` reject a `b` whose length is not `n`. ([#140], [#146])

### Changed

- `Config::exact_max_dim` bounds a block's dimension, not its solved variables: a floating block of `exact_max_dim + 1` vertices factors approximately. ([#182])
- `factorize` admits symmetric generalized diagonally dominant input: positive off-diagonals fold by a ±1 scaling or factor on a double cover, and a block short of dominance is scaled under `factorize_with`'s `ScalingPolicy`. ([#183])
- Python `Factor.solve`, `solve_into`, `matvec` and `rmatvec` take any 1-D integer or float array-like, and `matvec`/`rmatvec` scipy's `(n, 1)` column. ([#224])
- Python `Factor.solve_into` accepts `out` overlapping `b`, so `solve_into(v, v)` solves in place. ([#300])

### Added

- `input`: `Input`, taking `UpperCsrRef`, `Laplacian`, `Sddm`, `FactorWidth2` and `BipartiteFactorWidth2`, with `FactorWidth1` and `ViewError`. ([#183])
- `SchurMode`, `ScalingPolicy`, `ScalingReport` and `ReduceError`. ([#182])
- `Factor::blocks`, `Factor::into_blocks`, `BlockFactor` and `InnerParallelism`: a `BlockFactor` solves one block in place and persists on its own. ([#183])
- `Factor::scratch_len`. ([#146])
- A factor of more than `(u32::MAX - 1) / 2` vertices stores `u64` indices. ([#182])
- Python `factorize` warns when a block's deficit is clamped or its scaling does not settle. ([#181])
- Python `factorize_raw` takes arrays of any integer index dtype. ([#181])
- Python `Factor.workspace` returns a `Workspace`, which `Factor.solve_into` reuses across calls. ([#255])
- `Sgdd`: `factorize_with`'s two steps, `Sgdd::reduce` an `Input`, then `Sgdd::factor` it. ([#267])
- Python `ScalingPolicy`, which `factorize` and `factorize_raw` take as `scaling`. ([#301])
- `factorize_complement` factors a `BipartiteFactorWidth2`'s Schur complement onto a `Side`, formed under `Config::complement` as `factorize_with` forms it; the factor's vertex `k` is that side's `k`-th. ([#315])

### Fixed

- A `faer` matrix with unused slots inside a row is `CsrError::RowGap`; 0.6.0 read them as entries. ([#180])
- A grounded block whose weights span past `1/ε` ends its approximate elimination at the ground; it could solve inaccurately. ([#228])
- Python `factorize`, `factorize_raw`, `Factor.solve` (so `matvec`/`rmatvec`) and `Factor.solve_into` release the GIL for a matrix of more than 32768 stored entries, working only on their own copies meanwhile. ([#255])
- `Factor<f32>` keeps a high-degree pivot's `1/d` and fill weights; they drifted by `~mε` at degree `m`, up to a negative pivot. ([#275])
- AC2's degree queue takes memory linear in the input for any `split_merge`. ([#277])

## [0.6.0] - 2026-08-24

### Changed (breaking)

- A persisted `Factor` stores each neighbor's share of the original pivot, and
  `FACTOR_FORMAT_VERSION` is `0x41430004`. ([#131])
- A persisted column names the neighbor taking the remaining pivot rather than storing its
  share. ([#131])

### Changed

- Faster `Factor::solve` on dense factors, from composing elimination fractions at build
  time. ([#131])

## [0.5.0] - 2026-08-11

### Changed (breaking)

- A persisted `Factor` no longer carries `original_n`; it is derived from the blocks, and
  `FACTOR_FORMAT_VERSION` is `0x41430003`. ([#118])
- A deserialized `Factor` no longer rejects a tampered block anchor, which 0.4.0
  documented as rejected. ([#118])

### Changed

- A block routed to the exact dense backend no longer gets an elimination graph built for
  it. ([#84])
- `Error::NonFiniteValue` no longer takes precedence over `Error::Asymmetric`,
  `Error::PositiveOffDiagonal` and `Error::NotDiagonallyDominant`. ([#84])
- The Python bindings build against pyo3 and numpy 0.29. ([#89])

### Fixed

- The star sampler draws from `[0, 1)`; the previous mapping could return exactly `1.0`. ([#109])

## [0.4.0] - 2026-07-30

### Changed (breaking)

- Each connected component of a disconnected input is factored, grounded and projected
  on its own. ([#35], [#36])
- `solve`/`solve_into` cap the right-hand side at the original dimension;
  `solve_in_place` leaves one variable pinned per component.
- The serde representation of `Factor` is incompatible with 0.3.1, and a fixed seed can
  produce a different factor.
- `Config` gains a `backend` field, defaulting to exact dense Cholesky for a component
  of at most 24 solved variables. ([#83])
- `Error::Asymmetric`, `Error::NonFiniteValue`, `Error::NonFiniteRow` and
  `Error::NotDiagonallyDominant` reject non-symmetric, non-finite and non-dominant
  input, and duplicate entries are summed before the off-diagonal sign check.
- `Config::split_merge` of `Some(0)` or `Some(1)` selects standard AC.
- `low_level::CliqueTreeSampler` replaces `clique_tree_sample`/`clique_tree_sample_multi`:
  `seed` and `split_merge` move to `new`, `sample` takes the star index, and entries of
  equal weight order by neighbor index.
- `OwnedCsr::try_as_ref` is replaced by the infallible `as_csr_ref`, and `CsrRef`
  accessors return slices with the view's lifetime.
- A row surplus above `epsilon * scale * terms` is grounded, a deficit below it is
  `Error::NotDiagonallyDominant`. ([#78], [#85], [#91])
- A persisted `Factor` declares `format_version` and rejects any other version. ([#50])
- The sampler judges weights by sign rather than against an absolute floor, so a fixed
  seed's factor differs below `1e-14` (`f64`) / `1e-6` (`f32`). ([#92])

### Removed

- `Error::InvalidConfig` and `ConfigError`.

### Added

- `Config::backend`: `Backend::ExactBelow { max_dim, on_failure }` or
  `Backend::Approximate`, with `ExactFailure` choosing fallback or
  `Error::DenseFactorizationFailed`. ([#83])
- `Factor::fallbacks`, `Fallback`, `UnusablePivot` and `DenseFailure`. ([#83])
- Python `Backend`, `ExactFailure`, `DenseFailure`, `Fallback`, `Config(backend=...)`,
  `Factor.fallbacks`, and a `RuntimeWarning` per fallback. ([#83])
- `From<&OwnedCsr> for CsrRef`. ([#41])
- One `cp39-abi3` wheel per platform, plus `cp314t` free-threaded wheels. ([#61])
- `requires-python` drops to 3.9 and the `scipy` floor rises to 1.12. ([#61])
- `FACTOR_FORMAT_VERSION`. ([#50])
- An MSRV of Rust 1.85. ([#62])

### Fixed

- An invalid `Factor` — tampered anchor, mismatched `original_n`, non-finite or
  out-of-range values — is rejected at deserialize time. ([#37], [#80])
- `CliqueTreeSampler` no longer panics or emits non-finite fill on degenerate
  weights. ([#38])
- `u32` nonzero/edge overflow panics instead of truncating the factor. ([#39])
- A sub-unit-scale strictly-dominant SDDM is augmented, and elimination keeps the scale
  of an underflowing diagonal or per-copy share. ([#36])
- Approximate elimination keeps a small pivot's reciprocal wherever it is
  representable. ([#75])
- The approximate solve zeroes the uneliminated vertex between passes, so a small
  solution is no longer annihilated. ([#93])
- Each block draws from its own sampler stream, so a fixed seed factors it the same way
  under either backend. ([#82])
- A large diagonal no longer overflows the row-scale computation.

## [0.3.1] - 2026-07-10

### Fixed

- Reject positive off-diagonals (`Error::PositiveOffDiagonal`) at ingestion
  instead of silently dropping them, which had produced a wrong factor on
  non-SDDM input.

## [0.3.0] - 2026-06-15

### Changed

- Faster factorization via batched minimum-degree priority-queue updates. The
  randomized factor produced for a given seed may differ from 0.2.0;
  correctness is unaffected.

## [0.2.0] - 2026-05-26

### Changed (breaking)

- `low_level::clique_tree_sample` no longer takes a `pivot_diag` parameter.
- `CsrError` variants consolidated (20 → 12). `RowPtr*ExceedsU32`, `ColIndex*ExceedsU32`, `*ExceedsTargetIndexType` collapse into `IndexExceedsIndexType { kind: IndexKind }`. `NExceedsU32`, `NExceedsTargetIndexType`, `MatrixDimensionExceedsU32` collapse into `MatrixDimensionExceedsIndexType { n }`. `RowPtrNotRepresentableAsUsize` and `ColIndexNotRepresentableAsUsize` collapse into `IndexNotRepresentableAsUsize { kind, position }`. `RowIndexOutOfBounds` removed.
- `Error`, `ConfigError`, `CsrError`, `SolveError` are now `#[non_exhaustive]`; external `match` sites must add a wildcard arm.

### Added

- `IndexKind { RowPtr, ColIndex }` for disambiguating which CSR array an index error refers to.

### Removed

- `low_level::CdfSampler` — the `WeightedSampler` trait it implemented was crate-private, so external code could not wire it into anything.
- `low_level::EliminationSequence`, `low_level::EliminationStep` — factor-internal types with no external callers.
- `Factor::solve_into_with_projection` — folded into `solve_into` (always projects). For non-projecting solves, copy the RHS into the work buffer and call `solve_in_place`.
- `CsrRef::try_from_sprs`, `try_from_sprs_view`, `try_from_faer`, `try_from_faer_view` inherent methods — use the `TryFrom` impls instead (same conversions, same errors).
- `CsrRef::try_row` and `CsrRef::debug_validate` are no longer part of the public API.

### Fixed

- AC factorization no longer panics on marginally-SDD Laplacian inputs.

## [0.1.0] - 2026-03-10

Initial release of `approx-chol`, providing approximate Cholesky factorization
for graph Laplacians in Rust with Python bindings.

### Added

- Core approximate Cholesky factorization algorithm for SDD/SDDM matrices
- Fallible APIs with structured error types throughout
- CSR sparse matrix representation with checked accessors
- Triangular solve (`solve` and `solve_into`) for factored systems
- Clique sampling and star/clique-tree internals
- Serialization support via optional `serde` feature
- Python bindings (via PyO3/maturin) exposing factorization and solve
- Input validation and borrow-safe `solve_into` on the Python side
- Proptest suites for CSR construction and factorization invariants
- Preconditioner effectiveness tests
- Criterion benchmarks for factorization, sampling, solve, and CSR conversion
- Multi-platform CI with coverage reporting
- Dual MIT license for Rust crate and Python package

[Unreleased]: https://github.com/aai-institute/approx-chol/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/aai-institute/approx-chol/releases/tag/v0.7.0
[0.6.0]: https://github.com/aai-institute/approx-chol/releases/tag/v0.6.0
[0.5.0]: https://github.com/aai-institute/approx-chol/releases/tag/v0.5.0
[0.4.0]: https://github.com/aai-institute/approx-chol/releases/tag/v0.4.0
[0.3.1]: https://github.com/aai-institute/approx-chol/releases/tag/v0.3.1
[0.3.0]: https://github.com/aai-institute/approx-chol/releases/tag/v0.3.0
[0.2.0]: https://github.com/aai-institute/approx-chol/releases/tag/v0.2.0
[0.1.0]: https://github.com/aai-institute/approx-chol/releases/tag/v0.1.0

[#35]: https://github.com/aai-institute/approx-chol/issues/35
[#36]: https://github.com/aai-institute/approx-chol/issues/36
[#37]: https://github.com/aai-institute/approx-chol/issues/37
[#38]: https://github.com/aai-institute/approx-chol/issues/38
[#39]: https://github.com/aai-institute/approx-chol/issues/39
[#41]: https://github.com/aai-institute/approx-chol/issues/41
[#50]: https://github.com/aai-institute/approx-chol/issues/50
[#61]: https://github.com/aai-institute/approx-chol/issues/61
[#62]: https://github.com/aai-institute/approx-chol/issues/62
[#75]: https://github.com/aai-institute/approx-chol/issues/75
[#78]: https://github.com/aai-institute/approx-chol/issues/78
[#80]: https://github.com/aai-institute/approx-chol/issues/80
[#82]: https://github.com/aai-institute/approx-chol/issues/82
[#83]: https://github.com/aai-institute/approx-chol/issues/83
[#84]: https://github.com/aai-institute/approx-chol/issues/84
[#85]: https://github.com/aai-institute/approx-chol/issues/85
[#89]: https://github.com/aai-institute/approx-chol/issues/89
[#91]: https://github.com/aai-institute/approx-chol/issues/91
[#92]: https://github.com/aai-institute/approx-chol/issues/92
[#93]: https://github.com/aai-institute/approx-chol/issues/93
[#109]: https://github.com/aai-institute/approx-chol/issues/109
[#118]: https://github.com/aai-institute/approx-chol/issues/118
[#131]: https://github.com/aai-institute/approx-chol/pull/131
[#135]: https://github.com/aai-institute/approx-chol/issues/135
[#140]: https://github.com/aai-institute/approx-chol/issues/140
[#146]: https://github.com/aai-institute/approx-chol/issues/146
[#163]: https://github.com/aai-institute/approx-chol/issues/163
[#179]: https://github.com/aai-institute/approx-chol/issues/179
[#180]: https://github.com/aai-institute/approx-chol/issues/180
[#181]: https://github.com/aai-institute/approx-chol/issues/181
[#224]: https://github.com/aai-institute/approx-chol/pull/224
[#183]: https://github.com/aai-institute/approx-chol/issues/183
[#182]: https://github.com/aai-institute/approx-chol/issues/182
[#148]: https://github.com/aai-institute/approx-chol/issues/148
[#228]: https://github.com/aai-institute/approx-chol/pull/228
[#255]: https://github.com/aai-institute/approx-chol/issues/255
[#267]: https://github.com/aai-institute/approx-chol/issues/267
[#275]: https://github.com/aai-institute/approx-chol/issues/275
[#277]: https://github.com/aai-institute/approx-chol/issues/277
[#300]: https://github.com/aai-institute/approx-chol/issues/300
[#301]: https://github.com/aai-institute/approx-chol/issues/301
[#306]: https://github.com/aai-institute/approx-chol/pull/306
[#311]: https://github.com/aai-institute/approx-chol/issues/311
[#312]: https://github.com/aai-institute/approx-chol/issues/312
[#315]: https://github.com/aai-institute/approx-chol/pull/315
[#322]: https://github.com/aai-institute/approx-chol/pull/322
[#327]: https://github.com/aai-institute/approx-chol/issues/327
[#343]: https://github.com/aai-institute/approx-chol/issues/343
[#386]: https://github.com/aai-institute/approx-chol/pull/386
[#388]: https://github.com/aai-institute/approx-chol/pull/388
[#418]: https://github.com/aai-institute/approx-chol/issues/418
[#419]: https://github.com/aai-institute/approx-chol/issues/419
[#420]: https://github.com/aai-institute/approx-chol/issues/420
[#421]: https://github.com/aai-institute/approx-chol/issues/421
[#422]: https://github.com/aai-institute/approx-chol/issues/422
[#423]: https://github.com/aai-institute/approx-chol/issues/423
[#424]: https://github.com/aai-institute/approx-chol/issues/424
[#425]: https://github.com/aai-institute/approx-chol/issues/425
[#426]: https://github.com/aai-institute/approx-chol/issues/426
[#427]: https://github.com/aai-institute/approx-chol/issues/427
[#428]: https://github.com/aai-institute/approx-chol/issues/428
[#431]: https://github.com/aai-institute/approx-chol/issues/431
[#433]: https://github.com/aai-institute/approx-chol/issues/433
