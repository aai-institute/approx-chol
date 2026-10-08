# Design review brief: approx-chol `Laplacian` / `Sddm` input types

Repo: /Users/kristof/Projects/approx-chol (branch feat/133-sddm-input, HEAD 7421a8d), core crate
`crates/approx-chol`. Downstream consumer: /Users/kristof/Projects/within (`crates/within/src/block_elim/factor.rs`).
Anti-patterns to check against: /Users/kristof/.claude/skills/type-design/anti-patterns.md

## Goal

One verified internal representation (`Sddm`) that every public route lands in, so `factorize` trusts its
input; a caller who builds a `Laplacian`/`Sddm` pays no ingestion inside approx-chol, and only generic CSR
runs checks + component detection, inside its conversion, without doing any check twice.

## Non-negotiables

Performance and memory (all measured, one-binary interleaved A/B with control arm):
- CSR route must not regress vs HEAD. Summing degrees *during* the CSR walk beats a separate pass over the
  upper Laplacian (separate pass: +5% complete_n512, +12% complete_n20).
- Counting adjacency *capacities* (integer row lengths for the approximate graph) inside the union-find walk
  cost exact builds +9-11% (complete_n20); capacities stay in the approximate arm's graph build.
- Per-component copies cost up to 2.4x on many tiny components; components are zero-copy views.
- Dispatching Whole/Part per entry inside the hot loop cost the exact arm ~+3% (spills `local_of`); match once outside.
- No persistent per-vertex/per-edge array that is derivable from what is stored: storing weighted degrees in
  `Laplacian` was rejected as a memory regression (+29% of the Laplacian at degree 4) to save a one-time O(m) re-sum.
- `unreachable!` error mappings are not acceptable; neither is a regression traded for simpler types.

Domain / behaviour:
- Every grounded component gets its own ground (one shared ground measured 5-10x worse AC error, NaN on overflow).
- Floor rule: every stored magnitude (edge weight, positive surplus) >= `MIN_POSITIVE / EPSILON`, else rejected.
- Every diagonal (degree + surplus) finite; each ground's total surplus finite (it is the ground's degree in the
  approximate arm). n+1 must fit u32 (a grounded component's ground slot).
- Which defect wins on multi-defect input is not a contract.
- Public: `Laplacian`, `Sddm` public with private fields; `factorize(impl Into<Sddm>)`; `CsrRef` stays as the
  generic route; each constructor has its own error type. Issue text (#133/#137/#142/#143) is NOT a constraint.

## Consumers

| Consumer | Distinct actions |
|---|---|
| within (external, `block_elim/factor.rs`) | holds sorted upper edges `(lo, hi, w)` of a **connected floating Laplacian** (appends its own ground vertex today); mirrors to symmetric CSR only for approx-chol. Two regimes: dense small-n (n~400-4000, 390 nnz/row) and sparse huge-n. ~10 solves per factorize. Would build a `Laplacian` (or `Sddm` with surplus instead of its own ground) from upper arrays directly. |
| within retry (solver.rs:197-214) | after an exact factor fails with UnusablePivot, re-factors the same complement (n <= 24) approximately |
| Python bindings / CsrRef callers (sprs, faer features, tests) | hand a symmetric CSR of any integer index type, any SDDM, possibly disconnected |
| `builder::factor_blocks` | n; connected or not; iterate components; the component order becomes the factor's permutation |
| `BlockFactorizer::factor` / `cholesky` | per component: first vertex (sampler restart), eliminated count (backend routing), gauge -> `Block` variant |
| exact arm (`exact::factor`, `Component::entries`) | per component: entries among eliminated vertices in local numbering, diagonal = degree + surplus, floating pins its last vertex; maps local pivot back to global for errors |
| approximate arm (`Component::graph`) | per component: row lengths for capacities, edges in local numbering, ground edges (vertex, surplus) and a ground slot iff grounded |
| `Block::solve` | gauge decides embedding: grounded `[b; -sum b]` then shift by ground; floating project / apply / project |

## Guarantees

| Guarantee | Needed by | Established once, at |
|---|---|---|
| strict upper adjacency: row_ptrs from 0, non-decreasing, ends at len; cols sorted, > row, < n | every reader | `traverse` (array routes); CSR walk builds arrays that satisfy it by construction |
| weight finite and >= floor | both arms | `check_weight` (all three routes) |
| surplus len n, >= 0 (rejects NaN), zero or >= floor | ground edges, diagonals | `check_surplus` (Sddm::new); CSR: `RowBalance::of`, whose `Surplus(T)` is >= floor by construction |
| degree + surplus finite per vertex | exact diagonal, approx pivot | array routes: `Incidence::finish`; CSR: implied by `RowBalance` (scale = d + sum|own| finite and d >= sum|own| - noise, so degree <= MAX/2 (1+8eps)) |
| connected components (n = 0 has zero) | builder, both arms | union-find fused into each route's edge walk, one root per row (`DisjointSets`; inside `Incidence` on array routes) |
| finite ground total per component; surplus None iff every total is zero | approx ground degree; gauge | `grounding` (both Sddm routes) |
| per-component gauge | Component, Block | derived when the view is built (`Gauge::of`, any-scan as HEAD component.rs:49-54), not stored |
| n + 1 fits u32 | ground slot | array shape check / CSR |

## Draft, round 3 (signatures only)

```rust
#[derive(Clone)]
pub struct Laplacian<T = f64> {
    row_ptrs: Vec<u32>,
    neighbors: Vec<u32>,
    weights: Vec<T>,
    components: Components,
}

#[derive(Clone)]
pub struct Sddm<T = f64> {
    laplacian: Laplacian<T>,
    surplus: Option<Vec<T>>,   // Some iff some component's surplus total is positive
}

#[derive(Clone)]
enum Components {
    Connected,                                  // exactly one component, n >= 1
    Split { order: Vec<u32>, ends: Vec<u32> },  // any other count, including zero blocks for n = 0
}

impl<T: Real> Laplacian<T> {
    pub fn new(row_ptrs: Vec<u32>, neighbors: Vec<u32>, weights: Vec<T>) -> Result<Self, LaplacianError>;
}
impl<T: Real> Sddm<T> {
    pub fn new(row_ptrs: Vec<u32>, neighbors: Vec<u32>, weights: Vec<T>, surplus: Vec<T>) -> Result<Self, SddmError>;
}
impl<T> From<Laplacian<T>> for Sddm<T> {}                          // move, surplus = None
impl<T: Real, I: PrimInt> TryFrom<CsrRef<'_, T, I>> for Sddm<T> { type Error = Error; }
pub fn factorize<T: Real>(sddm: impl Into<Sddm<T>>) -> Factor<T>;
pub fn factorize_with<T: Real>(sddm: impl Into<Sddm<T>>, config: Config) -> Result<Factor<T>, UnusablePivot>;
// Deleted: low_level::Builder (low_level keeps CliqueTreeSampler). within migrates to
// factorize_with(Sddm::try_from(csr)?, config) and matches UnusablePivot.

// Public errors
pub enum Error { /* HEAD's CSR variants */ }  // minus DenseFactorizationFailed; SurplusOverflow -> GroundOverflow { vertex }
pub enum AdjacencyError { /* shape: row_ptrs empty / not from 0 / decreasing / end != len; col not > row, unsorted, >= n; n + 1 > u32 */
                          Weight { edge: (usize, usize), defect: WeightDefect } }
pub enum WeightDefect { NonFinite, NotPositive, BelowFloor }
pub enum LaplacianError { Adjacency(AdjacencyError), DegreeNotFinite { vertex: usize } }
pub enum SurplusDefect { Negative /* includes NaN: !(s >= 0) */, BelowFloor }
pub enum SddmError { Adjacency(AdjacencyError), SurplusLength { len: usize, n: usize },
                     Surplus { vertex: usize, defect: SurplusDefect },
                     DiagonalNotFinite { vertex: usize }, GroundOverflow { vertex: usize } }

// private to module `sddm`
struct Incidence<T> { degrees: Vec<T>, sets: DisjointSets }   // array routes only
impl<T: Real> Incidence<T> {
    fn row(&mut self, row: usize) -> RowIncidence<'_, T>;       // resolves the row's root once
    fn finish(self, surplus: Option<&[T]>) -> Result<Components, NotFinite>;  // NotFinite { vertex }; degrees never leave
}
struct RowIncidence<'a, T> { incidence: &'a mut Incidence<T>, row: u32, root: u32 }
impl<T: Real> RowIncidence<'_, T> { fn add(&mut self, col: usize, weight: T); }
fn traverse<T: Real>(row_ptrs: &[u32], neighbors: &[u32], weights: &[T]) -> Result<Incidence<T>, AdjacencyError>;
fn check_surplus<T: Real>(surplus: &[T], n: usize) -> Result<(), SddmError>;   // length, !(s >= 0), floor
fn check_weight<T: Real>(weight: T) -> Result<T, WeightDefect>;
fn grounding<T: Real>(components: &Components, surplus: Vec<T>) -> Result<Option<Vec<T>>, GroundOverflow>;
// DisjointSets (existing, graph/component/sets.rs): find + union_resolved per-row-root API kept;
// layout() -> Option<BlockLayout> becomes components(self) -> Components.

// CSR route (sddm/csr): one walk; per row `let mut root = sets.find(row)`, per kept edge `check_weight(-upper)`
// (after the stored-zero skip, before claiming the mirror) then `root = sets.union_resolved(root, col)`; NO degree sums.
// validate.rs:76-80, :118-123 deleted; claim's mirror finiteness test (:42) stays.
enum RowBalance<T> { NonFinite, Deficit, Negligible, BelowFloor, Surplus(T) }   // Surplus(T) >= floor

// factorization side
// View = &Laplacian + today's BlockVertices::{Whole(n), Part { vertices, local_of }}; local_of is one transient
// scratch buffer in factorize, refilled per component (HEAD builder.rs:77).
pub(crate) struct Component<'a, T> { view: View<'a, T>, gauge: Gauge<'a, T> }
pub(crate) enum Gauge<'a, T> { Floating, Grounded(&'a [T]) }
impl<'a, T: Real> Gauge<'a, T> { fn of(view: &View<'a, T>, surplus: Option<&'a [T]>) -> Self; }  // any-scan, HEAD component.rs:49-54
pub(crate) struct Block<T> { cholesky: Cholesky<T>, gauge: BlockGauge }
```

Routes:
| Route | Edge walk | After, O(n) |
|---|---|---|
| `Laplacian::new` | `traverse`: per row `incidence.row(r)`, per edge `check_weight` + `add` | `finish(None)` |
| `Sddm::new` | `traverse` | `check_surplus`, `finish(Some)`, `grounding` |
| CSR | mirror pairing + row sums, per edge `check_weight` + union, builds the arrays | `RowBalance` per row -> surplus, `sets.components()`, `grounding` |
| `Laplacian` -> `Sddm` | none | none |
| `factorize` | none | n = 0 is a zero-block `Split` -> `from_blocks(None, [], [])` |

Deleted vs HEAD: `UpperRows`, `Summed` and their degree sums, `Sddm::with_surplus`, `first_non_finite_diagonal`,
`graph::component::components()` and its second edge walk, `is_upper_adjacency`, validate.rs:76-80/:118-123, the
matrix-wide surplus total, builder.rs:60-62, `Factor::empty`, `low_level::Builder`, `Error::DenseFactorizationFailed`.

## Earlier findings and how they were resolved (attack the fixes too)
Round 1:
1. CSR surplus had no floor -> `RowBalance::BelowFloor`.
2. stored `grounded: Vec<bool>` was derivable -> deleted; gauge derived per view.
3. n = 0 classified Connected -> zero-block `Split`; guard and `Factor::empty` deleted.
4. per-edge `add(row, col)` hid the per-row root -> `Incidence::row` -> `RowIncidence::add`.
5. weight checks ran twice on CSR -> one `check_weight`; `check_surplus` no finiteness test.
Round 2:
1. CSR could store `Some(all zeros)` and solve a floating input as grounded -> `grounding` returns the Option, both Sddm routes call it.
2. CSR summed degrees only to re-prove what RowBalance proves -> CSR uses `DisjointSets` directly, no degrees
   (the reviewer's `Linking { sets }` wrapper was rejected: it wraps `DisjointSets` without adding anything).
3. `traverse` returned `LaplacianError`, misattributing on `Sddm::new` -> `AdjacencyError`, per-constructor top-level errors.
4. `Builder` / `Error` unspecified -> `Builder` deleted, `DenseFactorizationFailed` removed from `Error`, `Clone` for within's retry.
