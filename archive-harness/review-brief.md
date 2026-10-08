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
| surplus len n, finite, >= 0, zero or >= floor | ground edges, diagonals | `check_surplus` (Sddm::new); CSR derives surplus from row balance |
| degree + surplus finite per vertex | exact diagonal, approx pivot | `Incidence::finish` |
| connected components | builder, both arms | `Incidence` (union-find fused into the edge walk) |
| per-component grounded flag, finite ground total | Component gauge, Block, approx ground degree | `Grounding::new` |
| n + 1 fits u32 | ground slot | array shape check / CSR |

## Draft (signatures only)

```rust
pub struct Laplacian<T = f64> {
    row_ptrs: Vec<u32>,
    neighbors: Vec<u32>,
    weights: Vec<T>,
    components: Components,
}

pub struct Sddm<T = f64> {
    laplacian: Laplacian<T>,
    grounding: Option<Grounding<T>>,   // None when no component holds surplus
}

enum Components {
    Connected,
    // AUTHOR REVISION: the agreed design also stored `local: Vec<u32>` (vertex -> index in its block).
    // Dropped because it is derivable from order/ends in O(n) and would live as long as the Laplacian
    // (the rejected-degrees pattern); factorize keeps one transient `local_of` scratch as today.
    Split { order: Vec<u32>, ends: Vec<u32> },
}

struct Grounding<T> {
    surplus: Vec<T>,
    grounded: Vec<bool>,   // per component
}

impl<T: Real> Laplacian<T> {
    pub fn new(row_ptrs: Vec<u32>, neighbors: Vec<u32>, weights: Vec<T>) -> Result<Self, LaplacianError>;
}
impl<T: Real> Sddm<T> {
    pub fn new(row_ptrs: Vec<u32>, neighbors: Vec<u32>, weights: Vec<T>, surplus: Vec<T>) -> Result<Self, SddmError>;
}
impl<T> From<Laplacian<T>> for Sddm<T> {}                          // move, grounding = None
impl<T: Real, I: PrimInt> TryFrom<CsrRef<'_, T, I>> for Sddm<T> {}  // builds the arrays
pub fn factorize<T: Real>(sddm: impl Into<Sddm<T>>) -> Factor<T>;
pub fn factorize_with<T: Real>(sddm: impl Into<Sddm<T>>, config: Config) -> Result<Factor<T>, UnusablePivot>;

// private to module `sddm` (Laplacian moves into sddm.rs so only route bodies can write the literal)
struct Incidence<T> { degrees: Vec<T>, sets: DisjointSets }
impl<T: Real> Incidence<T> {
    fn add(&mut self, row: usize, col: usize, weight: T);
    fn finish(self, surplus: Option<&[T]>) -> Result<Components, DiagonalOverflow>;  // degrees never leave
}
impl<T: Real> Grounding<T> {
    fn new(components: &Components, surplus: Vec<T>) -> Result<Option<Self>, GroundOverflow>;
}
fn traverse<T: Real>(row_ptrs: &[u32], neighbors: &[u32], weights: &[T]) -> Result<Incidence<T>, LaplacianError>;
fn check_surplus<T: Real>(surplus: &[T], n: usize) -> Result<(), SddmError>;
fn check_weight<T: Real>(weight: T) -> Result<T, WeightDefect>;

// factorization side
// View = &Laplacian + today's BlockVertices::{Whole(n), Part { vertices, local_of }}
pub(crate) struct Component<'a, T> { view: View<'a, T>, gauge: Gauge<'a, T> }
pub(crate) enum Gauge<'a, T> { Floating, Grounded(&'a [T]) }
pub(crate) struct Block<T> { cholesky: Cholesky<T>, gauge: BlockGauge }  // was enum Block { Grounded(Cholesky), Floating(Cholesky) }
```

Routes:
| Route | Edge walk | After, O(n) |
|---|---|---|
| `Laplacian::new` | `traverse` | `finish(None)` |
| `Sddm::new` | `traverse` | `check_surplus`, `finish(Some)`, `Grounding::new` |
| CSR | own walk (mirror pairing, row balance, sign) building the arrays, `check_weight` + `add` | `finish(Some)`, `Grounding::new` |
| `Laplacian` -> `Sddm` | none | none |

Deleted vs HEAD: `UpperRows`, `Summed` (sddm/laplacian.rs), internal `Sddm`, `graph::component::components()` and
its second edge walk, `is_upper_adjacency`, validate.rs's sign/floor checks, the
matrix-wide surplus total.
