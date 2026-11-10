# Cholesky

How the Cholesky part factors one WDD matrix, and where each choice of the configuration lives.
The mathematics is steps 5–8 of `../algorithm.md`; the part's boundary is in `architecture.md`.
Planned: on `main` Reduction (`sgdd`) decides the sheets and checks the cover's ground; Cholesky
(`factor/sddm.rs`) builds the cover's rows from them.

## Decision

Cholesky receives one WDD matrix with its grounding, its balance and its gauge, the configuration,
and the component's key. Seven choices shape the factorization; each has one owner.

| Choice | Values | Owner | Decided |
|---|---|---|---|
| Arm | dense, approximate | `Config::exact_max_dim` | per component, by size; a bipartite one by its kept side |
| How a frustrated matrix is factored | Gremban cover, signed elimination | `FrustratedRoute` | per build, applied per component; the dense arm factors it as it is |
| Where a cover puts its surplus | ground vertex, surplus edges, auto | `CoverSurplus` | per build; auto per component |
| How a bipartite form's complement is formed | dense, exact, sampled | `SchurMode`, with the arm | per component; a dense kept side forms it dense |
| Clique sample | clique tree, Schur tree | `Sampling`, a `FillRule` | per build |
| Fill-in level | copies (clique tree), split (Schur tree) | the `Sampling` variant | per build |
| Sign of the eliminated entries | unsigned, signed | a `Sign` type | follows from balance and route |

- **One matrix type, typed by sign.** The matrix factored is a WDD matrix with its grounding,
  `Wdd<Unsigned>` for a Z-matrix and `Wdd<Signed>` for one with positive entries. The dense arm
  factors either as it is; its kernel is chosen by the sign type, not by a flag.
- **A representation is a matrix and its slot map, built together.** `Represented` pairs the
  matrix the elimination sees with `Layout`, which maps the elimination's slots to the component's
  vertices, chooses the pinned slot, and owns the lift and the descent:

  | Layout | Slots | Pinned | Built from |
  |---|---|---|---|
  | `Floating` | $n$ | the gauge's peak | a balanced matrix without surplus |
  | `Grounded` | $n + 1$, $g$ appended | $g$ | a balanced matrix with surplus; any matrix on the signed route |
  | `Cover { grounded }` | $2n$, $+1$ if grounded | $g$, else the peak's first copy | a frustrated matrix on the cover route |

- **The ground is never a pivot.** $g$ is pinned in every layout that has it, so its diagonal
  $\sum_i s_i$ is never formed and no range check guards it; each edge to $g$ is bounded by its
  other end, as every edge is. Signed elimination needs this anyway: cancellation adds edges to
  $g$ during the elimination. Each cancellation adds $2\min(|a|,|b|)$ to both ends without
  subtracting, so a frustrated matrix on the signed route stays grounded however small its
  frustration: its last pivot follows the frustration down to the input's representable range,
  and no near-balanced block is pinned (#435). On `main` since step G1 (#351).
- **Three constructors of `Represented`:** `as_is`, the matrix as it is, for the dense arm and the
  signed route; `sampled`, which builds the cover when the route asks for it; and
  `from_cover_rows`, for a bipartite complement formed on the cover. The cover's construction, its
  surplus edges and the auto rule live in one private function both cover constructors call.
  Which slot a neighbor lands on, sheet and crossing, is one function, used by the general cover
  and by both bipartite formations.
- **Cholesky receives facts, never Reduction's internals.** It takes the matrix with its
  grounding, the balance and the gauge, not the reduced record: the vertex of the factored matrix
  where a pin loses least and whether the scales span past $1/\sqrt\varepsilon$; without scales
  the gauge is the last vertex, not spanning. It does not read the scaling, and it does not report
  whether the component is singular: the lift knows that, and whether it regauges, from its
  construction. The auto rule twists within its own band, `TWIST_BAND`, $5\varepsilon$ of the
  trace, not Reduction's `SINGULAR_BAND` for floating.
- **Index width:** Cholesky's `fits_u32` says whether `u32` indexes the most slots any route
  takes, a cover's two sheets and its ground; the reduction is made before the configuration is
  known, so the approximate inverse picks the width from it before the route.
- **The elimination is generic over the clique sample, the copies and the sign.** The engine
  merges a star, caps its copies, cancels opposite signs into surplus and sorts it ascending by
  weight per copy, with $\omega_v$ and the prefix sums; the column of $\mathcal{L}$ is taken from
  that sorted star, exactly. A `FillRule` then samples only the fill, from magnitudes; the engine
  attaches the signs to what it emits. A rule never sees signs; the ground is one more neighbor.
  A rule emits each fill edge once, lower end first, with a weight of at least `MIN_POSITIVE`,
  and drops none. Degree losses from merging stay with the elimination's ordering.
- **Two kinds of star:** an elimination star is merged from the graph; a bipartite star is a row
  of $C$, already unique, under its own stream. Both go to the same rule.
- **A Schur tree has no persistent copies.** Its split exists only within one pivot, so the rule
  is implemented for single-copy graphs only, and the configuration has no field to ask for both.
- **The bipartite form** forms its Schur complement on the cover or with signed entries, as the
  route says: dense, or exactly by summing each clique through $C$, or sampled by the build's
  `FillRule`; then factors it as any matrix. The exact formation is not a rule: it sums by kept
  row, not by star.
- **Extension points.** A new clique sample is one `FillRule` and one `Sampling` variant. A new
  way to factor a frustrated matrix is one `Layout` variant and one branch of the sampled
  constructor. A rule that needs more than one pivot's star, or changes the column of
  $\mathcal{L}$, needs the elimination itself to change.

### Type sketch

A sketch of the crate-internal types until they exist; once the code lands it is the reference,
and this section shrinks to their names, as `architecture.md` asks of exports.

```rust
// Storage: what crosses from Reduction to Cholesky, through the approximate inverse.
pub(crate) struct Grounded<T, P> { pub(crate) adjacency: P, pub(crate) grounding: Grounding<T> }
pub(crate) enum OffDiagonal<T, I> { General(WddCsr<T, I>), Bipartite(Bipartite<T, I>) }
pub(crate) enum Balance { Balanced, Frustrated }
pub(crate) struct Gauge { pub(crate) peak: usize, pub(crate) spans_wide: bool }   // on the factored vertices

// Reduction's result per component; the approximate inverse keeps the lift and hands the rest on.
pub(crate) struct Reduced<T, I, P> {
    pub(crate) key: usize,                       // local vertex 0 in A, eliminated side first: keys every draw
    pub(crate) lift: Box<Lift<T, I>>,            // vertices, congruence, whether it projects and regauges
    pub(crate) matrix: Grounded<T, P>,
    pub(crate) balance: Balance,                 // singular = balanced and ungrounded
    pub(crate) gauge: Gauge,
    pub(crate) clamp: Option<Box<Clamp>>,        // its block's, the only owner (#390)
}
pub(crate) type ReducedComponent<T, I>  = Reduced<T, I, OffDiagonal<T, I>>;
pub(crate) type ReducedComplement<T, I> = Reduced<T, I, Bipartite<T, I>>;   // its lift already on K

// Cholesky's entry: the facts, not the record.
pub(crate) fn factor<T: Real, I: Index, P>(key: usize, matrix: Grounded<T, P>, balance: Balance,
    gauge: Gauge, config: &Config, workspace: &mut Workspace<T, I>) -> Factorization<T, I>;

// The matrix factored, typed by sign; S = Unsigned: Z-matrix, S = Signed: positive entries kept.
struct Wdd<T, I, S: Sign> { matrix: Grounded<T, WddCsr<T, I>>, sign: PhantomData<S> }
enum AnyWdd<T, I> { Unsigned(Wdd<T, I, Unsigned>), Signed(Wdd<T, I, Signed>) }   // the one sign branch

// A matrix and its slot map, only built together; the layout picks the pinned slot.
enum Layout { Floating, Grounded, Cover { grounded: bool } }
struct Represented<T, I> { matrix: AnyWdd<T, I>, layout: Layout }
impl<T: Real, I: Index> Represented<T, I> {
    fn as_is(m: Grounded<T, WddCsr<T, I>>, balance: Balance) -> Self;               // dense arm, signed route
    fn sampled(m: Grounded<T, WddCsr<T, I>>, balance: Balance, route: FrustratedRoute) -> Self;
    fn from_cover_rows(rows: CoverRows<T, I>) -> Self;                                // a complement formed on the cover
}
fn cover_neighbor(k: usize, sheet: usize, crosses: bool) -> usize;                    // the cover's slot of a neighbor
pub(crate) fn fits_u32(n: usize) -> bool;                                            // 2n + 1 slots

// The edge sign: Unsigned is zero-sized and compiles to the Laplacian elimination.
trait Sign: Copy + Send + Sync {
    const SIGNED: bool;                                          // also picks the dense kernel
    fn fill(self, other: Self) -> Self;                          // sign of a sampled edge
    fn merge<T: Real>(a: Entry<Self, T>, b: Entry<Self, T>) -> Merged<Self, T>;
}
struct Unsigned;
struct Signed(bool);
struct Entry<S, T> { sign: S, weight: T }
struct Merged<S, T> { kept: Entry<S, T>, cancelled: T }          // 2·min: surplus at v and an edge to g

// A clique sample: a prepared star in, fill edges out; the column of 𝓛 is the engine's.
struct Star<T, I, C: Copies> { /* sorted ascending, capped copies, ω_v, prefix sums */ }
struct Column<T, I> { /* the pivot's entries of 𝓛 and its 𝓓 */ }
struct FillEdge<T, I, C: Copies> { lo: I, hi: I, copies: C, weight: T }
trait FillRule<T, I, C: Copies> {
    fn fill(&mut self, star: &Star<T, I, C>, rng: &mut Rng, emit: impl FnMut(FillEdge<T, I, C>));
}
struct CliqueTree;                                               // for C = Single and C = Multi
struct SchurTree { split: NonZeroU32 }                           // for C = Single only
enum Formation { Dense, Exact, Sampled(Sampling) }               // a bipartite complement

// Randomness: the build's seed and the key mixed once; stage 0 the elimination's, e + 1 star e's.
struct Streams { key: u64 }

// The factorization of one component; a bipartite one eliminates F first. Its slot count is
// derived from the triangular factor, and a decoded one is checked against its layout.
struct Factorization<T, I> {
    eliminated: Option<Box<BlockElimination<T, I>>>,
    layout: Layout,
    pinned: usize,
    triangular: Triangular<T, I>,
}
enum Triangular<T, I> { Dense(Box<[T]>), Approximate(Box<Steps<T, I>>) }
```

## Why

- **Cover and signed elimination both kept:** they target the same $A_r$. In a Python prototype
  on synthetic frustrated graphs and on within's frustrated components (#347), signed elimination
  needed about half the work of the cover at equal iterations; the cover is the established
  reduction and stays available.
- **Ground vertex and surplus edges both kept:** surplus edges remove the cover's eigenvalue of at
  most the mean surplus, but add edges between real vertices. In the prototype they never needed
  less work than the ground vertex, from a surplus of $10^{-14}$ of the row sums to $1$, on grids
  and on within's frustrated components; above $10^{-2}$ they needed 1.3–2.3 times as much. Auto
  keeps `main`'s rule, surplus edges within the band of the trace: a ground vertex there left PCG
  without progress where the surplus sat on a light region (step F0, #352).
- **Clique tree the default, Schur tree opt-in:** in the same prototype the Schur tree needed two to
  three times the iterations on graphs whose weights span many orders of magnitude, on both
  routes, while it matched the clique tree on moderate weights.
- **One generic matrix type:** the folded, signed and dense inputs held the same data, an
  adjacency and a grounding, and differed only in sign; one type parameter replaces three types and
  a flag.
- **Layout built with its matrix:** a slot map set apart from the matrix could disagree with it;
  built together, it cannot. The pinned slot is part of the map: on `main` three places decide it.
- **Ground never a pivot:** its diagonal is the one quantity no row check bounds, a sum over the
  whole component; pinned, it is never formed, so Cholesky cannot fail by construction, not by a
  guard. One rule then serves every route, where `main` pins $g$ only when it spans wide.
- **Grounding with the matrix, not with the balance:** carried inside the balance, the matrix
  was split over two fields and reassembled by every consumer; the balance alone is a fact with
  no payload.
- **One crossing function for the cover:** on `main` the slot formula has one owner, but the
  rule for which sheet a neighbor lands on is written in the general cover and in the exact and
  the sampled bipartite formation; a route added to one would be missing from the others.
- **The engine sorts and takes the column:** the column of $\mathcal{L}$ is computed from the
  sorted star, its heaviest entry last; a rule free to skip the sort would change it.
- **Rules see neither signs nor the ground:** a star is a tree, hence balanced, so a local switching
  turns any rule for Laplacians into one for signed entries, and a ground vertex is one more
  neighbor. Every rule then serves every route.

## Plan

From `main` to this design; the steps outside Cholesky are in `architecture.md`, Plan. The
refactors change no factor: each is checked bit-identical on
the solve fixtures, the u32/u64 storage test and the wire-format payloads, apart from the format
version where stated. Features follow, each with integration tests in `crates/approx-chol/tests`.
Refactors and behavior changes are separate commits.

| Step | Change | Check |
|---|---|---|
| R1 | `Reduced<T, I, P>` with `key`, `Grounded` matrix, payload-free `Balance`, `Gauge`; `Restriction` renamed `Lift`; named `Bipartite`; the complement path receives `ReducedComplement` directly, no `unreachable!`; no positional tuples. Reverted by #371 (was #363): `main`'s `Reduced` is `Summands<IrreducibleBlock>`, its lift `Congruence` with `FactorTail`. After 0.8.0 | bit-identical |
| R1b | `Lift` built with whether it projects and regauges; `singular` no longer passed at solve time; `UpperAdjacency` renamed `WddCsr`, `Block`'s `restriction` serialized as `lift`; `BlockElimination`, `Streams`, `within_one` and the grain constant moved into Cholesky; its decoder no longer checks Reduction's side policy. After 0.8.0 | bit-identical; format version bumped |
| R2 | `Sign` markers, `Wdd<S>`, `AnyWdd`; the dense arm on `Wdd<S>`, its kernel chosen by `S::SIGNED`; reduction's `Signed` renamed `Folding`, `Arm::Exact` renamed `Dense`. After 0.8.0 | bit-identical |
| R3 | `Layout` and `Represented` with its three constructors; the pin chosen by `Layout`, `main`'s rule kept (the ground if grounded, else the gauge's peak if the scales span wide, else minimum degree's last); the grounded solve's `−Σb` and shift, no-ops since G1, removed, and a decoded grounded factor checked to pin its ground; the cover built in one private function, the bipartite formations through `cover_neighbor`, the dense one still on the cover, then folded; sheets and surplus edges confined there; the slot count derived. After 0.8.0 | bit-identical solves; format version bumped |
| R4 | `Star` (merged, unique), `Column`, `FillEdge`, `FillRule` with `CliqueTree` only; `Formation` keeps the exact formation. After 0.8.0 | bit-identical; Criterion benches |
| R5 | `S: Sign` through the elimination engine, `Unsigned` only. After 0.8.0 | bit-identical; Criterion benches |
| R6 | `Config { seed, sampling, frustrated, exact_max_dim, complement }`, `Eq` and `Hash`; `Sampling::CliqueTree { copies: NonZeroU32 }`, no `SchurTree` before F2; `FrustratedRoute::Cover(CoverSurplus)`, `CoverSurplus` `GroundVertex`, `SurplusEdges` or `Auto`; `SchurMode` also in a whole bipartite build; `split_merge` removed, in `SchurMode` too. On `feat/config-options`, not merged | Python and stub, stubtest, changelog; breaking |
| G1 | The ground pinned in every layout; Reduction's ground range check deleted. Done in #351: within unchanged bit for bit, synthetic +0.1–0.7% PCG iterations, at most +7.5% with one ground | A/B of PCG iterations and nnz(L) in Rust on the synthetic graphs and within's components; inputs the check rejected now factor |
| F0 | `Auto`'s rule: solve error at near-zero surplus, `f32` and `f64`, ground vertex against surplus edges, after G1. Done in #352: `main`'s rule kept; ground and edges equal when every row's surplus is small, the ground stalls PCG where the surplus sits on a light region, every such case within the band | the error test |
| F1 | Signed elimination, `FrustratedRoute::Signed`: `Signed`, cancellation into surplus at $v$ and an edge to $g$, signed Schur complement. In 0.8.0, ported from the #347 Python prototype, stacked on R6, with its own `Sign`, a type parameter of the elimination's methods, the sign held in the weight's sign bit and the buffers shared; R2's `Wdd<S>` comes after 0.8.0 | switched balanced equals its Z-matrix seed for seed; both routes on frustrated inputs; serde round trip; Criterion benches for the wider graph edge |
| F2 | `SchurTree`. After 0.8.0 | expectation equals the clique; every sample connected; at split 1 edge marginals $(w_{vi} + w_{vj})/\omega_v$, above it the auxiliary copies' marginals before contraction and unbiased weights after |

Open for R3: two lifts. Reduction's `Lift` maps a right-hand side into `M`'s coordinates and back;
Cholesky's `sddm::lift` copies one onto the cover's sheets, and `Layout` is to own that lift and
the descent. One of them needs another name before `Layout` lands.

Open for R1b: one record for Cholesky. `Reduced`'s fields are `pub(crate)`, as the sketch has
them, so code outside Reduction can pair a gauge with another lift. Splitting it into
`Reduced { lift, report, problem }`, `problem` holding the key, the matrix, the balance and the
gauge, with fields private to Reduction, would let the approximate inverse keep the lift and the
report and hand `problem` to Cholesky whole. Against it: Cholesky then takes one record, not four
facts, and the record needs an owner, likely Storage beside `Grounded`.
