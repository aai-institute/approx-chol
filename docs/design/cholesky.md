# Cholesky

How the Cholesky part factors one WDD matrix, and where each choice of the configuration lives.
The mathematics is steps 5–8 of `../algorithm.md`; the part's boundary is in `architecture.md`.
Planned: on `main` the cover is still built in Reduction (`sgdd`) and handed over as sheets.

## Decision

Cholesky receives one WDD matrix with its balance, the vertices it keeps, the configuration, the
seed and the component's stream. Five independent choices shape the factorization; each has one
owner.

| Choice | Values | Owner | Decided |
|---|---|---|---|
| How a frustrated matrix is factored | Gremban cover, signed elimination | `FrustratedRoute` | per build, applied per component |
| Where a cover puts its surplus | ground vertex, surplus edges, auto | `CoverSurplus` | per build; auto per component |
| Clique sample | clique tree, Schur tree | `Sampling`, a `FillRule` | per build |
| Fill-in level | copies (clique tree), split (Schur tree) | the `Sampling` variant | per build |
| Sign of the eliminated entries | unsigned, signed | a `Sign` type | follows from balance and route |

- **One matrix type, typed by sign.** The matrix factored is a WDD matrix with its grounding,
  `Wdd<Unsigned>` for a Z-matrix and `Wdd<Signed>` for one with positive entries. The dense arm
  factors either as it is; its kernel is chosen by the sign type, not by a flag.
- **A representation is a matrix and its slot map, built together.** `Represented` pairs the
  matrix the elimination sees with `Layout`, which maps the elimination's slots to the component's
  vertices and owns the lift and the descent:

  | Layout | Slots | Built from |
  |---|---|---|
  | `Floating` | $n$, one pinned | a balanced matrix without surplus |
  | `Grounded` | $n + 1$, $g$ appended | a balanced matrix with surplus |
  | `Cover { grounded }` | $2n$, $+1$ if grounded | a frustrated matrix on the cover route |
  | `Signed` | $n + 1$, $g$ pinned | a frustrated matrix on the signed route |

  `Represented` has two constructors: the exact one, the matrix as it is, for the dense arm and
  the signed route; and the sampled one, which builds the cover when the route asks for it. The
  cover's construction, its surplus edges and the auto rule live in one private function behind
  the second.
- **The elimination is generic over the clique sample, the copies and the sign.** A star is
  merged, its copies capped and its opposite signs cancelled into surplus, then handed to a
  `FillRule`, which emits fill edges. The column of $\mathcal{L}$ is exact; a rule samples only the
  fill, and sees magnitudes, never signs or the ground. A rule states whether it needs the star
  sorted. A task's workspace holds one engine per sign for the build's rule.
- **A Schur tree has no persistent copies.** Its split exists only within one pivot, so the rule
  is implemented for single-copy graphs only, and the configuration has no field to ask for both.
- **The bipartite form** forms its Schur complement on the cover or with signed entries, as the
  route says, exactly or by the same `FillRule`, then factors it as any matrix.
- **Extension points.** A new clique sample is one `FillRule` and one `Sampling` variant. A new
  way to factor a frustrated matrix is one `Layout` variant and one branch of the sampled
  constructor. A rule that needs more than one pivot's star, or changes the column of
  $\mathcal{L}$, needs the elimination itself to change.

### Type sketch

A sketch of the crate-internal types until they exist; once the code lands it is the reference,
and this section shrinks to their names, as `architecture.md` asks of exports.

```rust
// reduction → cholesky: one reduced component; P is what Cholesky factors.
pub(crate) struct Reduced<T, I, P> {
    pub(crate) key: usize,                       // the component's first vertex in A: keys every draw
    pub(crate) lift: Box<Lift<T, I>>,            // projection, scaling, switching; a Schur build's on K
    pub(crate) part: P,
    pub(crate) balance: Balance<T>,              // Balanced(Grounding) | Frustrated(Grounding)
    pub(crate) clamp: Option<Box<Clamp>>,
}
pub(crate) enum OffDiagonal<T, I> { General(WddCsr<T, I>), Bipartite(Bipartite<T, I>) }
pub(crate) type ReducedComponent<T, I>  = Reduced<T, I, OffDiagonal<T, I>>;
pub(crate) type ReducedComplement<T, I> = Reduced<T, I, Bipartite<T, I>>;   // its lift already on K

// The matrix factored, typed by sign; S = Unsigned: Z-matrix, S = Signed: positive entries kept.
struct Wdd<T, I, S: Sign> { adjacency: WddCsr<T, I>, grounding: Grounding<T>, sign: PhantomData<S> }
enum AnyWdd<T, I> { Unsigned(Wdd<T, I, Unsigned>), Signed(Wdd<T, I, Signed>) }   // the one sign branch

// A matrix and its slot map, only built together.
enum Layout { Floating, Grounded, Cover { grounded: bool }, Signed }
struct Represented<T, I> { matrix: AnyWdd<T, I>, layout: Layout }
impl<T: Real, I: Index> Represented<T, I> {
    fn exact(m: WddCsr<T, I>, balance: Balance<T>) -> Self;                          // dense arm, signed route
    fn sampled(m: WddCsr<T, I>, balance: Balance<T>, route: FrustratedRoute) -> Self; // builds the cover if asked
}

// The edge sign: Unsigned is zero-sized and compiles to the Laplacian elimination.
trait Sign: Copy + Send + Sync {
    const SIGNED: bool;                                          // also picks the dense kernel
    fn fill(self, other: Self) -> Self;                          // sign of a sampled edge
    fn merge<T: Real>(a: (Self, T), b: (Self, T)) -> ((Self, T), T);   // cancelled 2·min → surplus
}
struct Unsigned;
struct Signed(bool);

// A clique sample: a prepared star in, fill edges out; the column of 𝓛 stays exact.
struct Star<T, I, C: Copies, S: Sign> { /* merged neighbors, capped copies, weights, ω_v */ }
enum StarOrder { Ascending, Any }
trait FillRule<T, I, C: Copies, S: Sign> {
    const ORDER: StarOrder;
    fn fill(&mut self, star: &Star<T, I, C, S>, rng: &mut Rng, emit: impl FnMut(FillEdge<T, I, C, S>));
}
struct CliqueTree;                                               // for C = Single and C = Multi
struct SchurTree { split: NonZeroU32 }                           // for C = Single only

// The factorization of one component.
struct Factorization<T, I> { layout: Layout, pinned: usize, triangular: Triangular<T, I> }
enum Triangular<T, I> { Dense(Box<[T]>), Approximate(Box<Steps<T, I>>) }
```

## Why

- **Cover and signed elimination both kept:** they target the same $A_r$. In a Python prototype
  on synthetic frustrated graphs and on within's frustrated components (#347), signed elimination
  needed about half the work of the cover at equal iterations; the cover is the established
  reduction and stays available.
- **Ground vertex and surplus edges both kept:** surplus edges remove the cover's eigenvalue at the
  mean surplus, but add fill where the surplus is large. Where auto should switch is being
  measured; until then auto keeps the current rule.
- **Clique tree the default, Schur tree opt-in:** in the same prototype the Schur tree needed two to
  three times the iterations on graphs whose weights span many orders of magnitude, on both
  routes, while it matched the clique tree on moderate weights.
- **One generic matrix type:** the folded, signed and dense inputs held the same data, an
  adjacency and a grounding, and differed only in sign; one type parameter replaces three types and
  a flag.
- **Layout built with its matrix:** a slot map set apart from the matrix could disagree with it;
  built together, it cannot.
- **Rules see neither signs nor the ground:** a star is a tree, hence balanced, so a local switching
  turns any rule for Laplacians into one for signed entries, and a ground vertex is one more
  neighbor. Every rule then serves every route.
