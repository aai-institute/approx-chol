# Build interface

The contract is `../contracts/build.md`.

## Decision

```rust
// One call, all cores.
pub fn factorize<'a, T: Real, I: PrimInt>(input: impl Into<Input<'a, T, I>>)
    -> Result<ApproxInverse<T>, BuildError>;
pub fn factorize_with<'a, T: Real, I: PrimInt>(input: impl Into<Input<'a, T, I>>, config: Config,
    scaling: ScalingPolicy) -> Result<ApproxInverse<T>, BuildError>;
pub fn factorize_complement<'a, T: Real, I: PrimInt>(input: BipartiteFactorWidth2<'a, T, I>,
    kept: Side, config: Config, scaling: ScalingPolicy) -> Result<ApproxInverse<T>, BuildError>;

// Two stages, the budget named.
impl<T: Real> Sgdd<T> {
    pub fn reduce<'a, I: PrimInt>(input: impl Into<Input<'a, T, I>>, scaling: ScalingPolicy,
        threads: Threads) -> Result<Sgdd<T>, BuildError>;
    pub fn clamps(&self) -> &[Clamp];
    pub fn factor(self, config: Config, threads: Threads) -> ApproxInverse<T>;
}
impl<T: Real> Complement<T> {
    pub fn reduce_onto<'a, I: PrimInt>(input: BipartiteFactorWidth2<'a, T, I>, kept: Side,
        scaling: ScalingPolicy, threads: Threads) -> Result<Complement<T>, BuildError>;
    pub fn clamps(&self) -> &[Clamp];
    pub fn factor(self, config: Config, threads: Threads) -> ApproxInverse<T>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Config {
    pub seed: u64,
    pub sampling: Sampling,
    pub frustrated: FrustratedRoute,
    pub exact_max_dim: usize,
    pub complement: SchurMode,
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sampling {
    CliqueTree { copies: Multiplicity },
    SchurTree { split: NonZeroU32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Multiplicity(/* private: at least 1 */);
impl Multiplicity {
    pub const ONE: Multiplicity;
    pub fn new(k: u32) -> Option<Multiplicity>;
    pub fn get(self) -> NonZeroU32;
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FrustratedRoute {
    Cover(CoverSurplus),
    Signed,
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CoverSurplus {
    GroundVertex,
    SurplusEdges,
    #[default]
    Auto,
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SchurMode {
    Exact,
    Approximate { sampling: Sampling },
    #[default]
    Same,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalingPolicy { /* private */ }
impl ScalingPolicy {
    pub fn new(tolerance: f64, max_iterations: usize) -> Option<ScalingPolicy>;
    pub fn tolerance(&self) -> f64;
    pub fn max_iterations(&self) -> usize;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Threads {
    All,
    Count(NonZeroUsize),
}
#[cfg(feature = "faer")]
impl From<faer::Par> for Threads {}

#[derive(Clone, Debug)]
pub struct ApproxInverse<T> { /* private */ }   // Send + Sync

impl<T: Real> ApproxInverse<T> {
    pub fn n(&self) -> usize;
    pub fn components(&self) -> impl ExactSizeIterator<Item = Component<'_, T>>;
    pub fn clamps(&self) -> &[Clamp];
}

#[derive(Clone, Copy, Debug)]
pub struct Component<'f, T> { /* private */ }
impl<'f, T: Real> Component<'f, T> {
    pub fn vertices(&self) -> impl ExactSizeIterator<Item = usize> + 'f;
    pub fn null_vector(&self) -> Option<&'f [T]>;
    pub fn clamp(&self) -> Option<&'f Clamp>;
}

#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct Clamp {
    pub vertices: Vec<usize>,
    pub violation: f64,
    pub scaling: ScalingOutcome,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalingOutcome {
    Refuted,
    Unsettled,
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuildError {
    OutOfRange { vertex: usize },
    TooLarge { vertex: usize },
}
```

Everything here is at the crate root, as are `Real`, sealed, `PrimInt`, num-traits' re-exported, and
dyn-stack, re-exported as `approx_chol::dyn_stack`; only Submit's types live in `input`. Every error
type implements `Error` and `Display`.

`Config` and `ScalingPolicy` implement `Default`: seed 0, `CliqueTree` with `Multiplicity::ONE`,
`Cover(Auto)`, `exact_max_dim` 24, `Same`, tolerance $10^{-9}$ and 2048 iterations; both are serializable behind the `serde` feature,
and decoding validates as `new` does. `ScalingPolicy::new` accepts a finite tolerance in $[0, 1)$,
never applied below rounding, and any iteration count. `exact_max_dim` is the largest block
dimension factored exactly, a bipartite block's by its kept side. `complement` matters only in a
Schur build; `Same` forms the complement with `sampling`. `frustrated` matters only for a
frustrated component factored approximately; `Auto` places the surplus by the rule of
`../../algorithm.md`, step 5. The algorithm behind each
setting is in `../../algorithm.md`, steps 5 and 7.

- A component's `vertices` are in the order its factor numbers them, in $B$'s numbering;
  `null_vector`, when present, holds one entry per vertex in that order, as computed,
  unnormalized.
- A `Clamp`'s `vertices` are its component of $A$, ascending, in the input's numbering;
  `clamps` are ordered by their first vertex. `Component::clamp` is the clamp of the component
  of $A$ that holds the component, in a whole build its own. A Schur build numbers components on
  the kept side and clamps on the input, left side first.
- A `BuildError`'s `vertex` is the lowest vertex of its component, in the input's numbering.
- `Threads::All` is every thread of the caller's rayon pool; `Count(n)` caps the threads used at
  `n`, whatever the pool's size; faer's `Par::Seq` is `Count(1)`, `Par::Rayon(n)` is `Count(n)`.

Python, the input as in `interface/submit.md`:

```python
def factorize(matrix: sparray | spmatrix, *, config: Config | None = None,
              scaling: ScalingPolicy | None = None,
              threads: int | None = None) -> ApproxInverse: ...

class ApproxInverse(LinearOperator):   # applied as in interface/solve.md
    @property
    def components(self) -> tuple[Component, ...]: ...
    @property
    def clamps(self) -> tuple[Clamp, ...]: ...

class Component:   # read-only
    vertices: NDArray[np.intp]
    null_vector: NDArray[np.float64] | None
    clamp: Clamp | None

class Clamp:       # read-only
    vertices: NDArray[np.intp]
    violation: float
    scaling: Literal["refuted", "unsettled"]

class BuildError(ValueError):
    kind: Literal["out_of_range", "too_large"]
    vertex: int
```

```python
class Config:
    def __init__(self, seed: int | None = None, split: int | None = None,
                 sampling: Literal["clique_tree", "schur_tree"] | None = None,
                 frustrated: Literal["cover", "cover_ground_vertex", "cover_surplus_edges",
                                     "signed"] | None = None,
                 exact_max_dim: int | None = None) -> None: ...
class ScalingPolicy:
    def __init__(self, tolerance: float | None = None,
                 max_iterations: int | None = None) -> None: ...
```

Each argument left `None` takes the Rust default and is read back by a property of its name;
`split` is the `Multiplicity` of a clique tree and the `split` of a Schur tree, at least 1;
`"cover"` is `Cover(Auto)`. `threads=None` is every core. A `threads`, `split` or tolerance out of
range raises `ValueError`. Python builds `f64`. A clamp also warns with a
`RuntimeWarning`.

## Why

- **Two stages, `Sgdd` and `Complement`:** the reduction is complete before any factoring, so a
  caller reads the clamps and can stop there; the budget lives at this level only.
- **`ScalingPolicy` apart from `Config`:** the first configures Reduction, the second Cholesky;
  each stage takes only its own.
- **`Sampling` an enum, its copies in each variant:** a clique tree's copies persist in the graph
  and cap later merges; a Schur tree splits each neighbor only within one pivot. The two numbers
  mean different things, so no configuration pairs a Schur tree with persistent copies.
  `Multiplicity::ONE` is standard AC, one value where `Option<u32>` had three.
- **`FrustratedRoute` and `CoverSurplus` in the configuration:** both routes and both surplus
  encodings target the same $A_r$ (`../../algorithm.md`); they differ in fill and robustness, so
  the caller chooses. The defaults reproduce the current factorization.
- **`Component::clamp` beside `clamps`:** each component of $B$ lies in one component of $A$, so
  a caller reads its null vector and clamp together; `clamps` alone holds a clamp on a component
  of $A$ with no kept vertex. Both read one stored list.
- **The result is $B$, `ApproxInverse`:** its facts, its persistence and its apply all concern
  $B$; the factors $\mathcal{L}$ and $\mathcal{D}$ are not exposed, so no type names them.
- **Vertices in the factor's order:** placing a null vector needs its vertices in one order, not
  sorted; the factor's own costs neither a sort nor a second stored order.
- **Vertices as `usize`:** the storage index is internal; callers index their own arrays with
  `usize`.
- **Python without `factorize_complement`:** the Schur build serves within, which calls Rust.
