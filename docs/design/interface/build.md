# Build interface

The contract is `../contracts/build.md`. The observer and the stopped outcome are open until
`inflight` is ready; this interface leaves them out.

## Decision

```rust
// One call, all cores.
pub fn factorize<'a, T: Real, I: PrimInt>(input: impl Into<Input<'a, T, I>>)
    -> Result<ApproxCholesky<T>, BuildError>;
pub fn factorize_with<'a, T: Real, I: PrimInt>(input: impl Into<Input<'a, T, I>>, config: Config,
    scaling: ScalingPolicy) -> Result<ApproxCholesky<T>, BuildError>;
pub fn factorize_complement<'a, T: Real, I: PrimInt>(input: BipartiteFactorWidth2<'a, T, I>,
    kept: Side, config: Config, scaling: ScalingPolicy) -> Result<ApproxCholesky<T>, BuildError>;

// Two stages, the budget named.
impl<T: Real> Sgdd<T> {
    pub fn reduce<'a, I: PrimInt>(input: impl Into<Input<'a, T, I>>, scaling: ScalingPolicy,
        threads: Threads) -> Result<Sgdd<T>, BuildError>;
    pub fn clamps(&self) -> &[Clamp];
    pub fn factor(self, config: Config, threads: Threads) -> ApproxCholesky<T>;
}
impl<T: Real> Complement<T> {
    pub fn reduce_onto<'a, I: PrimInt>(input: BipartiteFactorWidth2<'a, T, I>, kept: Side,
        scaling: ScalingPolicy, threads: Threads) -> Result<Complement<T>, BuildError>;
    pub fn clamps(&self) -> &[Clamp];
    pub fn factor(self, config: Config, threads: Threads) -> ApproxCholesky<T>;
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Config {
    pub seed: u64,
    pub split_merge: NonZeroU32,
    pub exact_max_dim: usize,
    pub complement: SchurMode,
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SchurMode {
    Exact,
    Approximate { split_merge: NonZeroU32 },
    #[default]
    Same,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
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
pub struct ApproxCholesky<T> { /* private */ }   // Send + Sync

impl<T: Real> ApproxCholesky<T> {
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

`Config` and `ScalingPolicy` default to seed 0, `split_merge` 1, `exact_max_dim` 24, `Same`,
tolerance $10^{-9}$ and 2048 iterations; both are serializable behind the `serde` feature, and
decoding validates as `new` does. `ScalingPolicy::new` accepts a finite tolerance in $[0, 1)$,
never applied below rounding, and any iteration count. `complement` matters only in a Schur
build.

- A component's `vertices` are ascending, in $B$'s numbering; `null_vector`, when present, holds
  one entry per vertex in that order, as computed, unnormalized.
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
              scaling: ScalingPolicy | None = None, threads: int | None = None) -> ApproxCholesky: ...

class ApproxCholesky:
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
                 exact_max_dim: int | None = None) -> None: ...
class ScalingPolicy:
    def __init__(self, tolerance: float | None = None,
                 max_iterations: int | None = None) -> None: ...
```

Each argument left `None` takes the Rust default and is read back by a property of its name;
`split` is Rust's `split_merge`, at least 1. `threads=None` is every core. A `threads`, `split` or
tolerance out of range raises `ValueError`. Python builds `f64`. A clamp also warns with a
`RuntimeWarning`.

## Why

- **Two stages, `Sgdd` and `Complement`:** the reduction is complete before any factoring, so a
  caller reads the clamps and can stop there; the budget, and later the observer, live at this
  level only.
- **`ScalingPolicy` apart from `Config`:** the first configures Reduction, the second Cholesky;
  each stage takes only its own.
- **`split_merge: NonZeroU32`:** one value per fill-in level, 1 being standard AC.
- **`Component::clamp` beside `clamps`:** each component of $B$ lies in one component of $A$, so
  a caller reads its null vector and clamp together; `clamps` alone holds a clamp on a component
  of $A$ with no kept vertex. Both read one stored list.
- **Vertices as `usize`:** the storage index is internal; callers index their own arrays with
  `usize`.
- **Python without `factorize_complement`:** within, its one consumer, calls Rust.
