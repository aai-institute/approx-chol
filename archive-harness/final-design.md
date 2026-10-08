# Laplacian/Sddm design after 3 review rounds (2026-10-07)

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
    surplus: Option<Vec<T>>, // Some iff some component's surplus total is positive
}

#[derive(Clone)]
enum Components {
    Connected,                                 // exactly one component, n >= 1
    Split { order: Vec<u32>, ends: Vec<u32> }, // any other count; n = 0 is zero blocks
}

impl<T: Real> Laplacian<T> {
    pub fn new(row_ptrs: Vec<u32>, neighbors: Vec<u32>, weights: Vec<T>) -> Result<Self, LaplacianError>;
}
impl<T: Real> Sddm<T> {
    pub fn new(row_ptrs: Vec<u32>, neighbors: Vec<u32>, weights: Vec<T>, surplus: Vec<T>) -> Result<Self, SddmError>;
}
impl<T> From<Laplacian<T>> for Sddm<T> {}                                    // move
impl<T: Real, I: PrimInt> TryFrom<CsrRef<'_, T, I>> for Sddm<T> { type Error = Error; }
impl<'a, T, I> CsrRef<'a, T, I> { pub fn new(..) -> Result<Self, CsrError>; } // + nnz and n + 1 fit u32

pub fn factorize<T: Real>(sddm: impl Into<Sddm<T>>) -> Factor<T>;
pub fn factorize_with<T: Real>(sddm: impl Into<Sddm<T>>, config: Config) -> Result<Factor<T>, UnusablePivot>;

pub enum Error { PositiveOffDiagonal { .. }, NonFiniteValue { position }, Asymmetric { .. },
                 NotDiagonallyDominant { .. }, NonFiniteRow { row }, MagnitudeTooSmall { .. }, GroundOverflow { vertex } }
pub enum AdjacencyError { /* shape variants */ Weight { edge: (usize, usize), defect: WeightDefect } }
pub enum WeightDefect { NonFinite, NotPositive, BelowFloor }
pub enum LaplacianError { Adjacency(AdjacencyError), DegreeNotFinite { vertex: usize } }
pub enum SurplusDefect { Negative, BelowFloor }
pub enum SddmError { Adjacency(AdjacencyError), SurplusLength { len: usize, n: usize },
                     Surplus { vertex: usize, defect: SurplusDefect },
                     DiagonalNotFinite { vertex: usize }, GroundOverflow { vertex: usize } }

// private to module `sddm`
struct Incidence<T> { degrees: Vec<T>, sets: DisjointSets }  // all three routes
impl<T: Real> Incidence<T> {
    fn row(&mut self, row: usize) -> RowIncidence<'_, T>;
    fn finish(self, surplus: Option<&[T]>) -> Result<Components, NotFinite>;
}
struct RowIncidence<'a, T> { incidence: &'a mut Incidence<T>, row: u32, root: u32 }
impl<T: Real> RowIncidence<'_, T> { fn add(&mut self, col: usize, weight: T); }
fn traverse<T: Real>(row_ptrs: &[u32], neighbors: &[u32], weights: &[T]) -> Result<Incidence<T>, AdjacencyError>;
fn check_surplus<T: Real>(surplus: &[T], n: usize) -> Result<(), SddmError>; // len, s < 0, 0 < s < floor
fn check_weight<T: Real>(weight: T) -> Result<T, WeightDefect>;
fn grounding<T: Real>(components: &Components, surplus: Vec<T>) -> Result<Option<Vec<T>>, GroundOverflow>;
enum RowBalance<T> { NonFinite, Deficit, Negligible, BelowFloor, Surplus(T) }

// factorization
pub(crate) struct Component<'a, T> { view: View<'a, T>, gauge: Gauge<'a, T> }
pub(crate) enum Gauge<'a, T> { Floating, Grounded(&'a [T]) }
pub(crate) struct Block<T> { cholesky: Cholesky<T>, gauge: BlockGauge }
```
