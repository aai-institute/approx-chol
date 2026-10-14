mod approximate;
mod dense;

use rand::rngs::SmallRng;

use super::config::{Config, Multiplicity};
use super::sddm::{Sddm, Sheets};
use crate::narrow;
use crate::{widen, Real};
use approximate::ApproximateCholesky;
use dense::{DenseCholesky, PackedLaplacian};

/// A Cholesky routed dense and factored approximately instead: less accurate than asked.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fallback {
    /// The packed triangle does not fit in memory.
    WillNotFit {
        /// Dimension of the declined matrix, a cover's two sheets counted.
        dimension: usize,
    },
}

/// What the Cholesky factors, built in the form its arm asks for: exact for dense, as configured for approximate.
pub(super) trait CholeskyInput<'a, T: Real> {
    /// Known before either form is built, a cover's two sheets counted.
    fn dimension(&self) -> usize;

    /// Always exact: a dense Cholesky spends nothing on sparsity.
    fn for_dense(self) -> Sddm<'a, T>;

    /// Consumed, so an owned matrix is freed before fill grows the graph.
    fn for_approximate(self) -> Sddm<'a, T>;
}

impl<'a, T: Real> CholeskyInput<'a, T> for Sddm<'a, T> {
    fn dimension(&self) -> usize {
        self.n()
    }

    fn for_dense(self) -> Sddm<'a, T> {
        self
    }

    fn for_approximate(self) -> Sddm<'a, T> {
        self
    }
}

/// The factor of what the elimination leaves; it builds its own Triangular, so its facts come from that matrix.
#[derive(Clone, Debug)]
pub(super) struct Cholesky<T> {
    sheets: Sheets,
    gauge: Gauge,
    /// The matrix's dimension over its sheets, ground excluded.
    n: usize,
    triangular: Triangular<T>,
}

impl<T: Real> Cholesky<T> {
    /// Dense when the config says so and its Laplacian is reserved before the matrix is built, else approximate.
    pub(super) fn factor<'a>(
        input: impl CholeskyInput<'a, T>,
        draws: SmallRng,
        config: &Config,
    ) -> (Self, Option<Fallback>) {
        let approximate = |sddm| Self::approximate(sddm, config.approximate, draws);
        let dimension = input.dimension();
        if !config.routes_dense(dimension) {
            return (approximate(input.for_approximate()), None);
        }
        let Some(laplacian) = PackedLaplacian::reserve(dimension) else {
            let declined = Fallback::WillNotFit { dimension };
            return (approximate(input.for_approximate()), Some(declined));
        };
        let exact = input.for_dense();
        let (sheets, gauge, n) = (exact.sheets(), Gauge::of(&exact), exact.n());
        let triangular = Triangular::Dense(laplacian.factor(&exact));
        let dense = Self {
            sheets,
            gauge,
            n,
            triangular,
        };
        (dense, None)
    }

    fn approximate(sddm: Sddm<'_, T>, multiplicity: Multiplicity, draws: SmallRng) -> Self {
        let (sheets, gauge, n) = (sddm.sheets(), Gauge::of(&sddm), sddm.n());
        let triangular = Triangular::Approximate(approximate::factor(sddm, multiplicity, draws));
        Self {
            sheets,
            gauge,
            n,
            triangular,
        }
    }

    /// An anchored sheet substitutes on `x` in place; the approximate order may read the ground mid-recurrence, so it sits beside the rows.
    pub(super) fn scratch_len(&self) -> usize {
        match (self.sheets, self.gauge) {
            (Sheets::One, Gauge::Anchored) => 0,
            (_, Gauge::Grounded) => self.n + 1,
            (Sheets::Two, Gauge::Anchored) => self.n,
        }
    }

    /// Lifts to the sheets, fixes the gauge, substitutes, and descends.
    pub(super) fn solve(&self, x: &mut [T], scratch: &mut [T]) {
        let triangular = &self.triangular;
        match (self.sheets, self.gauge) {
            (Sheets::One, Gauge::Anchored) => triangular.substitute(x),
            (Sheets::One, Gauge::Grounded) => {
                let rows = &mut scratch[..=x.len()];
                let (ground, values) = rows.split_last_mut().expect("the ground's slot");
                values.copy_from_slice(x);
                let total = x.iter().fold(0.0, |sum, &b| sum + widen(b));
                *ground = narrow::<T>(-total);
                triangular.substitute(rows);
                let (ground, values) = rows.split_last().expect("the ground's slot");
                (x.iter_mut().zip(values)).for_each(|(x, &v)| *x = v - *ground);
            }
            (Sheets::Two, _) => {
                let rows = &mut scratch[..self.scratch_len()];
                Sheets::Two.lift(x, rows);
                // A lift sums to zero, so the ground's right-hand side is zero and its shift cancels in the descent.
                rows[2 * x.len()..].fill(T::zero());
                triangular.substitute(rows);
                Sheets::Two.descend(rows, x);
            }
        }
    }
}

/// How the factor's free slot is fixed.
#[derive(Clone, Copy, Debug)]
enum Gauge {
    /// The ground is the appended slot, right-hand side `−Σb`; every entry then shifts so it reads zero.
    Grounded,
    /// No surplus: the free slot reads zero, and only a floating component's projection or a cover's descent removes the constant.
    Anchored,
}

impl Gauge {
    fn of<T: Real>(sddm: &Sddm<'_, T>) -> Self {
        match sddm.grounded() {
            true => Self::Grounded,
            false => Self::Anchored,
        }
    }
}

/// The triangular factor, dense or approximate; both leave one slot uneliminated.
#[derive(Clone, Debug)]
enum Triangular<T> {
    Dense(DenseCholesky<T>),
    Approximate(ApproximateCholesky<T>),
}

impl<T: Real> Triangular<T> {
    fn substitute(&self, values: &mut [T]) {
        match self {
            Self::Dense(dense) => dense.substitute(values),
            Self::Approximate(approximate) => approximate.substitute(values),
        }
    }
}
