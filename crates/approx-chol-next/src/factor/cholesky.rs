mod approximate;
mod dense;

use rand::rngs::Xoshiro256PlusPlus;

use super::config::{Config, Multiplicity};
use super::sddm::{descend, lift, Sddm, Sheets};
use crate::narrow;
use crate::{widen, Index, Real};
use approximate::ApproximateCholesky;
use dense::PackedLaplacian;

/// A Cholesky routed dense and factored approximately instead: less accurate than asked.
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fallback {
    /// The packed triangle does not fit in memory.
    WillNotFit {
        /// Dimension of the declined matrix, a cover's two sheets counted.
        dimension: usize,
    },
}

/// What the Cholesky factors, built in the form its arm asks for: exact for dense, as configured for approximate.
pub(super) trait CholeskyInput<'a, T: Real, I: Index> {
    /// Known before either form is built, a cover's two sheets counted.
    fn dimension(&self) -> usize;

    /// Always exact: a dense Cholesky spends nothing on sparsity.
    fn for_dense(self) -> Sddm<'a, T, I>;

    /// Consumed, so an owned matrix is freed before fill grows the graph.
    fn for_approximate(self) -> Sddm<'a, T, I>;
}

impl<'a, T: Real, I: Index> CholeskyInput<'a, T, I> for Sddm<'a, T, I> {
    fn dimension(&self) -> usize {
        self.n()
    }

    fn for_dense(self) -> Sddm<'a, T, I> {
        self
    }

    fn for_approximate(self) -> Sddm<'a, T, I> {
        self
    }
}

/// The factor of what the elimination leaves; it builds its own Triangular, so its facts come from that matrix.
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(remote = "Self")
)]
#[derive(Clone, Debug)]
pub(super) struct Cholesky<T, I> {
    sheets: Sheets,
    /// The ground is the appended slot, right-hand side `−Σb`, every entry then shifted so it reads zero; else the free slot reads zero.
    grounded: bool,
    /// The matrix's dimension over its sheets, ground excluded.
    n: usize,
    triangular: Triangular<T, I>,
}

impl<T: Real, I: Index> Cholesky<T, I> {
    /// Dense when the config says so and its Laplacian is reserved before the matrix is built, else approximate.
    pub(super) fn factor<'a>(
        input: impl CholeskyInput<'a, T, I>,
        draws: Xoshiro256PlusPlus,
        config: &Config,
    ) -> (Self, Option<Fallback>) {
        let approximate = |sddm| Self::approximate(sddm, config.approximate, draws);
        let dimension = input.dimension();
        if dimension > config.dense_up_to {
            return (approximate(input.for_approximate()), None);
        }
        let Some(laplacian) = PackedLaplacian::reserve(dimension) else {
            let declined = Fallback::WillNotFit { dimension };
            return (approximate(input.for_approximate()), Some(declined));
        };
        let exact = input.for_dense();
        let (sheets, grounded, n) = (exact.sheets(), exact.grounded(), exact.n());
        let triangular = Triangular::Dense(laplacian.factor(&exact));
        let dense = Self {
            sheets,
            grounded,
            n,
            triangular,
        };
        (dense, None)
    }

    fn approximate(
        sddm: Sddm<'_, T, I>,
        multiplicity: Multiplicity,
        draws: Xoshiro256PlusPlus,
    ) -> Self {
        let (sheets, grounded, n) = (sddm.sheets(), sddm.grounded(), sddm.n());
        let triangular = Triangular::Approximate(approximate::factor(sddm, multiplicity, draws));
        Self {
            sheets,
            grounded,
            n,
            triangular,
        }
    }

    /// Whether it solves a singular Laplacian: one sheet, no ground.
    pub(super) fn singular(&self) -> bool {
        self.sheets == Sheets::One && !self.grounded
    }

    /// An anchored sheet substitutes on `x` in place; the approximate order may read the ground mid-recurrence, so it sits beside the rows.
    pub(super) fn scratch_len(&self) -> usize {
        match (self.sheets, self.grounded) {
            (Sheets::One, false) => 0,
            (_, true) => self.n + 1,
            (Sheets::Two, false) => self.n,
        }
    }

    /// Lifts to the sheets, fixes the gauge, substitutes, and descends.
    pub(super) fn solve(&self, x: &mut [T], scratch: &mut [T]) {
        let triangular = &self.triangular;
        match (self.sheets, self.grounded) {
            (Sheets::One, false) => triangular.substitute(x),
            (Sheets::One, true) => {
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
                lift(x, rows);
                // A lift sums to zero, so the ground's right-hand side is zero and its shift cancels in the descent.
                rows[2 * x.len()..].fill(T::zero());
                triangular.substitute(rows);
                descend(rows, x);
            }
        }
    }
}

/// The triangular factor, dense or approximate; both leave one slot uneliminated.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
enum Triangular<T, I> {
    /// Packed lower triangle of `L` in `LLᵀ`.
    Dense(Vec<T>),
    Approximate(ApproximateCholesky<T, I>),
}

impl<T: Real, I: Index> Triangular<T, I> {
    fn substitute(&self, values: &mut [T]) {
        match self {
            Self::Dense(factor) => dense::substitute(factor, values),
            Self::Approximate(approximate) => approximate.substitute(values),
        }
    }
}

#[cfg(feature = "serde")]
checked_serde!(Cholesky);

#[cfg(feature = "serde")]
impl<T: Real, I: Index> Cholesky<T, I> {
    fn checked(self) -> Option<Self> {
        if self.n == 0 {
            return None;
        }
        let slots = self.n.checked_add(usize::from(self.grounded))?;
        let fits = match &self.triangular {
            Triangular::Dense(dense) => dense::fits(dense, slots),
            Triangular::Approximate(approximate) => approximate.fits(slots),
        };
        fits.then_some(self)
    }

    /// Whether it solves a block of `kept` vertices on its sheets.
    pub(super) fn solves(&self, kept: usize) -> bool {
        Some(self.n) == self.sheets.count().checked_mul(kept)
    }
}
