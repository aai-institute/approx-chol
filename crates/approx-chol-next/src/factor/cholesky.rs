mod approximate;
mod dense;

pub use dense::{PivotDefect, UnusablePivot};

use super::clique_tree::SequentialDraws;
use super::config::{Config, Multiplicity, OnDenseFailure};
use super::sddm::{Sddm, Sheets};
use crate::Real;
use approximate::ApproximateCholesky;
use dense::{DenseCholesky, Room};

/// A Cholesky routed dense and factored approximately instead: less accurate than asked.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fallback {
    /// The dense Cholesky reached an unusable pivot.
    UnusablePivot(UnusablePivot),
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
    fn for_dense(&self) -> Sddm<'_, T>;

    /// Consumed, so an owned matrix is freed before fill grows the graph.
    fn for_approximate(self) -> Sddm<'a, T>;
}

impl<'a, T: Real> CholeskyInput<'a, T> for Sddm<'a, T> {
    fn dimension(&self) -> usize {
        self.n()
    }

    fn for_dense(&self) -> Sddm<'_, T> {
        self.view()
    }

    fn for_approximate(self) -> Sddm<'a, T> {
        self
    }
}

/// The factor of what the elimination leaves; it builds its own Triangular, so its facts come from that matrix.
#[derive(Clone, Debug)]
pub(super) struct Cholesky<T> {
    sheets: Sheets,
    gauge: Gauge<T>,
}

impl<T: Real> Cholesky<T> {
    /// Dense when the config says so and its room is reserved before the matrix is built, else approximate; `vertices` names each base vertex in `A`.
    pub(super) fn factor<'a>(
        input: impl CholeskyInput<'a, T>,
        vertices: &[u32],
        draws: SequentialDraws,
        config: &Config,
    ) -> Result<(Self, Option<Fallback>), UnusablePivot> {
        let approximate = |sddm| Self::approximate(sddm, config.approximate, draws);
        let dimension = input.dimension();
        if !config.routes_dense(dimension) {
            return Ok((approximate(input.for_approximate()), None));
        }
        let Some(room) = Room::reserve(dimension) else {
            let declined = Fallback::WillNotFit { dimension };
            return Ok((approximate(input.for_approximate()), Some(declined)));
        };
        let dense = {
            let exact = input.for_dense();
            let factor = room.factor(&exact, vertices);
            factor.map(|dense| Self::of(&exact)(Triangular::Dense(dense)))
        };
        match dense {
            Ok(factor) => Ok((factor, None)),
            Err(pivot) if config.on_dense_failure == OnDenseFailure::Error => Err(pivot),
            Err(pivot) => {
                let declined = Fallback::UnusablePivot(pivot);
                Ok((approximate(input.for_approximate()), Some(declined)))
            }
        }
    }

    fn approximate(sddm: Sddm<'_, T>, multiplicity: Multiplicity, draws: SequentialDraws) -> Self {
        let of = Self::of(&sddm);
        of(Triangular::Approximate(approximate::factor(
            sddm,
            multiplicity,
            draws,
        )))
    }

    /// The facts it keeps of its matrix, taken before the matrix is consumed.
    fn of(sddm: &Sddm<'_, T>) -> impl FnOnce(Triangular<T>) -> Self + use<T> {
        let (sheets, grounded) = (sddm.sheets(), sddm.grounded());
        move |cholesky| {
            let gauge = match grounded {
                true => Gauge::Grounded(cholesky),
                false => Gauge::Floating(cholesky),
            };
            Self { sheets, gauge }
        }
    }

    /// A floating sheet substitutes on `x` in place; the approximate order may read the ground mid-recurrence, so it sits beside the rows.
    pub(super) fn scratch_len(&self, kept: usize) -> usize {
        match (self.sheets, &self.gauge) {
            (Sheets::One, Gauge::Floating(_)) => 0,
            (sheets, Gauge::Grounded(_)) => sheets.count() * kept + 1,
            (Sheets::Two, Gauge::Floating(_)) => 2 * kept,
        }
    }

    /// Lifts to the sheets, fixes the gauge, substitutes, and descends.
    pub(super) fn solve(&self, x: &mut [T], scratch: &mut [T]) {
        todo!()
    }
}

/// How the factor's free slot is fixed.
#[derive(Clone, Debug)]
enum Gauge<T> {
    /// The ground is the appended slot, right-hand side `−Σb`; every entry then shifts so it reads zero.
    Grounded(Triangular<T>),
    /// The free slot reads zero; the component's projection, or a cover's descent, removes the constant.
    Floating(Triangular<T>),
}

/// The triangular factor, dense or approximate; both leave one slot uneliminated.
#[derive(Clone, Debug)]
enum Triangular<T> {
    Dense(DenseCholesky<T>),
    Approximate(ApproximateCholesky<T>),
}
