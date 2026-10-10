mod approximate;
mod dense;

pub(super) use dense::DenseCholesky;
pub use dense::{Fallback, PivotDefect, UnusablePivot};

use super::clique_tree::Streams;
use super::config::Multiplicity;
use super::sddm::{Sddm, Sheets};
use crate::Real;
use approximate::ApproximateCholesky;

/// The factor of what the elimination leaves; it builds its own Triangular, so its facts come from that matrix.
#[derive(Clone, Debug)]
pub(super) struct Cholesky<T> {
    sheets: Sheets,
    gauge: Gauge<T>,
}

impl<T: Real> Cholesky<T> {
    /// `vertices` names each base vertex in `A`, so a declined pivot is reported there.
    pub(super) fn dense(
        room: DenseCholesky<T>,
        sddm: &Sddm<'_, T>,
        vertices: &[u32],
    ) -> Result<Self, UnusablePivot> {
        let dense = room.factor(sddm, vertices)?;
        Ok(Self::of(sddm)(Triangular::Dense(dense)))
    }

    /// Consumes `sddm`, so an owned cover or complement is freed before fill grows the graph.
    pub(super) fn approximate(
        sddm: Sddm<'_, T>,
        multiplicity: Multiplicity,
        streams: Streams,
    ) -> Self {
        let of = Self::of(&sddm);
        of(Triangular::Approximate(approximate::eliminate(
            sddm,
            multiplicity,
            streams,
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
    /// The free slot reads zero; the part's projection, or a cover's descent, removes the constant.
    Floating(Triangular<T>),
}

/// The triangular factor, dense or approximate; both leave one slot uneliminated.
#[derive(Clone, Debug)]
enum Triangular<T> {
    Dense(DenseCholesky<T>),
    Approximate(ApproximateCholesky<T>),
}
