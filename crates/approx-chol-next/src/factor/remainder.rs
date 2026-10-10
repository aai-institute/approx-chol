mod dense;
mod min_degree;

pub use dense::{Fallback, PivotDefect, UnusablePivot};

use super::clique_tree::Streams;
use super::config::Multiplicity;
use super::sddm::{Sddm, Sheets};
use crate::Real;
use dense::DenseCholesky;
use min_degree::SampledCholesky;

/// The factor of what the elimination leaves; it builds its own Cholesky, so its facts come from that matrix.
#[derive(Clone, Debug)]
pub(super) struct SddmFactor<T> {
    sheets: Sheets,
    gauge: Gauge<T>,
}

impl<T: Real> SddmFactor<T> {
    /// `vertices` names each base vertex in `A`, so a declined pivot is reported there.
    pub(super) fn dense(sddm: &Sddm<'_, T>, vertices: &[u32]) -> Result<Self, Fallback> {
        let dense = DenseCholesky::factor(sddm, vertices)?;
        Ok(Self::of(sddm)(Cholesky::Dense(dense)))
    }

    /// Consumes `sddm`, so an owned cover or complement is freed before fill grows the graph.
    pub(super) fn sampled(sddm: Sddm<'_, T>, multiplicity: Multiplicity, streams: Streams) -> Self {
        let of = Self::of(&sddm);
        of(Cholesky::Sampled(min_degree::eliminate(
            sddm,
            multiplicity,
            streams,
        )))
    }

    /// The facts it keeps of its matrix, taken before the matrix is consumed.
    fn of(sddm: &Sddm<'_, T>) -> impl FnOnce(Cholesky<T>) -> Self + use<T> {
        let (sheets, grounded) = (sddm.sheets(), sddm.grounded());
        move |cholesky| {
            let gauge = match grounded {
                true => Gauge::Grounded(cholesky),
                false => Gauge::Floating(cholesky),
            };
            Self { sheets, gauge }
        }
    }

    /// A floating sheet substitutes on `x` in place; the sampled order may read the ground mid-recurrence, so it sits beside the rows.
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
    Grounded(Cholesky<T>),
    /// The free slot reads zero; the part's projection, or a cover's descent, removes the constant.
    Floating(Cholesky<T>),
}

/// Exact or sampled; both leave one slot uneliminated.
#[derive(Clone, Debug)]
enum Cholesky<T> {
    Dense(DenseCholesky<T>),
    Sampled(SampledCholesky<T>),
}
