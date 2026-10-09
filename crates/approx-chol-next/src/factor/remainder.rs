use crate::clique_tree::Streams;
use crate::config::Multiplicity;
use crate::dense::{DenseCholesky, Fallback};
use crate::min_degree::{self, SampledCholesky};
use crate::sddm::{Sddm, Sheets};
use crate::Real;

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
    fn of(sddm: &Sddm<'_, T>) -> impl FnOnce(Cholesky<T>) -> Self {
        let (sheets, grounded) = (sddm.sheets(), sddm.grounded());
        move |cholesky| {
            let gauge = match grounded {
                true => Gauge::Grounded(cholesky),
                false => Gauge::Floating(cholesky),
            };
            Self { sheets, gauge }
        }
    }

    /// The lifted rows and a grounded factor's ground slot, for `kept` vertices.
    pub(super) fn scratch_len(&self, kept: usize) -> usize {
        let ground = usize::from(matches!(self.gauge, Gauge::Grounded(_)));
        self.sheets.count() * kept + ground
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
