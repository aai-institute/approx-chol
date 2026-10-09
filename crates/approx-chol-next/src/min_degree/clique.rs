use super::multiplicity::EdgeCount;
use super::star::Star;
use crate::sampling::CdfSampler;
use crate::Real;

/// Fill edges `(lo, hi, weight)`, `lo < hi`.
pub(super) type Fill<T> = Vec<(u32, u32, T)>;

/// One column of the factor: shares of the original pivot, the remainder named, not weighted.
pub(super) struct Column<T> {
    pub(super) neighbors: Vec<u32>,
    pub(super) coefficients: Vec<T>,
    /// `None` keeps the whole pivot: an empty star hands nothing on.
    pub(super) remainder: Option<u32>,
}

impl<T: Real> Column<T> {
    pub(super) fn new() -> Self {
        Self {
            neighbors: Vec::new(),
            coefficients: Vec::new(),
            remainder: None,
        }
    }
}

/// The exact column, and a clique tree sampled in its place (GKS 2023, Algorithm 5); returns the pivot.
pub(super) fn sample_column<T: Real, C: EdgeCount>(
    star: &Star<T, C>,
    sampler: &mut CdfSampler<T>,
    column: &mut Column<T>,
    fill: &mut Fill<T>,
) -> T {
    todo!()
}
