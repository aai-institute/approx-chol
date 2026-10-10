use crate::factor::clique_tree::Column;
use crate::Real;

/// A step's neighbors start at the previous step's `end`, so no second array can disagree on count.
#[derive(Clone, Copy, Debug)]
struct StepHeader<T> {
    vertex: u32,
    /// `usize` like `Rows`' pointers: a sampled exact complement can pass `u32::MAX` entries.
    end: usize,
    /// `D⁻¹` premultiplied by the retained fraction, so no pass rebuilds it.
    pivot_scale: T,
}

/// The eliminated columns, appended in pivot order.
#[derive(Clone, Debug)]
pub(super) struct Steps<T> {
    headers: Vec<StepHeader<T>>,
    neighbors: Vec<u32>,
    coefficients: Vec<T>,
}

impl<T: Real> Steps<T> {
    pub(super) fn with_capacity(n: usize) -> Self {
        Self {
            headers: Vec::with_capacity(n),
            neighbors: Vec::new(),
            coefficients: Vec::new(),
        }
    }

    /// The one site deriving the remainder's share.
    pub(super) fn push(&mut self, vertex: usize, pivot: T, column: &Column<T>) {
        todo!()
    }

    /// A factor cannot exist without naming the vertex its steps left behind.
    pub(super) fn finish(self, uneliminated: usize) -> SampledCholesky<T> {
        SampledCholesky {
            steps: self,
            uneliminated: uneliminated as u32,
        }
    }
}

/// `LDLᵀ` with sampled columns: a serial recurrence, so shares are composed against the original pivot.
#[derive(Clone, Debug)]
pub(in crate::factor::remainder) struct SampledCholesky<T> {
    steps: Steps<T>,
    /// The one vertex no step eliminates, so nothing divides its entry by a pivot.
    uneliminated: u32,
}

impl<T: Real> SampledCholesky<T> {
    /// Forward, zero the uneliminated entry, backward; the gather dispatches `[]`, `[_]`, lanes by row length.
    pub(in crate::factor::remainder) fn substitute(&self, y: &mut [T]) {
        todo!()
    }
}
