use super::clique::Column;
use crate::Real;

/// A step's neighbors start at the previous step's `end`, so no second array can disagree on count.
#[derive(Clone, Copy, Debug)]
struct StepHeader<T> {
    vertex: u32,
    end: u32,
    /// `D⁻¹` premultiplied by the retained fraction, so no pass rebuilds it.
    pivot_scale: T,
}

/// Minimum-degree columns: a serial recurrence, so shares are composed against the original pivot.
#[derive(Clone, Debug)]
pub(crate) struct Sequence<T> {
    steps: Vec<StepHeader<T>>,
    neighbors: Vec<u32>,
    coefficients: Vec<T>,
    /// The one vertex no step eliminates, so nothing divides its entry by a pivot.
    uneliminated: u32,
}

impl<T: Real> Sequence<T> {
    /// Forward, zero the uneliminated entry, backward.
    pub(crate) fn substitute(&self, y: &mut [T]) {
        todo!()
    }
}

/// A sequence cannot exist without naming the vertex its steps left behind.
pub(super) struct SequenceBuilder<T> {
    steps: Vec<StepHeader<T>>,
    neighbors: Vec<u32>,
    coefficients: Vec<T>,
}

impl<T: Real> SequenceBuilder<T> {
    pub(super) fn with_capacity(n: usize) -> Self {
        Self {
            steps: Vec::with_capacity(n),
            neighbors: Vec::new(),
            coefficients: Vec::new(),
        }
    }

    /// The one site deriving the remainder's share.
    pub(super) fn push(&mut self, vertex: usize, pivot: T, column: &Column<T>) {
        todo!()
    }

    pub(super) fn finish(self, uneliminated: usize) -> Sequence<T> {
        Sequence {
            steps: self.steps,
            neighbors: self.neighbors,
            coefficients: self.coefficients,
            uneliminated: uneliminated as u32,
        }
    }
}
