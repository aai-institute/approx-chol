use super::clique::Fill;
use super::graph::EliminationGraph;
use super::multiplicity::EdgeCount;
use super::star::Star;
use crate::Real;

/// Bucket queue over live degree estimates (Julia's `ApproxCholPQ`).
pub(super) struct DynamicOrdering {
    elems: Vec<QueueElem>,
    lists: Vec<u32>,
    min_list: usize,
    bucket_base: usize,
}

struct QueueElem {
    prev: u32,
    next: u32,
    key: u32,
}

impl DynamicOrdering {
    pub(super) fn new<C: EdgeCount, T: Real>(graph: &EliminationGraph<C, T>) -> Self {
        todo!()
    }

    /// Panics on an empty queue: every step leaves at least one vertex behind.
    pub(super) fn next_vertex(&mut self) -> usize {
        todo!()
    }

    /// The star lost its pivot, merged copies collapsed, and fill raised degrees.
    pub(super) fn update<C: EdgeCount, T: Real>(&mut self, star: &Star<T, C>, fill: &Fill<T>) {
        todo!()
    }
}
