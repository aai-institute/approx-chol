use super::graph::EliminationGraph;
use crate::factor::clique_tree::{Copies, Fill};
use crate::Real;

/// Bucket queue over live degree estimates (Julia's `ApproxCholPQ`).
pub(super) struct MinDegreeOrder {
    elems: Vec<QueueElem>,
    lists: Vec<u32>,
    min_list: usize,
    bucket_base: usize,
    deltas: DegreeDeltas,
}

/// Net degree change per touched vertex, so each moves once per pivot, not once per fill endpoint.
struct DegreeDeltas {
    /// One slot per vertex, zero at rest: a list would sort per pivot (measured 1.5-1.7x the star sort).
    net: Vec<i64>,
    touched: Vec<u32>,
}

/// 12 bytes: `Option<u32>` keys measured +2-3% on grids.
struct QueueElem {
    prev: u32,
    next: u32,
    /// `u32::MAX` once removed; a live key saturates below it, as only the order matters.
    key: u32,
}

impl MinDegreeOrder {
    /// Buckets span `split · n`, Laplacians.jl's AC2 scale.
    pub(super) fn new<T: Real, C: Copies>(graph: &EliminationGraph<T, C>, split: C::Split) -> Self {
        todo!()
    }

    /// Panics on an empty queue: every step leaves at least one vertex behind.
    pub(super) fn next_vertex(&mut self) -> usize {
        todo!()
    }

    /// The star's degree losses, then fill's gains.
    pub(super) fn update<T: Real>(
        &mut self,
        losses: impl Iterator<Item = (u32, u32)>,
        fill: &Fill<T>,
    ) {
        todo!()
    }
}
