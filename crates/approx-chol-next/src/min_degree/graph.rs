use super::multiplicity::EdgeCount;
use crate::sgdd::Sddm;
use crate::Real;

/// One half of an edge; `mirror` indexes the other half in `to`'s list, so removal is O(1).
#[derive(Clone, Copy, Debug)]
pub(super) struct Edge<C, T> {
    pub(super) to: u32,
    pub(super) mirror: u32,
    pub(super) copies: C,
    pub(super) weight: T,
}

/// The mutable graph minimum-degree eliminates; a grounded one carries its ground as the last vertex.
pub(super) struct EliminationGraph<C, T> {
    lists: Vec<Vec<Edge<C, T>>>,
}

impl<C: EdgeCount, T: Real> EliminationGraph<C, T> {
    /// The ground is a vertex like any other: forcing it last measured +0-10% PCG iterations.
    pub(super) fn of(sddm: &Sddm<'_, T>, split: C::Split) -> Self {
        todo!()
    }

    pub(super) fn n(&self) -> usize {
        self.lists.len()
    }

    pub(super) fn degree(&self, v: usize) -> usize {
        self.lists[v].len()
    }

    pub(super) fn edges(&self, v: usize) -> &[Edge<C, T>] {
        &self.lists[v]
    }

    /// Drops `v`'s edges from both sides.
    pub(super) fn eliminate_vertex(&mut self, v: usize) {
        todo!()
    }

    pub(super) fn add_fill(&mut self, u: u32, w: u32, weight: T) {
        todo!()
    }
}
