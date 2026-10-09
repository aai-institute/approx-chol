use crate::clique_tree::Copies;
use crate::sddm::Sddm;
use crate::Real;

/// One half of an edge; `mirror` indexes the other half in `to`'s list, so removal is O(1).
#[derive(Clone, Copy, Debug)]
struct Edge<T, C> {
    to: u32,
    mirror: u32,
    copies: C,
    weight: T,
}

/// The mutable graph minimum-degree eliminates; a grounded one carries its ground as the last vertex.
pub(super) struct EliminationGraph<T, C> {
    lists: Vec<Vec<Edge<T, C>>>,
}

impl<T: Real, C: Copies> EliminationGraph<T, C> {
    /// The ground is a vertex like any other: forcing it last measured +0-10% PCG iterations.
    pub(super) fn of(sddm: &Sddm<'_, T>, split: C::Split) -> Self {
        todo!()
    }

    pub(super) fn n(&self) -> usize {
        self.lists.len()
    }

    /// Virtual copies, not stored entries: AC2 orders by the multigraph's degree.
    pub(super) fn degree(&self, v: usize) -> usize {
        self.lists[v].iter().map(|e| e.copies.get() as usize).sum()
    }

    pub(super) fn neighbors(&self, v: usize) -> impl Iterator<Item = (u32, C, T)> + '_ {
        self.lists[v].iter().map(|e| (e.to, e.copies, e.weight))
    }

    /// Drops `v`'s edges from both sides.
    pub(super) fn eliminate_vertex(&mut self, v: usize) {
        todo!()
    }

    pub(super) fn add_fill(&mut self, u: u32, w: u32, weight: T) {
        todo!()
    }
}
