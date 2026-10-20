use crate::factor::clique_tree::Copies;
use crate::factor::sddm::Sddm;
use crate::{Index, Real};

/// One half of an edge; `mirror` indexes the other half in `to`'s list, so removal is O(1).
#[derive(Clone, Copy, Debug)]
struct Edge<T, I, C> {
    to: I,
    mirror: u32,
    copies: C,
    weight: T,
}

/// The mutable graph minimum-degree eliminates; a grounded one carries its ground as the last vertex.
pub(super) struct EliminationGraph<T, I, C> {
    lists: Vec<Vec<Edge<T, I, C>>>,
}

/// A block this small keeps its lists for the task's next; a larger one frees each at elimination (peak −11…−14%).
const REUSED_UP_TO: usize = 1024;

impl<T: Real, I: Index, C: Copies> EliminationGraph<T, I, C> {
    pub(super) fn new() -> Self {
        Self { lists: Vec::new() }
    }

    /// The ground is a vertex like any other: forcing it last measured +0-10% PCG iterations.
    pub(super) fn load(&mut self, sddm: &Sddm<T, I>, split: C::Split) {
        let n = sddm.n() + usize::from(sddm.grounded());
        // One slot past the degree: the first fill edge reallocating measured +2-3% on grids.
        let mut capacities = vec![1; n];
        sddm.grounded_edges(|k, j, _| {
            capacities[k.index()] += 1;
            capacities[j.index()] += 1;
        });
        self.lists.truncate(n);
        let reused = self.lists.len();
        (self.lists.iter_mut().zip(&capacities)).for_each(|(list, &c)| list.reserve_exact(c));
        (self.lists).extend(capacities[reused..].iter().map(|&c| Vec::with_capacity(c)));
        let copies = C::initial(split);
        sddm.grounded_edges(|k, j, w| self.push_pair(k, j, copies, w));
    }

    pub(super) fn n(&self) -> usize {
        self.lists.len()
    }

    /// Twice the edges: a factor's entries start near this.
    pub(super) fn edge_ends(&self) -> usize {
        self.lists.iter().map(Vec::len).sum()
    }

    /// Virtual copies, not stored entries: AC2 orders by the multigraph's degree.
    pub(super) fn degree(&self, v: usize) -> u64 {
        self.lists[v]
            .iter()
            .map(|e| u64::from(e.copies.get()))
            .sum()
    }

    pub(super) fn neighbors(&self, v: usize) -> impl Iterator<Item = (I, C, T)> + '_ {
        self.lists[v].iter().map(|e| (e.to, e.copies, e.weight))
    }

    /// Drops `v`'s edges from both sides.
    #[inline(never)] // Inlined into its one call site, it measured +1.6% of build.
    pub(super) fn eliminate_vertex(&mut self, v: usize) {
        while let Some(edge) = self.lists[v].pop() {
            self.swap_remove(edge.to.index(), edge.mirror as usize);
        }
        if self.lists.len() > REUSED_UP_TO {
            self.lists[v] = Vec::new();
        }
    }

    pub(super) fn push_pair(&mut self, u: I, v: I, copies: C, weight: T) {
        let (mirror_u, mirror_v) = (self.lists[v.index()].len(), self.lists[u.index()].len());
        // Truncating a u32 mirror would corrupt removal, so this holds in release too.
        assert!(
            mirror_u.max(mirror_v) < u32::MAX as usize,
            "a list exceeds u32 edges"
        );
        let edge = |to, mirror| Edge {
            to,
            mirror: mirror as u32,
            copies,
            weight,
        };
        self.lists[u.index()].push(edge(v, mirror_u));
        self.lists[v.index()].push(edge(u, mirror_v));
    }

    /// Swap-removes `u`'s edge at `at`, repairing the moved edge's mirror.
    fn swap_remove(&mut self, u: usize, at: usize) {
        self.lists[u].swap_remove(at);
        if let Some(&moved) = self.lists[u].get(at) {
            self.lists[moved.to.index()][moved.mirror as usize].mirror = at as u32;
        }
    }
}
