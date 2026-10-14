use crate::factor::clique_tree::Copies;
use crate::factor::sddm::Sddm;
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
        let n = sddm.n() + usize::from(sddm.grounded());
        // One slot past the degree: the first fill edge reallocating measured +2-3% on grids.
        let mut capacities = vec![1; n];
        for (k, j, _) in sddm.grounded_edges() {
            capacities[k as usize] += 1;
            capacities[j as usize] += 1;
        }
        let lists = capacities.into_iter().map(Vec::with_capacity).collect();
        let mut graph = Self { lists };
        let copies = C::initial(split);
        for (k, j, w) in sddm.grounded_edges() {
            graph.push_pair(k, j, copies, w);
        }
        graph
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

    pub(super) fn neighbors(&self, v: usize) -> impl Iterator<Item = (u32, C, T)> + '_ {
        self.lists[v].iter().map(|e| (e.to, e.copies, e.weight))
    }

    /// Drops `v`'s edges from both sides.
    #[inline(never)] // Inlined into its one call site, it measured +1.6% of build.
    pub(super) fn eliminate_vertex(&mut self, v: usize) {
        while let Some(edge) = self.lists[v].pop() {
            self.swap_remove(edge.to as usize, edge.mirror as usize);
        }
        self.lists[v] = Vec::new();
    }

    pub(super) fn push_pair(&mut self, u: u32, v: u32, copies: C, weight: T) {
        let (mirror_u, mirror_v) = (self.lists[v as usize].len(), self.lists[u as usize].len());
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
        self.lists[u as usize].push(edge(v, mirror_u));
        self.lists[v as usize].push(edge(u, mirror_v));
    }

    /// Swap-removes `u`'s edge at `at`, repairing the moved edge's mirror.
    fn swap_remove(&mut self, u: usize, at: usize) {
        self.lists[u].swap_remove(at);
        if let Some(&moved) = self.lists[u].get(at) {
            self.lists[moved.to as usize][moved.mirror as usize].mirror = at as u32;
        }
    }
}
