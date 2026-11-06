mod lists;

use self::lists::Lists;
use crate::factor::clique_tree::Copies;
use crate::factor::sddm::Sddm;
use crate::{Index, Real};

/// One half of an edge; `mirror` indexes the other half in `to`'s list, so removal is O(1), and a list moving whole keeps it.
#[derive(Clone, Copy, Debug)]
struct Edge<T, I, C> {
    to: I,
    mirror: u32,
    copies: C,
    weight: T,
}

/// The mutable graph minimum-degree eliminates; a grounded one carries its ground as the last vertex.
pub(super) struct EliminationGraph<T, I, C> {
    lists: Lists<Edge<T, I, C>, I>,
}

impl<T: Real, I: Index, C: Copies> EliminationGraph<T, I, C> {
    pub(super) fn new() -> Self {
        Self {
            lists: Lists::new(),
        }
    }

    /// The ground, when there is one, is the last vertex; the order pins it.
    pub(super) fn load(&mut self, sddm: &Sddm<T, I>, split: C) {
        let n = sddm.n() + usize::from(sddm.grounded());
        // One slot past the degree: the first fill edge reallocating measured +2-3% on grids.
        let mut room = vec![1u32; n];
        // Saturated, a list past u32 slots fails in its growth, not as a wrapped room.
        sddm.grounded_edges(|k, j, _| {
            room[k.index()] = room[k.index()].saturating_add(1);
            room[j.index()] = room[j.index()].saturating_add(1);
        });
        let unused = Edge {
            to: I::from_index(0),
            mirror: 0,
            copies: split,
            weight: T::zero(),
        };
        self.lists.load(&room, unused);
        sddm.grounded_edges(|k, j, w| self.push_pair(k, j, split, w));
    }

    pub(super) fn n(&self) -> usize {
        self.lists.n()
    }

    /// Twice the edges: a factor's entries start near this.
    pub(super) fn edge_ends(&self) -> usize {
        (0..self.n()).map(|v| self.lists.len(v) as usize).sum()
    }

    /// Virtual copies, not stored entries: AC2 orders by the multigraph's degree.
    pub(super) fn degree(&self, v: usize) -> u64 {
        self.lists
            .get(v)
            .iter()
            .map(|e| u64::from(e.copies.get()))
            .sum()
    }

    pub(super) fn neighbors(&self, v: usize) -> impl Iterator<Item = (I, C, T)> + '_ {
        self.lists.get(v).iter().map(|e| (e.to, e.copies, e.weight))
    }

    /// Drops `v`'s edges from both sides.
    #[inline(never)] // Inlined into its one call site, it measured +1.6% of build.
    pub(super) fn eliminate_vertex(&mut self, v: usize) {
        while let Some(edge) = self.lists.pop(v) {
            self.swap_remove(edge.to.index(), edge.mirror as usize);
        }
        self.lists.release(v);
    }

    pub(super) fn push_pair(&mut self, u: I, v: I, copies: C, weight: T) {
        // A list's length is a u32 that its growth keeps in range, so a mirror never truncates.
        let (mirror_u, mirror_v) = (self.lists.len(v.index()), self.lists.len(u.index()));
        let edge = |to, mirror| Edge {
            to,
            mirror,
            copies,
            weight,
        };
        self.lists.push(u.index(), edge(v, mirror_u));
        self.lists.push(v.index(), edge(u, mirror_v));
    }

    /// Swap-removes `u`'s edge at `at`, repairing the moved edge's mirror.
    fn swap_remove(&mut self, u: usize, at: usize) {
        if let Some(moved) = self.lists.swap_remove(u, at) {
            self.lists.get_mut(moved.to.index())[moved.mirror as usize].mirror = at as u32;
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::collection::vec;
    use proptest::prelude::*;

    use super::{Edge, EliminationGraph};
    use crate::factor::cholesky::approximate::REUSED_UP_TO;
    use crate::factor::clique_tree::Single;

    const N: usize = 8;

    proptest! {
        /// Pairs pushed and vertices eliminated leave each mirror pointing back, through moves, in-place growth and compaction.
        #[test]
        fn mirrors_point_back(ops in vec((0..N, 0..N, 0u8..8), 0..300)) {
            let mut graph = EliminationGraph::<f64, u32, Single>::new();
            // Past REUSED_UP_TO, so eliminations compact; only the first N vertices are used.
            let mut room = vec![0; REUSED_UP_TO + 1];
            room[..N].fill(1);
            graph.lists.load(&room, Edge { to: 0, mirror: 0, copies: Single, weight: 0.0 });
            let (mut alive, mut edges) = ([true; N], Vec::new());
            for (id, (u, v, eliminate)) in ops.into_iter().enumerate() {
                if eliminate == 0 && alive[u] {
                    graph.eliminate_vertex(u);
                    alive[u] = false;
                    edges.retain(|&(a, b, _)| a != u && b != u);
                } else if u != v && alive[u] && alive[v] {
                    graph.push_pair(u as u32, v as u32, Single, id as f64);
                    edges.push((u, v, id as f64));
                }
                graph.lists.assert_consistent();
                for x in 0..N {
                    for (at, edge) in graph.lists.get(x).iter().enumerate() {
                        let back = graph.lists.get(edge.to as usize)[edge.mirror as usize];
                        prop_assert_eq!((back.to as usize, back.mirror as usize), (x, at));
                    }
                    let mut have: Vec<_> = graph.neighbors(x).map(|(to, _, w)| (to as usize, w)).collect();
                    let mut want: Vec<_> = (edges.iter())
                        .filter_map(|&(a, b, w)| (a == x).then_some((b, w)).or((b == x).then_some((a, w))))
                        .collect();
                    have.sort_by(|p, q| p.1.total_cmp(&q.1));
                    want.sort_by(|p, q| p.1.total_cmp(&q.1));
                    prop_assert_eq!(have, want);
                }
            }
        }
    }
}
