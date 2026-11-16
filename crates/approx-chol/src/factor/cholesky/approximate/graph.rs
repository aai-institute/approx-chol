mod lists;

use self::lists::Lists;
use super::Remaining;
use crate::factor::clique_tree::{Copies, Degrees};
use crate::factor::sdd::Sdd;
use crate::{Index, Real};

/// One half of an edge; `link` indexes the other half in `to`'s list, so removal is O(1), and a list moving whole keeps it.
#[derive(Clone, Copy, Debug)]
struct Edge<T, I, C> {
    to: I,
    /// The mirror below [`MIRRORS`], then [`DROPPED_UP_TO`]'s bits: a 1-byte field rounds the edge to 24 bytes.
    link: u32,
    copies: C,
    weight: T,
}

/// A list's length bound: its mirrors leave `link` two bits.
const MIRRORS: u32 = 1 << 30;

/// Copies a merge's cap dropped, which the degree still counts: phase-1 fixest PCG iterations 17 → 50 without, ≈ at this cap.
pub(super) const DROPPED_UP_TO: u32 = 3;

/// `mirror` below [`MIRRORS`], `dropped` above.
fn link(mirror: u32, dropped: u32) -> u32 {
    mirror + dropped * MIRRORS
}

impl<T, I, C: Copies> Edge<T, I, C> {
    /// The copies the degree counts.
    fn stood(&self) -> u32 {
        self.copies.get() + self.dropped()
    }

    fn mirror(&self) -> usize {
        (self.link % MIRRORS) as usize
    }

    fn dropped(&self) -> u32 {
        self.link / MIRRORS
    }
}

/// The mutable graph minimum-degree eliminates; a grounded one carries its ground as the last vertex.
pub(super) struct EliminationGraph<T, I, C: Copies> {
    lists: Lists<Edge<T, I, C>, I>,
    degrees: C::Degrees,
}

impl<T: Real, I: Index, C: Copies> EliminationGraph<T, I, C> {
    pub(super) fn new() -> Self {
        Self {
            lists: Lists::new(),
            degrees: C::Degrees::default(),
        }
    }

    /// Each of `n` vertices' edges in `m`, `m`'s then its ground's, plus one slot: the first fill edge reallocating measured +2-3% on grids.
    pub(super) fn room(m: &Sdd<T, I>, n: usize) -> Vec<u32> {
        let mut room = vec![1u32; n];
        // Saturated, a list past u32 slots fails in its growth, not as a wrapped room.
        m.grounded_edges(|k, j, _| {
            room[k.index()] = room[k.index()].saturating_add(1);
            room[j.index()] = room[j.index()].saturating_add(1);
        });
        room
    }

    /// `m`'s edges, in the [`room`](Self::room) it counted; a ground past them the order pins.
    pub(super) fn load(&mut self, m: &Sdd<T, I>, room: &[u32], split: C) {
        self.reset(room, split);
        m.grounded_edges(|k, j, w| self.push_pair(k, j, split, w));
    }

    /// `edges`, each `(u, v, weight, copies)` summed before the cap; a vertex `remaining` marks gone gets no room, the others one slot past their edges.
    pub(super) fn load_merged(
        &mut self,
        edges: &[(I, I, T, u32)],
        remaining: &Remaining,
        split: C,
    ) {
        let n = remaining.ids.len();
        let mut room: Vec<u32> = (0..n).map(|v| u32::from(!remaining.gone(v))).collect();
        for &(k, j, ..) in edges {
            room[k.index()] = room[k.index()].saturating_add(1);
            room[j.index()] = room[j.index()].saturating_add(1);
        }
        self.reset(&room, split);
        drop(room);
        for &(k, j, weight, merged) in edges {
            let copies = split.cap(merged);
            let dropped = (merged - copies.get()).min(DROPPED_UP_TO);
            self.push(k, j, copies, dropped, weight);
        }
    }

    /// Vertex `v` edgeless, with room for `room[v]` edges.
    pub(super) fn reset(&mut self, room: &[u32], split: C) {
        let unused = Edge {
            to: I::from_index(0),
            link: 0,
            copies: split,
            weight: T::zero(),
        };
        self.lists.load(room, unused);
        self.degrees.load(room.len());
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
        self.degrees.get(v, self.lists.len(v))
    }

    pub(super) fn neighbors(&self, v: usize) -> impl Iterator<Item = (I, C, T)> + '_ {
        self.lists.get(v).iter().map(|e| (e.to, e.copies, e.weight))
    }

    /// Drops `v`'s edges from both sides.
    #[inline(never)] // Inlined into its one call site, it measured +1.6% of build.
    pub(super) fn eliminate_vertex(&mut self, v: usize) {
        while let Some(edge) = self.lists.pop(v) {
            self.degrees.remove(edge.to.index(), edge.stood());
            self.swap_remove(edge.to.index(), edge.mirror());
        }
        self.lists.release(v);
        self.degrees.release(v);
    }

    pub(super) fn push_pair(&mut self, u: I, v: I, copies: C, weight: T) {
        self.push(u, v, copies, 0, weight);
    }

    fn push(&mut self, u: I, v: I, copies: C, dropped: u32, weight: T) {
        let (mirror_u, mirror_v) = (self.lists.len(v.index()), self.lists.len(u.index()));
        assert!(
            mirror_u.max(mirror_v) < MIRRORS,
            "a list exceeds 2^30 edges"
        );
        let edge = |to, mirror| Edge {
            to,
            link: link(mirror, dropped),
            copies,
            weight,
        };
        let half = edge(v, mirror_u);
        let stood = half.stood();
        self.lists.push(u.index(), half);
        self.lists.push(v.index(), edge(u, mirror_v));
        self.degrees.add(u.index(), stood);
        self.degrees.add(v.index(), stood);
    }

    /// Swap-removes `u`'s edge at `at`, repairing the moved edge's mirror.
    fn swap_remove(&mut self, u: usize, at: usize) {
        if let Some(moved) = self.lists.swap_remove(u, at) {
            let mirror = &mut self.lists.get_mut(moved.to.index())[moved.mirror()];
            mirror.link = link(at as u32, mirror.dropped());
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::collection::vec;
    use proptest::prelude::*;

    use super::{EliminationGraph, DROPPED_UP_TO};
    use crate::factor::cholesky::approximate::REUSED_UP_TO;
    use crate::factor::clique_tree::{Copies, Multi, Single};

    const N: usize = 8;

    proptest! {
        /// Pairs pushed and vertices eliminated leave each mirror pointing back, through moves, in-place growth and compaction.
        #[test]
        fn mirrors_point_back(ops in vec((0..N, 0..N, 0u8..8), 0..300)) {
            let mut graph = EliminationGraph::<f64, u32, Single>::new();
            // Past REUSED_UP_TO, so eliminations compact; only the first N vertices are used.
            let mut room = vec![0; REUSED_UP_TO + 1];
            room[..N].fill(1);
            graph.reset(&room, Single);
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
                        let back = graph.lists.get(edge.to as usize)[edge.mirror()];
                        prop_assert_eq!((back.to as usize, back.mirror()), (x, at));
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

        /// A vertex's degree is its edges' copies and dropped copies, through pushes, moves and eliminations, AC and AC2.
        #[test]
        fn degrees_sum_copies(ops in vec((0..N, 0..N, 0u8..8, 1u32..5, 0..=DROPPED_UP_TO), 0..300)) {
            degrees_track_lists(Single, &ops)?;
            degrees_track_lists(Multi::split(4).expect("AC2"), &ops)?;
        }
    }

    fn degrees_track_lists<C: Copies>(
        split: C,
        ops: &[(usize, usize, u8, u32, u32)],
    ) -> Result<(), TestCaseError> {
        let mut graph = EliminationGraph::<f64, u32, C>::new();
        let mut room = vec![0; REUSED_UP_TO + 1];
        room[..N].fill(1);
        graph.reset(&room, split);
        let (mut alive, mut edges) = ([true; N], Vec::new());
        for &(u, v, eliminate, copies, dropped) in ops {
            if eliminate == 0 && alive[u] {
                graph.eliminate_vertex(u);
                alive[u] = false;
                edges.retain(|&(a, b, _)| a != u && b != u);
            } else if u != v && alive[u] && alive[v] {
                let copies = split.cap(copies);
                graph.push(u as u32, v as u32, copies, dropped, 1.0);
                edges.push((u, v, u64::from(copies.get() + dropped)));
            }
            for x in 0..N {
                let stood: u64 = edges
                    .iter()
                    .filter(|&&(a, b, _)| a == x || b == x)
                    .map(|e| e.2)
                    .sum();
                prop_assert_eq!(graph.degree(x), stood);
            }
        }
        Ok(())
    }
}
