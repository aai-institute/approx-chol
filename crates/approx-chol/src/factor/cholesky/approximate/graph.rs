mod lists;

use self::lists::Lists;
use crate::factor::clique_tree::{absorb, Copies, Degrees};
use crate::factor::sdd::Sdd;
use crate::factor::sign::Sign;
use crate::{Index, Real};

/// Longest list [`EliminationGraph::push_fill`] searches: banded matrices need 8, 32 measured the same, and every search costs fill-heavy bipartite graphs build time.
const MERGED_UP_TO: u32 = 8;

/// One half of an edge; `mirror` indexes the other half in `to`'s list, so removal is O(1), and a list moving whole keeps it.
#[derive(Clone, Copy, Debug)]
struct Edge<T, I, C> {
    to: I,
    mirror: u32,
    copies: C,
    weight: T,
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

    /// `m`'s vertices, then its ground when it has one, which the order pins.
    pub(super) fn load(&mut self, m: &Sdd<T, I>, split: C) {
        // One slot past the degree: the first fill edge reallocating measured +2-3% on grids.
        let mut room = vec![1u32; m.n() + usize::from(m.ground().is_some())];
        // Saturated, a list past u32 slots fails in its growth, not as a wrapped room.
        m.grounded_edges(|k, j, _| {
            room[k.index()] = room[k.index()].saturating_add(1);
            room[j.index()] = room[j.index()].saturating_add(1);
        });
        self.reset(&room, split);
        m.grounded_edges(|k, j, w| self.push_pair(k, j, split, w));
    }

    /// Vertex `v` edgeless, with room for `room[v]` edges.
    pub(super) fn reset(&mut self, room: &[u32], split: C) {
        let unused = Edge {
            to: I::from_index(0),
            mirror: 0,
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
            self.degrees.remove(edge.to.index(), edge.copies.get());
            self.swap_remove(edge.to.index(), edge.mirror as usize);
        }
        self.lists.release(v);
        self.degrees.release(v);
    }

    pub(super) fn push_pair(&mut self, u: I, v: I, copies: C, weight: T) {
        let (len_u, len_v) = (self.lists.len(u.index()), self.lists.len(v.index()));
        self.push_at(u, v, (len_u, len_v), copies, weight);
    }

    /// `u — v` appended to lists of `lens`, each half's mirror the other list's length.
    fn push_at(&mut self, u: I, v: I, (mirror_v, mirror_u): (u32, u32), copies: C, weight: T) {
        // A list's length is a u32 that its growth keeps in range, so a mirror never truncates.
        let edge = |to, mirror| Edge {
            to,
            mirror,
            copies,
            weight,
        };
        self.lists.push(u.index(), edge(v, mirror_u));
        self.lists.push(v.index(), edge(u, mirror_v));
        self.degrees.add(u.index(), copies.get());
        self.degrees.add(v.index(), copies.get());
    }

    /// Absorbed into an `u — v` pair already held, so the degree counts it once, where the shorter list is searchable; past that the star's merge absorbs the duplicate.
    #[inline(always)] // Out of line or left to rustc, the pivot loop measured +2% of a bipartite build.
    pub(super) fn push_fill<S: Sign<T, I>>(
        &mut self,
        (u, v, copies, weight): (I, I, C, T),
        split: C,
        sign: S,
    ) {
        let lens = (self.lists.len(u.index()), self.lists.len(v.index()));
        // Read once: past the search, a second read of both lengths measured +2% of a bipartite build.
        if lens.0.min(lens.1) > MERGED_UP_TO {
            return self.push_at(u, v, lens, copies, weight);
        }
        let (a, b) = if lens.0 <= lens.1 { (u, v) } else { (v, u) };
        let held = (self.lists.get(a.index()).iter()).position(|edge| edge.to == b);
        let Some(at) = held else {
            return self.push_at(u, v, lens, copies, weight);
        };
        let edge = self.lists.get(a.index())[at];
        let (mut sum, mut merged) = (edge.copies, edge.weight);
        let cancelled = absorb(sign, split, (&mut sum, &mut merged), (copies, weight));
        for end in [a, b] {
            self.degrees.remove(end.index(), edge.copies.get());
        }
        if merged == T::zero() {
            self.swap_remove(b.index(), edge.mirror as usize);
            self.swap_remove(a.index(), at);
        } else {
            for end in [a, b] {
                self.degrees.add(end.index(), sum.get());
            }
            let half = &mut self.lists.get_mut(a.index())[at];
            (half.copies, half.weight) = (sum, merged);
            let half = &mut self.lists.get_mut(b.index())[edge.mirror as usize];
            (half.copies, half.weight) = (sum, merged);
        }
        if let Some(ground) = sign.ground().filter(|_| cancelled != T::zero()) {
            self.ground_both((a, b), ground, cancelled, split, sign);
        }
    }

    /// `cancelled` grounded at both `ends`, as the star grounds a merge's; out of line, so `push_fill` inlines without its recursion.
    #[inline(never)]
    fn ground_both<S: Sign<T, I>>(
        &mut self,
        ends: (I, I),
        ground: I,
        cancelled: T,
        split: C,
        sign: S,
    ) {
        for end in [ends.0, ends.1] {
            self.push_fill((end, ground, split, cancelled), split, sign);
        }
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

    use super::EliminationGraph;
    use crate::factor::cholesky::approximate::REUSED_UP_TO;
    use crate::factor::clique_tree::{Copies, Multi, Single};
    use crate::factor::sign::{Signed, Unsigned};

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

        /// A vertex's degree is its list's copies, through pushes and eliminations, AC and AC2.
        #[test]
        fn degrees_sum_copies(ops in vec((0..N, 0..N, 0u8..8, 1u32..5), 0..300)) {
            degrees_track_lists(Single, &ops)?;
            degrees_track_lists(Multi::split(4).expect("AC2"), &ops)?;
        }

        /// Lists never past `N - 1 < MERGED_UP_TO`, so every fill pushed merges: one entry a pair, its weights summed, its copies capped, the degree their sum.
        #[test]
        fn short_lists_merge_fill(pushes in vec((0..N, 0..N, 1u32..5, 1u8..9), 0..200)) {
            let split = Multi::split(4).expect("AC2");
            let mut graph = EliminationGraph::<f64, u32, Multi>::new();
            graph.reset(&[1; N], split);
            let mut want = std::collections::HashMap::new();
            for &(u, v, copies, weight) in pushes.iter().filter(|p| p.0 != p.1) {
                let edge = (u.min(v) as u32, u.max(v) as u32, split.cap(copies), f64::from(weight));
                graph.push_fill(edge, split, Unsigned);
                let (sum, total) = want.entry((u.min(v), u.max(v))).or_insert((0, 0.0));
                (*sum, *total) = (*sum + copies, *total + f64::from(weight));
            }
            graph.lists.assert_consistent();
            for x in 0..N {
                let have: Vec<_> = graph.neighbors(x).map(|(to, c, w)| (to as usize, c.get(), w)).collect();
                let mut ends: Vec<_> = have.iter().map(|h| h.0).collect();
                ends.sort_unstable();
                ends.dedup();
                prop_assert_eq!(ends.len(), have.len());
                for (to, copies, weight) in have {
                    let (sum, total) = want[&(x.min(to), x.max(to))];
                    prop_assert_eq!((copies, weight), (sum.min(4), total));
                }
                let copies: u64 = graph.neighbors(x).map(|(_, c, _)| u64::from(c.get())).sum();
                prop_assert_eq!(graph.degree(x), copies);
            }
        }
    }

    /// Opposite duplicates cancel twice the smaller magnitude onto the ground at both ends, and a pair cancelled to zero leaves both lists.
    #[test]
    fn signed_fill_cancels_onto_the_ground() {
        let sign = Signed { ground: 3u32 };
        let mut graph = EliminationGraph::<f64, u32, Single>::new();
        graph.reset(&[2; 4], Single);
        graph.push_fill((0, 1, Single, 2.0), Single, sign);
        graph.push_fill((0, 1, Single, -0.5), Single, sign);
        let lists = |graph: &EliminationGraph<f64, u32, Single>, x| {
            graph
                .neighbors(x)
                .map(|(to, _, w)| (to, w))
                .collect::<Vec<_>>()
        };
        assert_eq!(lists(&graph, 0), [(1, 1.5), (3, 1.0)]);
        assert_eq!(lists(&graph, 1), [(0, 1.5), (3, 1.0)]);
        graph.push_fill((0, 1, Single, -1.5), Single, sign);
        assert_eq!(lists(&graph, 0), [(3, 4.0)]);
        assert_eq!(lists(&graph, 1), [(3, 4.0)]);
        graph.lists.assert_consistent();
    }

    fn degrees_track_lists<C: Copies>(
        split: C,
        ops: &[(usize, usize, u8, u32)],
    ) -> Result<(), TestCaseError> {
        let mut graph = EliminationGraph::<f64, u32, C>::new();
        let mut room = vec![0; REUSED_UP_TO + 1];
        room[..N].fill(1);
        graph.reset(&room, split);
        let mut alive = [true; N];
        for &(u, v, eliminate, copies) in ops {
            if eliminate == 0 && alive[u] {
                graph.eliminate_vertex(u);
                alive[u] = false;
            } else if u != v && alive[u] && alive[v] {
                graph.push_pair(u as u32, v as u32, split.cap(copies), 1.0);
            }
            for x in 0..N {
                let copies: u64 = graph.neighbors(x).map(|(_, c, _)| u64::from(c.get())).sum();
                prop_assert_eq!(graph.degree(x), copies);
            }
        }
        Ok(())
    }
}
