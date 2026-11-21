mod lists;

use self::lists::Lists;
use crate::factor::clique_tree::{absorb, Copies, Degrees, Pair};
use crate::factor::sdd::Sdd;
use crate::factor::sign::Sign;
use crate::{Index, Real};

/// Whether a pivot's fill is searched for in the shorter of its ends' lists, so a duplicate merges into its pair and the degrees count it once.
///
/// A search costs that list's length; an unmerged duplicate costs only a degree too high, which matters where it moves the order. Banded and fixest fill nearly always lands on a held pair (96-100%), the fronts it inflates then lose the order to fresh vertices (banded 55 PCG iterations instead of 14, fixest 5M 39 instead of 17); power-law and person × firm fill rarely does (2-20%) until the remainder is nearly complete, where searches cost their builds up to 4x.
/// The ground edges a signed cancellation owes are the pivot's fill too, so they search under, and count toward, the same record.
pub(super) struct FillSearch {
    /// The pivot's stored entries; a list within twice them belongs to a vertex near the order's minimum.
    pivot_entries: u32,
    /// Its searches so far that found the pair, and that did not.
    found: u32,
    missed: u32,
    /// More pivots left than four times its neighbors; at most that, the remainder is nearly complete and every order of it fills alike (2-4 measured alike, 8 cost bands 0.5-1.5 PCG iterations).
    order_matters: bool,
}

impl FillSearch {
    /// For the fill of a pivot holding `entries`, `neighbors` distinct, with `remaining` pivots left after it.
    pub(super) fn new(entries: u32, neighbors: usize, remaining: usize) -> Self {
        Self {
            pivot_entries: entries,
            found: 0,
            missed: 0,
            order_matters: remaining > 4 * neighbors,
        }
    }

    /// Whether a pair whose shorter list holds `shorter` entries is searched for.
    fn worth(&self, shorter: u32) -> bool {
        // Could its ends be pivoted soon? A degree too high would pass them over.
        let competitive = shorter <= self.pivot_entries.saturating_mul(2);
        // Does this pivot's fill keep landing on held pairs? Allowing one miss, fixest took 40 PCG iterations instead of 17; 2-4 measured alike.
        let paying_off = self.missed < 3 + 2 * self.found;
        // Does the order still matter? Not once the remainder is nearly complete.
        competitive && paying_off && self.order_matters
    }

    fn record(&mut self, found: bool) {
        if found {
            self.found += 1;
        } else {
            self.missed += 1;
        }
    }
}

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

    /// Each of `m`'s vertices, then its ground's, its edges plus one slot: the first fill edge reallocating measured +2-3% on grids.
    pub(super) fn room(m: &Sdd<T, I>) -> Vec<u32> {
        let mut room = vec![1u32; m.n() + usize::from(m.ground().is_some())];
        // Saturated, a list past u32 slots fails in its growth, not as a wrapped room.
        m.grounded_edges(|k, j, _| {
            room[k.index()] = room[k.index()].saturating_add(1);
            room[j.index()] = room[j.index()].saturating_add(1);
        });
        room
    }

    /// `m`'s edges in the [`room`](Self::room) it counted; its ground the order pins.
    pub(super) fn load(&mut self, m: &Sdd<T, I>, room: &[u32], split: C) {
        self.reset(room, split);
        m.grounded_edges(|k, j, w| self.push_pair(k, j, split, w));
    }

    /// `pairs`, each once with its capped copies; a vertex `gone` gets no room.
    pub(super) fn load_merged(
        &mut self,
        pairs: &[Pair<T, I, C>],
        gone: impl Fn(usize) -> bool,
        n: usize,
        split: C,
    ) {
        let mut room: Vec<u32> = (0..n).map(|v| u32::from(!gone(v))).collect();
        for &(k, j, _, _) in pairs {
            room[k.index()] = room[k.index()].saturating_add(1);
            room[j.index()] = room[j.index()].saturating_add(1);
        }
        self.reset(&room, split);
        drop(room);
        for &(k, j, copies, weight) in pairs {
            debug_assert!(copies.get() <= split.get(), "a merge caps every pair");
            self.push_pair(k, j, copies, weight);
        }
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

    pub(super) fn neighbors(&self, v: usize) -> impl ExactSizeIterator<Item = (I, C, T)> + '_ {
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
    #[inline(always)] // A merged load's second caller left it out of line: AC2 bipartite builds +2.5%.
    fn push_at(&mut self, u: I, v: I, (len_u, len_v): (u32, u32), copies: C, weight: T) {
        // A list's length is a u32 that its growth keeps in range, so a mirror never truncates.
        let edge = |to, mirror| Edge {
            to,
            mirror,
            copies,
            weight,
        };
        self.lists.push(u.index(), edge(v, len_v));
        self.lists.push(v.index(), edge(u, len_u));
        self.degrees.add(u.index(), copies.get());
        self.degrees.add(v.index(), copies.get());
    }

    /// Absorbed into an `u — v` pair already held, so the degree counts it once, where `search` finds it worth a search; otherwise the star's merge absorbs the duplicate.
    #[inline(always)] // Out of line or left to rustc, the pivot loop measured +2% of a bipartite build.
    pub(super) fn push_fill<S: Sign<T, I>>(
        &mut self,
        (u, v, copies, weight): Pair<T, I, C>,
        search: &mut FillSearch,
        split: C,
        sign: S,
    ) {
        let lens = (self.lists.len(u.index()), self.lists.len(v.index()));
        // Read once: past the search, a second read of both lengths measured +2% of a bipartite build.
        if !search.worth(lens.0.min(lens.1)) {
            return self.push_at(u, v, lens, copies, weight);
        }
        let (a, b) = if lens.0 <= lens.1 { (u, v) } else { (v, u) };
        let held = (self.lists.get(a.index()).iter()).position(|edge| edge.to == b);
        search.record(held.is_some());
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
            self.ground_both((a, b), ground, cancelled, search, split, sign);
        }
    }

    /// `cancelled` grounded at both `ends`, as the star grounds a merge's; out of line, so `push_fill` inlines without its recursion.
    #[inline(never)]
    fn ground_both<S: Sign<T, I>>(
        &mut self,
        ends: (I, I),
        ground: I,
        cancelled: T,
        search: &mut FillSearch,
        split: C,
        sign: S,
    ) {
        for end in [ends.0, ends.1] {
            self.push_fill((end, ground, split, cancelled), search, split, sign);
        }
    }

    /// Swap-removes `u`'s edge at `at`, repairing the moved edge's mirror.
    #[inline(always)] // Out of line once the merged load added code, as `push_at`.
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

    use super::{EliminationGraph, FillSearch};
    use crate::factor::cholesky::approximate::REUSED_UP_TO;
    use crate::factor::clique_tree::{Copies, Multi, Single};
    use crate::factor::sign::{Signed, Unsigned};

    const N: usize = 8;

    /// A fresh pivot's search, every list within reach and the remainder sparse: it searches until it misses thrice.
    fn searching() -> FillSearch {
        FillSearch::new(N as u32, 1, usize::MAX)
    }

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

        /// Each fill pushed as a fresh pivot's, so every one is searched and merges: one entry a pair, its weights summed, its copies capped, the degree their sum.
        #[test]
        fn short_lists_merge_fill(pushes in vec((0..N, 0..N, 1u32..5, 1u8..9), 0..200)) {
            let split = Multi::split(4).expect("AC2");
            let mut graph = EliminationGraph::<f64, u32, Multi>::new();
            graph.reset(&[1; N], split);
            let mut want = std::collections::HashMap::new();
            for &(u, v, copies, weight) in pushes.iter().filter(|p| p.0 != p.1) {
                let edge = (u.min(v) as u32, u.max(v) as u32, split.cap(copies), f64::from(weight));
                graph.push_fill(edge, &mut searching(), split, Unsigned);
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

        /// Signed AC2 fill, cancelling partly or to zero, among eliminations: each degree stays its list's copies, the ground's too.
        #[test]
        fn signed_fill_keeps_degrees(ops in vec((0..N, 0..N, 0u8..8, 1u32..5, -3i8..4), 0..300)) {
            let split = Multi::split(4).expect("AC2");
            let sign = Signed { ground: N as u32 };
            let mut graph = EliminationGraph::<f64, u32, Multi>::new();
            graph.reset(&[1; N + 1], split);
            let mut alive = [true; N];
            for (u, v, eliminate, copies, weight) in ops {
                if eliminate == 0 && alive[u] {
                    graph.eliminate_vertex(u);
                    alive[u] = false;
                } else if u != v && alive[u] && alive[v] && weight != 0 {
                    let edge = (u.min(v) as u32, u.max(v) as u32, split.cap(copies), f64::from(weight));
                    graph.push_fill(edge, &mut searching(), split, sign);
                }
                graph.lists.assert_consistent();
                for x in 0..=N {
                    let copies: u64 = graph.neighbors(x).map(|(_, c, _)| u64::from(c.get())).sum();
                    prop_assert_eq!(graph.degree(x), copies);
                }
            }
        }
    }

    /// A search is made while the shorter list is within twice the pivot's entries, its misses stay below three plus twice its hits, and the remainder is sparse.
    #[test]
    fn fill_search_asks_three_questions() {
        let search = FillSearch::new(4, 4, 100);
        assert!(search.worth(8) && !search.worth(9));
        let mut search = FillSearch::new(4, 4, 100);
        for _ in 0..3 {
            assert!(search.worth(1));
            search.record(false);
        }
        assert!(!search.worth(1));
        let mut search = FillSearch::new(4, 4, 100);
        search.record(true);
        for _ in 0..5 {
            assert!(search.worth(1));
            search.record(false);
        }
        assert!(!search.worth(1));
        assert!(!FillSearch::new(4, 4, 16).worth(1) && FillSearch::new(4, 4, 17).worth(1));
    }

    /// A searched pair merges and the degree counts it once; past the pivot's reach, a duplicate stands beside it and the degree counts both.
    #[test]
    fn fill_merges_within_reach() {
        let mut graph = EliminationGraph::<f64, u32, Single>::new();
        graph.reset(&[1; 24], Single);
        graph.push_pair(0, 1, Single, 1.0);
        for k in 0..7 {
            graph.push_pair(0, 2 + k, Single, 1.0);
            graph.push_pair(1, 9 + k, Single, 1.0);
        }
        let to_one = |graph: &EliminationGraph<f64, u32, Single>| {
            graph.neighbors(0).filter(|&(to, _, _)| to == 1).count()
        };
        graph.push_fill(
            (0, 1, Single, 1.0),
            &mut FillSearch::new(4, 1, 24),
            Single,
            Unsigned,
        );
        assert_eq!((to_one(&graph), graph.degree(0)), (1, 8));
        graph.push_fill(
            (0, 1, Single, 1.0),
            &mut FillSearch::new(3, 1, 24),
            Single,
            Unsigned,
        );
        assert_eq!((to_one(&graph), graph.degree(0)), (2, 9));
        graph.lists.assert_consistent();
    }

    /// A merged pair loads as one entry, its weight whole and its copies capped; the degree counts them, AC and AC2, also once a neighbor goes.
    #[test]
    fn a_merged_pair_loads_as_one_capped_entry() {
        fn check<C: Copies>(split: C) {
            let pairs: Vec<_> = (1..9u32)
                .map(|neighbor| (0, neighbor, split.cap(neighbor), f64::from(neighbor)))
                .collect();
            let mut graph = EliminationGraph::<f64, u32, C>::new();
            graph.load_merged(&pairs, |_| false, 9, split);
            graph.lists.assert_consistent();
            for neighbor in 1..9u32 {
                let entries: Vec<_> = graph.neighbors(neighbor as usize).collect();
                assert_eq!(entries.len(), 1, "{neighbor}");
                assert_eq!(
                    (entries[0].1.get(), entries[0].2),
                    (split.cap(neighbor).get(), f64::from(neighbor))
                );
            }
            let capped = |neighbor: u32| u64::from(split.cap(neighbor).get());
            let want: u64 = (1..9).map(capped).sum();
            assert_eq!(graph.degree(0), want);
            graph.eliminate_vertex(8);
            assert_eq!(graph.degree(0), want - capped(8));
        }
        check(Single);
        check(Multi::split(4).expect("AC2"));
    }

    /// Opposite duplicates cancel twice the smaller magnitude onto the ground at both ends, and a pair cancelled to zero leaves both lists; the owed ground edges search under the pivot's one record.
    #[test]
    fn signed_fill_cancels_onto_the_ground() {
        let sign = Signed { ground: 3u32 };
        let mut graph = EliminationGraph::<f64, u32, Single>::new();
        graph.reset(&[2; 4], Single);
        let mut search = searching();
        graph.push_fill((0, 1, Single, 2.0), &mut search, Single, sign);
        graph.push_fill((0, 1, Single, -0.5), &mut search, Single, sign);
        let lists = |graph: &EliminationGraph<f64, u32, Single>, x| {
            graph
                .neighbors(x)
                .map(|(to, _, w)| (to, w))
                .collect::<Vec<_>>()
        };
        assert_eq!(lists(&graph, 0), [(1, 1.5), (3, 1.0)]);
        assert_eq!(lists(&graph, 1), [(0, 1.5), (3, 1.0)]);
        graph.push_fill((0, 1, Single, -1.5), &mut search, Single, sign);
        assert_eq!(lists(&graph, 0), [(3, 4.0)]);
        assert_eq!(lists(&graph, 1), [(3, 4.0)]);
        // One record: each push and owed ground edge searched; misses on the new pairs (0, 1), (0, 3), (1, 3), hits after.
        assert_eq!((search.found, search.missed), (4, 3));
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
