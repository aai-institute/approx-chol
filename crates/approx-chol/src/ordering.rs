//! Elimination ordering abstraction for approximate Cholesky factorization.
//!
//! Separates how the next vertex to eliminate is selected from the
//! elimination loop. Two implementations:
//! - `StaticOrdering`: pre-computed AMD ordering (notify methods are no-ops)
//! - `DynamicOrdering`: bucket-based priority queue that adapts to fill-in
//!   during elimination (ports Julia's `ApproxCholPQ`)

use super::graph::{EliminationGraph, Neighbor};
use crate::Real;

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

pub(crate) trait EliminationOrdering<T: Real> {
    /// Pick the next vertex to eliminate, or `None` if done.
    fn next_vertex(&mut self) -> Option<usize>;

    /// One live edge incident to `v` was removed (degree estimate decrements by 1).
    fn notify_neighbor_removed(&mut self, _v: u32) {}

    /// `n` live edges incident to `v` were removed.
    fn notify_neighbor_removed_n(&mut self, v: u32, n: u32) {
        for _ in 0..n {
            self.notify_neighbor_removed(v);
        }
    }

    /// A vertex was eliminated; its (uneliminated) neighbors may have changed degree.
    fn notify_eliminated(&mut self, _v: usize, neighbors: &[(u32, T)]) {
        for &(u, _) in neighbors {
            self.notify_neighbor_removed(u);
        }
    }

    /// A fill edge was added between `u` and `v`.
    fn notify_fill_edge(&mut self, u: u32, v: u32);

    /// Duplicate edges to `v` were merged during compression (degree decreased by 1).
    fn notify_edges_merged(&mut self, v: u32) {
        self.notify_neighbor_removed(v);
    }

    /// `n` duplicate edges to `v` were merged during compression.
    fn notify_edges_merged_n(&mut self, v: u32, n: u32) {
        self.notify_neighbor_removed_n(v, n);
    }
}

// ---------------------------------------------------------------------------
// StaticOrdering — extracted from the former `approximate_minimum_degree_order`
// ---------------------------------------------------------------------------

/// Pre-computed approximate minimum degree ordering. The order is fixed at
/// construction time; `notify_*` methods are no-ops.
pub struct StaticOrdering {
    order: Vec<usize>,
    pos: usize,
}

impl StaticOrdering {
    /// Build AMD ordering using a `DynamicOrdering` priority queue.
    ///
    /// On a pristine graph (no fill yet), the PQ adapts to degree changes
    /// from neighbor elimination, producing a true approximate minimum-degree
    /// ordering rather than a snapshot-based one.
    ///
    /// Returns the ordering and the total degree sum (for capacity hints).
    pub(crate) fn from_graph<T: Real, G: EliminationGraph<T>>(graph: &mut G) -> (Self, usize) {
        let n = graph.n();
        let degrees: Vec<usize> = (0..n).map(|v| graph.degree(v)).collect();
        let degree_sum: usize = degrees.iter().sum();
        let mut pq = DynamicOrdering::new(n, degrees.into_iter());
        let mut order = Vec::with_capacity(n);
        let mut scratch: Vec<Neighbor<T>> = Vec::new();

        while let Some(v) = pq.next_vertex() {
            order.push(v);
            graph.live_neighbors(v, &mut scratch);
            for nbr in &scratch {
                pq.notify_neighbor_removed_n(nbr.to, nbr.count);
            }
        }

        (StaticOrdering { order, pos: 0 }, degree_sum)
    }
}

impl<T: Real> EliminationOrdering<T> for StaticOrdering {
    fn next_vertex(&mut self) -> Option<usize> {
        if self.pos < self.order.len() {
            let v = self.order[self.pos];
            self.pos += 1;
            Some(v)
        } else {
            None
        }
    }

    #[inline]
    fn notify_neighbor_removed(&mut self, _v: u32) {}

    #[inline]
    fn notify_fill_edge(&mut self, _u: u32, _v: u32) {}
}

// ---------------------------------------------------------------------------
// DynamicOrdering — bucket-based priority queue (Julia `ApproxCholPQ` port)
// ---------------------------------------------------------------------------

/// Linked-list terminator: marks that there is no previous or next element in a bucket chain.
const SENTINEL: u32 = u32::MAX;

/// One element in the bucket priority queue, representing a single vertex.
///
/// `key == u32::MAX` means the element has been removed from the priority queue
/// (popped or logically deleted); operations like `inc`/`dec` skip removed elements.
struct PQElem {
    prev: u32, // SENTINEL = head of bucket list
    next: u32, // SENTINEL = tail of bucket list
    key: u32,  // current degree estimate
}

/// Bucket-based priority queue for dynamic minimum-degree ordering.
///
/// Vertices are distributed into buckets by their current degree estimate via
/// [`key_map`]. Each bucket is a doubly-linked list threaded through [`PQElem`].
/// `min_list` is a *lower bound* on the index of the minimum non-empty bucket;
/// [`pop`](Self::pop) scans upward from `min_list` to find the actual minimum.
///
/// Requires `n <= u32::MAX` (asserted at construction).
pub struct DynamicOrdering {
    elems: Vec<PQElem>, // indexed by vertex id
    lists: Vec<u32>,    // bucket heads, indexed by key_map(degree)
    min_list: usize,    // lower bound on minimum non-empty bucket
    n_items: usize,
    bucket_base: usize,
    bucket_upper: usize,
}

/// Map degree to bucket index.
///
/// Degrees <= `bucket_base` get individual buckets; higher degrees are grouped
/// via `bucket_base + degree / bucket_base`, capped at `bucket_upper`.
fn key_map(degree: usize, bucket_base: usize, bucket_upper: usize) -> usize {
    if degree <= bucket_base {
        degree
    } else {
        (bucket_base + degree / bucket_base).min(bucket_upper)
    }
}

impl DynamicOrdering {
    /// Pop the vertex with the minimum degree estimate.
    ///
    /// `min_list` is a lower bound; we scan upward until a non-empty bucket is found.
    fn pop(&mut self) -> Option<usize> {
        if self.n_items == 0 {
            return None;
        }
        while self.min_list < self.lists.len() && self.lists[self.min_list] == SENTINEL {
            self.min_list += 1;
        }
        if self.min_list >= self.lists.len() {
            return None;
        }
        let i = self.lists[self.min_list] as usize;
        let next = self.elems[i].next;
        self.lists[self.min_list] = next;
        if next != SENTINEL {
            self.elems[next as usize].prev = SENTINEL;
        }
        self.elems[i].key = u32::MAX; // mark as removed
        self.n_items -= 1;
        Some(i)
    }

    /// Move element `i` to the bucket for `new_key`, re-linking the doubly-linked lists.
    fn pq_move(&mut self, i: usize, new_key: u32) {
        let old_key = self.elems[i].key;
        let old_list = key_map(old_key as usize, self.bucket_base, self.bucket_upper);
        let new_list = key_map(new_key as usize, self.bucket_base, self.bucket_upper);

        self.elems[i].key = new_key;
        if old_list == new_list {
            return;
        }

        // Remove from old list
        let prev = self.elems[i].prev;
        let next = self.elems[i].next;
        if prev != SENTINEL {
            // Interior or tail: patch predecessor's next pointer
            self.elems[prev as usize].next = next;
        } else {
            // Head of bucket: advance the bucket head to our successor
            self.lists[old_list] = next;
        }
        if next != SENTINEL {
            // Interior or head: patch successor's prev pointer
            self.elems[next as usize].prev = prev;
        }
        // (If next == SENTINEL, we were the tail; nothing to patch.)

        // Insert at head of new list
        let old_head = self.lists[new_list];
        self.elems[i].prev = SENTINEL; // new head has no predecessor
        self.elems[i].next = old_head;
        if old_head != SENTINEL {
            debug_assert!(i <= u32::MAX as usize);
            self.elems[old_head as usize].prev = i as u32;
        }
        debug_assert!(i <= u32::MAX as usize);
        self.lists[new_list] = i as u32;

        if new_list < self.min_list {
            self.min_list = new_list;
        }
    }

    fn inc(&mut self, i: usize) {
        let key = self.elems[i].key;
        if key >= u32::MAX - 1 {
            return;
        }
        self.pq_move(i, key + 1);
    }

    fn dec(&mut self, i: usize) {
        let old_key = self.elems[i].key;
        if old_key == 0 || old_key == u32::MAX {
            return;
        }
        self.pq_move(i, old_key - 1);
    }

    fn dec_n(&mut self, i: usize, n: u32) {
        let old_key = self.elems[i].key;
        if old_key == 0 || old_key == u32::MAX || n == 0 {
            return;
        }
        self.pq_move(i, old_key.saturating_sub(n));
    }

    #[inline]
    pub(crate) fn next_vertex(&mut self) -> Option<usize> {
        self.pop()
    }

    #[inline]
    pub(crate) fn notify_neighbor_removed(&mut self, v: u32) {
        self.dec(v as usize);
    }

    #[inline]
    pub(crate) fn notify_neighbor_removed_n(&mut self, v: u32, n: u32) {
        self.dec_n(v as usize, n);
    }

    #[inline]
    pub(crate) fn notify_fill_edge(&mut self, u: u32, v: u32) {
        self.inc(u as usize);
        self.inc(v as usize);
    }
}

impl<T: Real> EliminationOrdering<T> for DynamicOrdering {
    fn next_vertex(&mut self) -> Option<usize> {
        DynamicOrdering::next_vertex(self)
    }

    fn notify_neighbor_removed(&mut self, v: u32) {
        DynamicOrdering::notify_neighbor_removed(self, v);
    }

    fn notify_neighbor_removed_n(&mut self, v: u32, n: u32) {
        DynamicOrdering::notify_neighbor_removed_n(self, v, n);
    }

    fn notify_fill_edge(&mut self, u: u32, v: u32) {
        DynamicOrdering::notify_fill_edge(self, u, v);
    }

    fn notify_edges_merged(&mut self, v: u32) {
        DynamicOrdering::notify_neighbor_removed(self, v);
    }

    fn notify_edges_merged_n(&mut self, v: u32, n: u32) {
        DynamicOrdering::notify_neighbor_removed_n(self, v, n);
    }
}

impl DynamicOrdering {
    pub(crate) fn new(n: usize, degrees: impl Iterator<Item = usize>) -> Self {
        Self::new_with_scale(n, degrees, 1)
    }

    pub(crate) fn new_with_scale(
        n: usize,
        degrees: impl Iterator<Item = usize>,
        degree_scale: usize,
    ) -> Self {
        assert!(
            n <= u32::MAX as usize,
            "number of vertices exceeds u32::MAX"
        );
        // Julia AC2 parity: keyMap uses `k = split*n`, bucket array length `2*k+1`.
        // Use scale=1 for standard AC.
        let bucket_base = degree_scale.saturating_mul(n).max(1);
        let n_lists = bucket_base.saturating_mul(2).saturating_add(1);
        let bucket_upper = n_lists - 1;
        let mut lists = vec![SENTINEL; n_lists];
        let mut elems = Vec::with_capacity(n);
        let mut min_list = n_lists;
        let mut n_items = 0;

        for (v, deg) in degrees.enumerate() {
            debug_assert!(deg <= u32::MAX as usize);
            let key = deg as u32;
            let list = key_map(deg, bucket_base, bucket_upper);
            let old_head = lists[list];
            elems.push(PQElem {
                prev: SENTINEL,
                next: old_head,
                key,
            });
            if old_head != SENTINEL {
                debug_assert!(v <= u32::MAX as usize);
                elems[old_head as usize].prev = v as u32;
            }
            debug_assert!(v <= u32::MAX as usize);
            lists[list] = v as u32;
            if list < min_list {
                min_list = list;
            }
            n_items += 1;
        }

        if min_list == n_lists {
            min_list = 0;
        }

        DynamicOrdering {
            elems,
            lists,
            min_list,
            n_items,
            bucket_base,
            bucket_upper,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_map() {
        let k = 10;
        let upper = 2 * k;
        assert_eq!(key_map(0, k, upper), 0);
        assert_eq!(key_map(5, k, upper), 5);
        assert_eq!(key_map(10, k, upper), 10);
        assert_eq!(key_map(15, k, upper), 11); // 10 + 15/10 = 11
        assert_eq!(key_map(20, k, upper), 12); // 10 + 20/10 = 12
        assert_eq!(key_map(10_000, k, upper), upper); // capped at upper bucket
    }

    #[test]
    fn test_pop_order() {
        // 4 vertices with degrees [3, 1, 2, 0]
        let mut pq = DynamicOrdering::new(4, [3, 1, 2, 0].into_iter());

        // Should pop in order of increasing degree
        assert_eq!(pq.next_vertex(), Some(3)); // degree 0
        assert_eq!(pq.next_vertex(), Some(1)); // degree 1
        assert_eq!(pq.next_vertex(), Some(2)); // degree 2
        assert_eq!(pq.next_vertex(), Some(0)); // degree 3
        assert_eq!(pq.next_vertex(), None);
    }

    #[test]
    fn test_inc_dec() {
        // 3 vertices with degrees [2, 1, 3]
        let mut pq = DynamicOrdering::new(3, [2, 1, 3].into_iter());

        // Pop vertex 1 (degree 1, lowest)
        assert_eq!(pq.pop(), Some(1));

        // Increment vertex 0 (degree 2 → 3)
        pq.inc(0);
        assert_eq!(pq.elems[0].key, 3);

        // Decrement vertex 2 (degree 3 → 2)
        pq.dec(2);
        assert_eq!(pq.elems[2].key, 2);

        // Now vertex 2 (degree 2) should come before vertex 0 (degree 3)
        assert_eq!(pq.pop(), Some(2));
        assert_eq!(pq.pop(), Some(0));
        assert_eq!(pq.pop(), None);
    }

    #[test]
    fn test_notify_fill_edge() {
        let mut pq = DynamicOrdering::new(3, [1, 1, 1].into_iter());

        // Fill edge between 0 and 2 → both inc by 1
        pq.notify_fill_edge(0u32, 2u32);
        assert_eq!(pq.elems[0].key, 2);
        assert_eq!(pq.elems[1].key, 1);
        assert_eq!(pq.elems[2].key, 2);

        // Vertex 1 (degree 1) should pop first
        assert_eq!(pq.pop(), Some(1));
    }

    #[test]
    fn test_notify_edges_merged() {
        let mut pq = DynamicOrdering::new(3, [3, 2, 1].into_iter());

        // Merging edges to vertex 0 → degree decreases
        <DynamicOrdering as EliminationOrdering<f64>>::notify_edges_merged(&mut pq, 0u32);
        assert_eq!(pq.elems[0].key, 2);

        <DynamicOrdering as EliminationOrdering<f64>>::notify_edges_merged(&mut pq, 0u32);
        assert_eq!(pq.elems[0].key, 1);
    }

    #[test]
    fn test_notify_edges_merged_n() {
        let mut pq = DynamicOrdering::new(3, [5, 2, 1].into_iter());
        <DynamicOrdering as EliminationOrdering<f64>>::notify_edges_merged_n(&mut pq, 0u32, 3);
        assert_eq!(pq.elems[0].key, 2);
    }

    #[test]
    fn test_split_scaled_bucket_layout() {
        let pq = DynamicOrdering::new_with_scale(4, [1, 2, 3, 4].into_iter(), 2);
        assert_eq!(pq.bucket_base, 8);
        assert_eq!(pq.lists.len(), 17);
    }

    #[test]
    fn test_empty_pq() {
        let mut pq = DynamicOrdering::new(0, std::iter::empty());
        assert_eq!(pq.next_vertex(), None);
    }

    #[test]
    fn test_dec_at_zero() {
        let mut pq = DynamicOrdering::new(1, [0].into_iter());
        pq.dec(0); // should not underflow
        assert_eq!(pq.elems[0].key, 0);
        assert_eq!(pq.pop(), Some(0));
    }
}
