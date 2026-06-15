//! Elimination ordering abstraction for approximate Cholesky factorization.
//!
//! Separates how the next vertex to eliminate is selected from the
//! elimination loop.
//! - `DynamicOrdering`: bucket-based priority queue that adapts to fill-in
//!   during elimination (ports Julia's `ApproxCholPQ`)

use crate::CsrError;

// ---------------------------------------------------------------------------
// DynamicOrdering — bucket-based priority queue (Julia `ApproxCholPQ` port)
// ---------------------------------------------------------------------------

/// Linked-list terminator: marks that there is no previous or next element in a bucket chain.
const SENTINEL: u32 = u32::MAX;

/// One element in the bucket priority queue, representing a single vertex.
///
/// `key == u32::MAX` means the element has been removed from the priority queue
/// (popped or logically deleted); `apply_delta` skips removed elements.
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
/// Requires `n <= u32::MAX` (validated at construction).
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

    /// Apply a signed net degree change to vertex `i` in one bucket move.
    ///
    /// `i64` so the full `u32` count range can be negated and summed without the
    /// sign-flip an `i32` cast would cause; the clamp floors at zero.
    fn apply_delta(&mut self, i: usize, delta: i64) {
        let key = self.elems[i].key;
        if key == u32::MAX {
            return;
        }
        let new_key = (key as i64 + delta).clamp(0, (u32::MAX - 1) as i64) as u32;
        if new_key != key {
            self.pq_move(i, new_key);
        }
    }

    /// Decrease vertex `i` by `n`, flooring at zero. Used for the immediate
    /// merge-compression decrement (see `apply_merged_counts`).
    #[inline]
    pub(crate) fn decrease(&mut self, i: usize, n: u32) {
        self.apply_delta(i, -(n as i64));
    }

    #[inline]
    pub(crate) fn next_vertex(&mut self) -> Option<usize> {
        self.pop()
    }
}

/// Accumulates net per-vertex degree changes for one elimination step, then
/// applies them as one bucket move per affected vertex on [`flush`](Self::flush).
///
/// Tracks which vertices it touched so `flush` resets exactly those — the buffer
/// is all-zero between steps no matter which vertices a step hits, so the caller
/// need not enumerate them.
pub(crate) struct DegreeDeltas {
    buf: Vec<i64>,
    touched: Vec<u32>,
}

impl DegreeDeltas {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            buf: vec![0; n],
            touched: Vec::new(),
        }
    }

    #[inline]
    pub(crate) fn increase(&mut self, v: u32, n: u32) {
        self.add(v, n as i64);
    }

    #[inline]
    pub(crate) fn decrease(&mut self, v: u32, n: u32) {
        self.add(v, -(n as i64));
    }

    #[inline]
    fn add(&mut self, v: u32, delta: i64) {
        let i = v as usize;
        if self.buf[i] == 0 {
            self.touched.push(v);
        }
        self.buf[i] += delta;
    }

    pub(crate) fn flush(&mut self, ordering: &mut DynamicOrdering) {
        for &v in &self.touched {
            let i = v as usize;
            let d = self.buf[i];
            self.buf[i] = 0;
            if d != 0 {
                ordering.apply_delta(i, d);
            }
        }
        self.touched.clear();
    }
}

impl DynamicOrdering {
    pub(crate) fn new(n: usize, degrees: impl Iterator<Item = usize>) -> Result<Self, CsrError> {
        Self::new_with_scale(n, degrees, 1)
    }

    pub(crate) fn new_with_scale(
        n: usize,
        degrees: impl Iterator<Item = usize>,
        degree_scale: usize,
    ) -> Result<Self, CsrError> {
        if n > u32::MAX as usize {
            return Err(CsrError::MatrixDimensionExceedsIndexType { n });
        }
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

        Ok(DynamicOrdering {
            elems,
            lists,
            min_list,
            n_items,
            bucket_base,
            bucket_upper,
        })
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
        let mut pq = DynamicOrdering::new(4, [3, 1, 2, 0].into_iter()).expect("valid n");

        // Should pop in order of increasing degree
        assert_eq!(pq.next_vertex(), Some(3)); // degree 0
        assert_eq!(pq.next_vertex(), Some(1)); // degree 1
        assert_eq!(pq.next_vertex(), Some(2)); // degree 2
        assert_eq!(pq.next_vertex(), Some(0)); // degree 3
        assert_eq!(pq.next_vertex(), None);
    }

    #[test]
    fn test_apply_delta_inc_dec() {
        // 3 vertices with degrees [2, 1, 3]
        let mut pq = DynamicOrdering::new(3, [2, 1, 3].into_iter()).expect("valid n");

        // Pop vertex 1 (degree 1, lowest)
        assert_eq!(pq.pop(), Some(1));

        // Increment vertex 0 (degree 2 → 3)
        pq.apply_delta(0, 1);
        assert_eq!(pq.elems[0].key, 3);

        // Decrement vertex 2 (degree 3 → 2)
        pq.apply_delta(2, -1);
        assert_eq!(pq.elems[2].key, 2);

        // Now vertex 2 (degree 2) should come before vertex 0 (degree 3)
        assert_eq!(pq.pop(), Some(2));
        assert_eq!(pq.pop(), Some(0));
        assert_eq!(pq.pop(), None);
    }

    #[test]
    fn test_apply_delta_fill_edge() {
        let mut pq = DynamicOrdering::new(3, [1, 1, 1].into_iter()).expect("valid n");

        // Fill edge between 0 and 2 → each endpoint's degree estimate +1.
        pq.apply_delta(0, 1);
        pq.apply_delta(2, 1);
        assert_eq!(pq.elems[0].key, 2);
        assert_eq!(pq.elems[1].key, 1);
        assert_eq!(pq.elems[2].key, 2);

        // Vertex 1 (degree 1) should pop first
        assert_eq!(pq.pop(), Some(1));
    }

    #[test]
    fn test_apply_delta_net() {
        // A signed net delta is applied in one bucket move; underflow clamps at 0.
        let mut pq = DynamicOrdering::new(3, [5, 2, 1].into_iter()).expect("valid n");
        pq.apply_delta(0, -2); // 5 → 3
        assert_eq!(pq.elems[0].key, 3);
        pq.apply_delta(0, -5); // 3 - 5 clamps to 0
        assert_eq!(pq.elems[0].key, 0);
    }

    #[test]
    fn test_merged_edges_decrease_degree() {
        let mut pq = DynamicOrdering::new(3, [3, 2, 1].into_iter()).expect("valid n");

        // Compression merges a duplicate edge to vertex 0 → degree estimate -1.
        pq.apply_delta(0, -1);
        assert_eq!(pq.elems[0].key, 2);

        pq.apply_delta(0, -1);
        assert_eq!(pq.elems[0].key, 1);
    }

    #[test]
    fn test_merged_edges_decrease_degree_by_n() {
        let mut pq = DynamicOrdering::new(3, [5, 2, 1].into_iter()).expect("valid n");
        pq.apply_delta(0, -3);
        assert_eq!(pq.elems[0].key, 2);
    }

    #[test]
    fn test_decrease_large_count_keeps_sign() {
        // `decrease` takes a `u32` and negates it as `i64` internally, so a count
        // above i32::MAX stays a *decrease*: with an i32 delta, `-(count as i32)`
        // would sign-flip to a large positive and *raise* the degree.
        let mut pq = DynamicOrdering::new(2, [10, 1].into_iter()).expect("valid n");
        let count: u32 = 3_000_000_000; // > i32::MAX
        pq.decrease(0, count); // 10 - 3e9 clamps to 0, never raises
        assert_eq!(pq.elems[0].key, 0);
    }

    #[test]
    fn test_split_scaled_bucket_layout() {
        let pq = DynamicOrdering::new_with_scale(4, [1, 2, 3, 4].into_iter(), 2).expect("valid n");
        assert_eq!(pq.bucket_base, 8);
        assert_eq!(pq.lists.len(), 17);
    }

    #[test]
    fn test_empty_pq() {
        let mut pq = DynamicOrdering::new(0, std::iter::empty()).expect("valid n");
        assert_eq!(pq.next_vertex(), None);
    }

    #[test]
    fn test_apply_delta_at_zero_clamps() {
        let mut pq = DynamicOrdering::new(1, [0].into_iter()).expect("valid n");
        pq.apply_delta(0, -1); // should not underflow
        assert_eq!(pq.elems[0].key, 0);
        assert_eq!(pq.pop(), Some(0));
    }

    #[test]
    fn test_degree_deltas_flush_applies_net_per_vertex() {
        let mut pq = DynamicOrdering::new(3, [5, 5, 5].into_iter()).expect("valid n");
        let mut deltas = DegreeDeltas::new(3);

        // Vertex 0: +1 +1 -3 = net -1. Vertex 1: +2. Vertex 2: untouched.
        deltas.increase(0, 1);
        deltas.increase(0, 1);
        deltas.decrease(0, 3);
        deltas.increase(1, 2);
        deltas.flush(&mut pq);

        assert_eq!(pq.elems[0].key, 4); // 5 - 1
        assert_eq!(pq.elems[1].key, 7); // 5 + 2
        assert_eq!(pq.elems[2].key, 5); // untouched

        // flush resets the buffer for every touched vertex, so a second flush
        // with no accumulated deltas is a no-op (no stale carryover).
        deltas.flush(&mut pq);
        assert_eq!(pq.elems[0].key, 4);
        assert_eq!(pq.elems[1].key, 7);
        assert_eq!(pq.elems[2].key, 5);
    }
}
