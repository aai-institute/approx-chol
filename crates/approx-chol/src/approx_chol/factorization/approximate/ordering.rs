//! Bucket priority queue over live degree estimates (ports Julia's `ApproxCholPQ`).

/// Linked-list terminator.
const SENTINEL: u32 = u32::MAX;

/// `key == u32::MAX` marks an element removed; `apply_delta` skips those.
struct PQElem {
    prev: u32, // SENTINEL = head of bucket list
    next: u32, // SENTINEL = tail of bucket list
    key: u32,  // current degree estimate
}

/// Each bucket is a doubly-linked list threaded through [`PQElem`]; `min_list` is a
/// *lower bound* on the minimum non-empty bucket, which
/// [`next_vertex`](Self::next_vertex) scans upward from.
pub(super) struct DynamicOrdering {
    elems: Vec<PQElem>, // indexed by vertex id
    lists: Vec<u32>,    // bucket heads, indexed by key_map(degree)
    min_list: usize,    // lower bound on minimum non-empty bucket
    n_items: usize,
    bucket_base: usize,
}

/// The one place the bucket count follows from the base, so the cap below and
/// [`DynamicOrdering::new`] cannot disagree about which bucket is last.
fn n_buckets(bucket_base: usize) -> usize {
    bucket_base.saturating_mul(2).saturating_add(1)
}

/// Degrees at or below `bucket_base` get their own bucket; higher ones group by
/// `bucket_base + degree / bucket_base`.
fn key_map(degree: usize, bucket_base: usize) -> usize {
    if degree <= bucket_base {
        degree
    } else {
        (bucket_base + degree / bucket_base).min(n_buckets(bucket_base) - 1)
    }
}

impl DynamicOrdering {
    pub(super) fn next_vertex(&mut self) -> Option<usize> {
        if self.n_items == 0 {
            return None;
        }
        while self.min_list < self.lists.len() && self.lists[self.min_list] == SENTINEL {
            let previous = self.min_list;
            self.min_list += 1;
            // A broken advance leaves this condition re-checking the same bucket
            // forever instead of failing, so a bad increment hangs rather than
            // panics.
            debug_assert!(
                self.min_list > previous,
                "next_vertex's bucket scan failed to advance"
            );
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

    fn pq_move(&mut self, i: usize, new_key: u32) {
        let old_key = self.elems[i].key;
        let old_list = key_map(old_key as usize, self.bucket_base);
        let new_list = key_map(new_key as usize, self.bucket_base);

        self.elems[i].key = new_key;
        if old_list == new_list {
            return;
        }

        let prev = self.elems[i].prev;
        let next = self.elems[i].next;
        if prev != SENTINEL {
            self.elems[prev as usize].next = next;
        } else {
            self.lists[old_list] = next;
        }
        if next != SENTINEL {
            self.elems[next as usize].prev = prev;
        }

        let old_head = self.lists[new_list];
        self.elems[i].prev = SENTINEL;
        self.elems[i].next = old_head;
        if old_head != SENTINEL {
            self.elems[old_head as usize].prev = i as u32;
        }
        self.lists[new_list] = i as u32;

        if new_list < self.min_list {
            self.min_list = new_list;
        }
    }

    /// `i64` so the full `u32` count range negates and sums without the sign flip an
    /// `i32` cast would cause.
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

    /// The immediate merge-compression decrement; see `apply_removed_copies`.
    #[inline]
    pub(super) fn decrease(&mut self, i: usize, n: u32) {
        self.apply_delta(i, -(n as i64));
    }
}

/// One bucket move per affected vertex on [`flush`](Self::flush), which resets exactly
/// the vertices it touched so the caller need not enumerate them.
pub(super) struct DegreeDeltas {
    buf: Vec<i64>,
    touched: Vec<u32>,
}

impl DegreeDeltas {
    pub(super) fn new(n: usize) -> Self {
        Self {
            buf: vec![0; n],
            touched: Vec::new(),
        }
    }

    #[inline]
    pub(super) fn increase(&mut self, v: u32, n: u32) {
        self.add(v, n as i64);
    }

    #[inline]
    pub(super) fn decrease(&mut self, v: u32, n: u32) {
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

    pub(super) fn flush(&mut self, ordering: &mut DynamicOrdering) {
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
    pub(super) fn new(degrees: &[usize], degree_scale: usize) -> Self {
        let n = degrees.len();
        // Julia AC2 parity: keyMap uses `k = split*n`, bucket array length `2*k+1`.
        // Use scale=1 for standard AC.
        let bucket_base = degree_scale.saturating_mul(n).max(1);
        let n_lists = n_buckets(bucket_base);
        let mut lists = vec![SENTINEL; n_lists];
        let mut elems = Vec::with_capacity(n);
        let mut min_list = n_lists;
        let mut n_items = 0;

        for (v, &deg) in degrees.iter().enumerate() {
            let key = deg as u32;
            let list = key_map(deg, bucket_base);
            let old_head = lists[list];
            elems.push(PQElem {
                prev: SENTINEL,
                next: old_head,
                key,
            });
            if old_head != SENTINEL {
                elems[old_head as usize].prev = v as u32;
            }
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_map() {
        let k = 10;
        assert_eq!(key_map(0, k), 0);
        assert_eq!(key_map(5, k), 5);
        assert_eq!(key_map(10, k), 10);
        assert_eq!(key_map(15, k), 11); // 10 + 15/10 = 11
        assert_eq!(key_map(20, k), 12); // 10 + 20/10 = 12
        assert_eq!(key_map(10_000, k), n_buckets(k) - 1); // capped at last bucket
    }

    #[test]
    fn test_pop_order() {
        // 4 vertices with degrees [3, 1, 2, 0]
        let mut pq = DynamicOrdering::new(&[3, 1, 2, 0], 1);

        // Should pop in order of increasing degree
        assert_eq!(pq.next_vertex(), Some(3)); // degree 0
        assert_eq!(pq.next_vertex(), Some(1)); // degree 1
        assert_eq!(pq.next_vertex(), Some(2)); // degree 2
        assert_eq!(pq.next_vertex(), Some(0)); // degree 3
        assert_eq!(pq.next_vertex(), None);
    }

    type DeltaCase<'a> = (&'a str, &'a [usize], &'a [(usize, i64)], &'a [u32]);

    /// Fill, removal and merge compression are all one operation on the key.
    #[test]
    fn test_apply_delta_moves_the_key_by_the_delta() {
        let cases: [DeltaCase<'_>; 4] = [
            (
                "fill on two endpoints",
                &[1, 1, 1],
                &[(0, 1), (2, 1)],
                &[2, 1, 2],
            ),
            (
                "increment and decrement",
                &[2, 1, 3],
                &[(0, 1), (2, -1)],
                &[3, 1, 2],
            ),
            ("net decrease", &[5, 2, 1], &[(0, -2), (0, -1)], &[2, 2, 1]),
            ("underflow floors at zero", &[1, 2], &[(0, -5)], &[0, 2]),
        ];
        for (label, degrees, deltas, expected) in cases {
            let mut pq = DynamicOrdering::new(degrees, 1);
            for &(vertex, delta) in deltas {
                pq.apply_delta(vertex, delta);
            }
            for (vertex, &key) in expected.iter().enumerate() {
                assert_eq!(pq.elems[vertex].key, key, "{label}: vertex {vertex}");
            }
        }
    }

    /// The key change has to re-bucket, not just re-label.
    #[test]
    fn test_apply_delta_rebuckets() {
        let mut pq = DynamicOrdering::new(&[2, 1, 3], 1);
        assert_eq!(pq.next_vertex(), Some(1));
        pq.apply_delta(0, 1);
        pq.apply_delta(2, -1);
        assert_eq!(pq.next_vertex(), Some(2));
        assert_eq!(pq.next_vertex(), Some(0));
        assert_eq!(pq.next_vertex(), None);
    }

    #[test]
    fn test_decrease_large_count_keeps_sign() {
        // `decrease` takes a `u32` and negates it as `i64` internally, so a count
        // above i32::MAX stays a *decrease*: with an i32 delta, `-(count as i32)`
        // would sign-flip to a large positive and *raise* the degree.
        let mut pq = DynamicOrdering::new(&[10, 1], 1);
        let count: u32 = 3_000_000_000; // > i32::MAX
        pq.decrease(0, count); // 10 - 3e9 clamps to 0, never raises
        assert_eq!(pq.elems[0].key, 0);
    }

    #[test]
    fn test_split_scaled_bucket_layout() {
        let pq = DynamicOrdering::new(&[1, 2, 3, 4], 2);
        assert_eq!(pq.bucket_base, 8);
        assert_eq!(pq.lists.len(), 17);
    }

    #[test]
    fn test_empty_pq() {
        let mut pq = DynamicOrdering::new(&[], 1);
        assert_eq!(pq.next_vertex(), None);
    }

    #[test]
    fn test_degree_deltas_flush_applies_net_per_vertex() {
        let mut pq = DynamicOrdering::new(&[5, 5, 5], 1);
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
