//! Bucket priority queue over live degree estimates.

/// Linked-list terminator.
const SENTINEL: u32 = u32::MAX;

/// `key == u32::MAX` marks an element removed; `apply_delta` skips those.
struct PQElem {
    prev: u32,
    next: u32,
    key: u32,
}

/// Buckets are linked lists through [`PQElem`]; `min_list` is only a lower bound on the minimum.
pub(super) struct DynamicOrdering {
    elems: Vec<PQElem>,
    lists: Vec<u32>,
    min_list: usize,
    bucket_base: usize,
}

/// Sole derivation of the bucket count, so `key_map`'s cap and `new` agree on the last bucket.
fn n_buckets(bucket_base: usize) -> usize {
    bucket_base.saturating_mul(2).saturating_add(1)
}

fn key_map(degree: usize, bucket_base: usize) -> usize {
    if degree <= bucket_base {
        degree
    } else {
        (bucket_base + degree / bucket_base).min(n_buckets(bucket_base) - 1)
    }
}

impl DynamicOrdering {
    pub(super) fn next_vertex(&mut self) -> Option<usize> {
        while self.min_list < self.lists.len() && self.lists[self.min_list] == SENTINEL {
            let previous = self.min_list;
            self.min_list += 1;
            // A broken advance would hang re-checking one bucket; this turns that into a panic.
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
        self.elems[i].key = u32::MAX;
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

    /// `i64` so negating a full-range `u32` count cannot sign-flip as an `i32` would.
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

/// One bucket move per affected vertex on [`flush`](Self::flush), which resets what it touched.
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
        // Matches Laplacians.jl AC2: bucket base `k = split * n`, `2k + 1` buckets.
        let bucket_base = degree_scale.saturating_mul(n).max(1);
        let n_lists = n_buckets(bucket_base);
        let mut lists = vec![SENTINEL; n_lists];
        let mut elems = Vec::with_capacity(n);
        let mut min_list = n_lists;

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
        }

        DynamicOrdering {
            elems,
            lists,
            min_list,
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
        assert_eq!(key_map(15, k), 11);
        assert_eq!(key_map(20, k), 12);
        assert_eq!(key_map(10_000, k), n_buckets(k) - 1);
    }

    #[test]
    fn test_pop_order() {
        let mut pq = DynamicOrdering::new(&[3, 1, 2, 0], 1);

        assert_eq!(pq.next_vertex(), Some(3));
        assert_eq!(pq.next_vertex(), Some(1));
        assert_eq!(pq.next_vertex(), Some(2));
        assert_eq!(pq.next_vertex(), Some(0));
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
        // An `i32` delta would sign-flip a count above `i32::MAX` into a degree increase.
        let mut pq = DynamicOrdering::new(&[10, 1], 1);
        let count: u32 = 3_000_000_000;
        pq.decrease(0, count);
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

        deltas.increase(0, 1);
        deltas.increase(0, 1);
        deltas.decrease(0, 3);
        deltas.increase(1, 2);
        deltas.flush(&mut pq);

        assert_eq!(pq.elems[0].key, 4);
        assert_eq!(pq.elems[1].key, 7);
        assert_eq!(pq.elems[2].key, 5);

        // A second flush with nothing accumulated must not replay stale deltas.
        deltas.flush(&mut pq);
        assert_eq!(pq.elems[0].key, 4);
        assert_eq!(pq.elems[1].key, 7);
        assert_eq!(pq.elems[2].key, 5);
    }
}
