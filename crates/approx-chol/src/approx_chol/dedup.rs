use core::cmp::Ordering;

use crate::graph::Neighbor;
use crate::Real;
use num_traits::NumCast;

#[derive(Clone, Copy)]
struct Ac2SortEntry<T: Real> {
    idx: u32,
    weight: T,
    count: u32,
}

/// Neighborhoods with at most this many entries use sort-based dedup (O(d log d),
/// cache-friendly for small d). Larger neighborhoods use scatter-gather (O(d) via
/// indexed buffers, but with higher constant from random-access pattern).
const SCATTER_THRESHOLD: usize = 32;

/// Sort entries by weight (ascending), breaking ties by vertex index.
fn sort_by_weight_then_index<T: Real>(entries: &mut [(u32, T)]) {
    entries.sort_unstable_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
}

/// Shared scratch for dedup variants.
struct DedupScratch<T: Real> {
    /// `scatter[idx]` accumulates weight for vertex `idx`.
    scatter: Vec<T>,
    /// Tracks first-seen vertices for AC scatter dedup.
    scatter_seen: Vec<bool>,
    /// Tracks unique vertex indices seen in the current pass.
    unique: Vec<u32>,
    /// Number of vertices in the graph (for buffer sizing).
    n: usize,
}

impl<T: Real> DedupScratch<T> {
    fn new(n: usize) -> Self {
        Self {
            scatter: Vec::new(),
            scatter_seen: Vec::new(),
            unique: Vec::new(),
            n,
        }
    }

    fn ensure_scatter_buffers(&mut self) {
        if self.scatter.len() < self.n {
            self.scatter.resize(self.n, T::zero());
            self.scatter_seen.resize(self.n, false);
        }
    }
}

/// AC dedup workspace (weights only, tracks merged vertex ids).
pub(super) struct AcDedupWorkspace<T: Real> {
    scratch: DedupScratch<T>,
    /// Vertex indices whose entries were merged.
    merged: Vec<u32>,
}

impl<T: Real> AcDedupWorkspace<T> {
    pub fn new(n: usize) -> Self {
        Self {
            scratch: DedupScratch::new(n),
            merged: Vec::new(),
        }
    }

    /// Vertex indices whose entries were merged during the last dedup call.
    pub fn merged(&self) -> &[u32] {
        &self.merged
    }

    /// Deduplicate raw tuples for AC path and sort by weight ascending.
    pub fn dedup(&mut self, raw: &mut [Neighbor<T>], entries: &mut Vec<(u32, T)>) {
        if raw.len() <= SCATTER_THRESHOLD {
            self.dedup_sort_small(raw, entries);
        } else {
            self.dedup_scatter(raw, entries);
        }
    }

    fn dedup_sort_small(&mut self, raw: &mut [Neighbor<T>], entries: &mut Vec<(u32, T)>) {
        self.dedup_sort_core(raw, entries);
        sort_by_weight_then_index(entries);
    }

    fn dedup_sort_core(&mut self, raw: &mut [Neighbor<T>], entries: &mut Vec<(u32, T)>) {
        self.merged.clear();
        entries.clear();
        if raw.is_empty() {
            return;
        }
        if raw.len() == 1 {
            entries.push((raw[0].to, raw[0].fill_weight));
            return;
        }

        raw.sort_unstable_by_key(|n| n.to);

        let mut write = 0;
        for read in 1..raw.len() {
            if raw[write].to == raw[read].to {
                raw[write].fill_weight = raw[write].fill_weight + raw[read].fill_weight;
                self.merged.push(raw[write].to);
            } else {
                entries.push((raw[write].to, raw[write].fill_weight));
                write += 1;
                raw[write] = raw[read];
            }
        }
        entries.push((raw[write].to, raw[write].fill_weight));
    }

    fn dedup_scatter(&mut self, raw: &[Neighbor<T>], entries: &mut Vec<(u32, T)>) {
        self.scratch.ensure_scatter_buffers();
        self.scratch.unique.clear();
        self.merged.clear();
        entries.clear();

        for nbr in raw {
            let idx = nbr.to as usize;
            if !self.scratch.scatter_seen[idx] {
                self.scratch.scatter_seen[idx] = true;
                self.scratch.unique.push(nbr.to);
            } else {
                self.merged.push(nbr.to);
            }
            self.scratch.scatter[idx] = self.scratch.scatter[idx] + nbr.fill_weight;
        }

        for &idx in &self.scratch.unique {
            let idx_usize = idx as usize;
            entries.push((idx, self.scratch.scatter[idx_usize]));
            self.scratch.scatter[idx_usize] = T::zero();
            self.scratch.scatter_seen[idx_usize] = false;
        }
        sort_by_weight_then_index(entries);
    }
}

/// AC2 dedup workspace (weights + multiplicities + merge-cap reporting).
pub(super) struct Ac2DedupWorkspace<T: Real> {
    scratch: DedupScratch<T>,
    /// Scatter buffer for multi-edge counts during scatter-gather dedup.
    scatter_counts: Vec<u32>,
    /// Compressed merge counts for AC2 merge-limit discards.
    merged_counts: Vec<(u32, u32)>,
    /// Reusable packed scratch for AC2 avg-weight sorting.
    sort_entries: Vec<Ac2SortEntry<T>>,
}

impl<T: Real> Ac2DedupWorkspace<T> {
    pub fn new(n: usize) -> Self {
        Self {
            scratch: DedupScratch::new(n),
            scatter_counts: Vec::new(),
            merged_counts: Vec::new(),
            sort_entries: Vec::new(),
        }
    }

    /// `(vertex, count)` pairs for merge-limit discards during the last dedup.
    pub fn merged_counts(&self) -> &[(u32, u32)] {
        &self.merged_counts
    }

    /// Deduplicate raw tuples for AC2 path, apply merge cap, and sort by avg-weight.
    pub fn dedup(
        &mut self,
        raw: &mut [Neighbor<T>],
        entries: &mut Vec<(u32, T)>,
        counts: &mut Vec<u32>,
        merge_limit: u32,
    ) {
        if raw.len() <= SCATTER_THRESHOLD {
            self.dedup_sort_small(raw, entries, counts, merge_limit);
        } else {
            self.dedup_scatter(raw, entries, counts, merge_limit);
        }
    }

    fn dedup_sort_small(
        &mut self,
        raw: &mut [Neighbor<T>],
        entries: &mut Vec<(u32, T)>,
        counts: &mut Vec<u32>,
        merge_limit: u32,
    ) {
        self.dedup_sort_core(raw, entries, counts);
        Self::apply_merge_limit(entries, counts, merge_limit, &mut self.merged_counts);
        self.sort_by_avg_weight(entries, counts);
    }

    fn dedup_sort_core(
        &mut self,
        raw: &mut [Neighbor<T>],
        entries: &mut Vec<(u32, T)>,
        counts: &mut Vec<u32>,
    ) {
        self.merged_counts.clear();
        entries.clear();
        counts.clear();
        if raw.is_empty() {
            return;
        }
        if raw.len() == 1 {
            entries.push((raw[0].to, raw[0].fill_weight));
            counts.push(raw[0].count);
            return;
        }

        raw.sort_unstable_by_key(|n| n.to);

        let mut write = 0;
        let mut count: u32 = raw[0].count;
        for read in 1..raw.len() {
            if raw[write].to == raw[read].to {
                raw[write].fill_weight = raw[write].fill_weight + raw[read].fill_weight;
                count = count.saturating_add(raw[read].count);
            } else {
                entries.push((raw[write].to, raw[write].fill_weight));
                counts.push(count);
                count = raw[read].count;
                write += 1;
                raw[write] = raw[read];
            }
        }
        entries.push((raw[write].to, raw[write].fill_weight));
        counts.push(count);
    }

    fn dedup_scatter(
        &mut self,
        raw: &[Neighbor<T>],
        entries: &mut Vec<(u32, T)>,
        counts: &mut Vec<u32>,
        merge_limit: u32,
    ) {
        self.scratch.ensure_scatter_buffers();
        self.scratch.unique.clear();
        self.merged_counts.clear();
        entries.clear();
        counts.clear();
        if self.scatter_counts.len() < self.scratch.n {
            self.scatter_counts.resize(self.scratch.n, 0);
        }

        for nbr in raw {
            let idx = nbr.to as usize;
            if self.scatter_counts[idx] == 0 {
                self.scratch.unique.push(nbr.to);
            }
            self.scratch.scatter[idx] = self.scratch.scatter[idx] + nbr.fill_weight;
            self.scatter_counts[idx] = self.scatter_counts[idx].saturating_add(nbr.count);
        }

        for &idx in &self.scratch.unique {
            let idx_usize = idx as usize;
            entries.push((idx, self.scratch.scatter[idx_usize]));
            counts.push(self.scatter_counts[idx_usize]);
            self.scratch.scatter[idx_usize] = T::zero();
            self.scatter_counts[idx_usize] = 0;
        }

        Self::apply_merge_limit(entries, counts, merge_limit, &mut self.merged_counts);
        self.sort_by_avg_weight(entries, counts);
    }

    /// Apply merge limit: cap multi-edge counts, preserving total weight.
    fn apply_merge_limit(
        entries: &[(u32, T)],
        counts: &mut [u32],
        merge_limit: u32,
        merged_counts: &mut Vec<(u32, u32)>,
    ) {
        let limit = merge_limit;
        for i in 0..entries.len() {
            let count = counts[i];
            if count > limit {
                let discarded = count - limit;
                counts[i] = limit;
                let idx = entries[i].0;
                merged_counts.push((idx, discarded));
            }
        }
    }

    /// Sort entries by average weight ascending, then by vertex index.
    fn sort_by_avg_weight(&mut self, entries: &mut [(u32, T)], counts: &mut [u32]) {
        let len = entries.len();
        self.sort_entries.clear();
        self.sort_entries.reserve(len);
        for i in 0..len {
            self.sort_entries.push(Ac2SortEntry {
                idx: entries[i].0,
                weight: entries[i].1,
                count: counts[i],
            });
        }

        debug_assert!(
            self.sort_entries.iter().all(|e| e.count > 0),
            "sort_by_avg_weight: all counts must be positive for meaningful avg-weight comparison"
        );

        self.sort_entries.sort_unstable_by(|a, b| {
            // Cross-multiply to compare a.weight/a.count vs b.weight/b.count
            // without division.
            let lhs = a.weight * <T as NumCast>::from(b.count).expect("count to scalar");
            let rhs = b.weight * <T as NumCast>::from(a.count).expect("count to scalar");
            lhs.partial_cmp(&rhs)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.idx.cmp(&b.idx))
        });

        for (dst, item) in self.sort_entries.iter().enumerate() {
            entries[dst] = (item.idx, item.weight);
            counts[dst] = item.count;
        }
    }
}

#[cfg(test)]
mod tests;
