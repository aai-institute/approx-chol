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

/// Deduplication workspace for star neighborhood compression.
pub(super) struct DedupWorkspace<T: Real> {
    /// Scatter buffer: `scatter[idx]` accumulates weight for vertex `idx`.
    scatter: Vec<T>,
    /// Boolean companion to `scatter`: tracks first-seen vertices.
    scatter_seen: Vec<bool>,
    /// Scatter buffer for multi-edge counts during scatter-gather dedup.
    scatter_counts: Vec<u32>,
    /// Tracks unique vertex indices seen during the scatter phase.
    unique: Vec<u32>,
    /// Vertex indices whose entries were merged.
    merged: Vec<u32>,
    /// Compressed merge counts for AC2 merge-limit discards.
    merged_counts: Vec<(u32, u32)>,
    /// Number of vertices in the graph (needed to size the scatter buffer).
    n: usize,
    /// Reusable packed scratch for AC2 avg-weight sorting.
    sort_entries: Vec<Ac2SortEntry<T>>,
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

impl<T: Real> DedupWorkspace<T> {
    pub fn new(n: usize) -> Self {
        Self {
            scatter: Vec::new(),
            scatter_seen: Vec::new(),
            scatter_counts: Vec::new(), // keep lazy — only used in AC2 path
            unique: Vec::new(),
            merged: Vec::new(),
            merged_counts: Vec::new(),
            n,
            sort_entries: Vec::new(),
        }
    }

    /// Vertex indices whose entries were merged during the last dedup call.
    pub fn merged(&self) -> &[u32] {
        &self.merged
    }

    /// `(vertex, count)` pairs for merge-limit discards during the last AC2 dedup.
    pub fn merged_counts(&self) -> &[(u32, u32)] {
        &self.merged_counts
    }

    /// Deduplicate raw 3-tuples for the AC (approximate Cholesky) path.
    ///
    /// Takes `raw` entries `(vertex_index, weight, count)` that may contain duplicate
    /// vertex indices (from multiple fill edges landing on the same neighbor). Merges
    /// duplicates by summing their weights, and writes the deduplicated result into
    /// `entries` as `(vertex_index, total_weight)` pairs sorted by weight ascending
    /// (ties broken by vertex index).
    ///
    /// After the call, [`merged()`](Self::merged) returns the vertex indices that had
    /// duplicate entries (i.e., whose weights were summed). The count field in `raw` is
    /// ignored by the AC path.
    ///
    /// Uses a sort-based path for small inputs (`<= SCATTER_THRESHOLD`) and a
    /// scatter-gather path for larger neighborhoods.
    pub fn dedup_ac(&mut self, raw: &mut [Neighbor<T>], entries: &mut Vec<(u32, T)>) {
        if raw.len() <= SCATTER_THRESHOLD {
            self.dedup_sort_small_ac(raw, entries);
        } else {
            self.dedup_scatter_ac(raw, entries);
        }
    }

    /// Sort-based dedup core: merge duplicate vertex indices in a sorted raw slice.
    ///
    /// When `track_counts` is true (AC2 path), accumulates edge counts with `saturating_add`
    /// and populates `counts`. When false (AC path), records merged vertex indices in `self.merged`.
    fn dedup_sort_core(
        &mut self,
        raw: &mut [Neighbor<T>],
        entries: &mut Vec<(u32, T)>,
        counts: &mut Vec<u32>,
        track_counts: bool,
    ) {
        self.merged.clear();
        entries.clear();
        if track_counts {
            self.merged_counts.clear();
            counts.clear();
        }
        if raw.is_empty() {
            return;
        }
        if raw.len() == 1 {
            entries.push((raw[0].to, raw[0].fill_weight));
            if track_counts {
                counts.push(raw[0].count);
            }
            return;
        }

        raw.sort_unstable_by_key(|n| n.to);

        let mut write = 0;
        let mut count: u32 = raw[0].count;
        for read in 1..raw.len() {
            if raw[write].to == raw[read].to {
                raw[write].fill_weight = raw[write].fill_weight + raw[read].fill_weight;
                if track_counts {
                    count = count.saturating_add(raw[read].count);
                } else {
                    self.merged.push(raw[write].to);
                }
            } else {
                entries.push((raw[write].to, raw[write].fill_weight));
                if track_counts {
                    counts.push(count);
                    count = raw[read].count;
                }
                write += 1;
                raw[write] = raw[read];
            }
        }
        entries.push((raw[write].to, raw[write].fill_weight));
        if track_counts {
            counts.push(count);
        }
    }

    /// Sort-based dedup for small neighborhoods (cache-friendly, O(d log d)).
    fn dedup_sort_small_ac(&mut self, raw: &mut [Neighbor<T>], entries: &mut Vec<(u32, T)>) {
        self.dedup_sort_core(raw, entries, &mut Vec::new(), false);
        sort_by_weight_then_index(entries);
    }

    /// Sort-based dedup for small neighborhoods — AC2 variant.
    fn dedup_sort_small_ac2(
        &mut self,
        raw: &mut [Neighbor<T>],
        entries: &mut Vec<(u32, T)>,
        counts: &mut Vec<u32>,
        merge_limit: u32,
    ) {
        self.dedup_sort_core(raw, entries, counts, true);
        Self::apply_merge_limit(entries, counts, merge_limit, &mut self.merged_counts);
        self.sort_by_avg_weight(entries, counts);
    }

    /// Allocate scatter buffers on first use (when a neighborhood exceeds SCATTER_THRESHOLD).
    fn ensure_scatter_buffers(&mut self) {
        if self.scatter.len() < self.n {
            self.scatter.resize(self.n, T::zero());
            self.scatter_seen.resize(self.n, false);
        }
    }

    /// Scatter-gather dedup core: accumulate weights (and optionally counts) via scatter buffers.
    ///
    /// When `track_counts` is true (AC2 path), uses `scatter_counts` for seen-tracking and
    /// accumulates edge counts. When false (AC path), uses `scatter_seen` for seen-tracking
    /// and records merged vertex indices in `self.merged`.
    fn dedup_scatter_core(
        &mut self,
        raw: &[Neighbor<T>],
        entries: &mut Vec<(u32, T)>,
        counts: &mut Vec<u32>,
        track_counts: bool,
    ) {
        self.ensure_scatter_buffers();
        self.unique.clear();
        self.merged.clear();
        entries.clear();

        if track_counts {
            if self.scatter_counts.is_empty() {
                self.scatter_counts.resize(self.n, 0);
            }
            self.merged_counts.clear();
            counts.clear();

            for nbr in raw.iter() {
                if self.scatter_counts[nbr.to as usize] == 0 {
                    self.unique.push(nbr.to);
                }
                self.scatter[nbr.to as usize] = self.scatter[nbr.to as usize] + nbr.fill_weight;
                self.scatter_counts[nbr.to as usize] =
                    self.scatter_counts[nbr.to as usize].saturating_add(nbr.count);
            }

            for &idx in &self.unique {
                entries.push((idx, self.scatter[idx as usize]));
                counts.push(self.scatter_counts[idx as usize]);
                self.scatter[idx as usize] = T::zero();
                self.scatter_counts[idx as usize] = 0;
            }
        } else {
            for nbr in raw.iter() {
                if !self.scatter_seen[nbr.to as usize] {
                    self.scatter_seen[nbr.to as usize] = true;
                    self.unique.push(nbr.to);
                } else {
                    self.merged.push(nbr.to);
                }
                self.scatter[nbr.to as usize] = self.scatter[nbr.to as usize] + nbr.fill_weight;
            }

            for &idx in &self.unique {
                entries.push((idx, self.scatter[idx as usize]));
                self.scatter[idx as usize] = T::zero();
                self.scatter_seen[idx as usize] = false;
            }
        }
    }

    /// Scatter-gather dedup for large neighborhoods (O(d) dedup, O(d log d) final sort).
    fn dedup_scatter_ac(&mut self, raw: &[Neighbor<T>], entries: &mut Vec<(u32, T)>) {
        self.dedup_scatter_core(raw, entries, &mut Vec::new(), false);
        sort_by_weight_then_index(entries);
    }

    /// Scatter-gather dedup for large neighborhoods — AC2 variant.
    fn dedup_scatter_ac2(
        &mut self,
        raw: &[Neighbor<T>],
        entries: &mut Vec<(u32, T)>,
        counts: &mut Vec<u32>,
        merge_limit: u32,
    ) {
        self.dedup_scatter_core(raw, entries, counts, true);
        Self::apply_merge_limit(entries, counts, merge_limit, &mut self.merged_counts);
        self.sort_by_avg_weight(entries, counts);
    }

    /// Deduplicate raw 3-tuples for the AC2 (multi-edge approximate Cholesky) path.
    ///
    /// Takes `raw` entries `(vertex_index, weight, count)` with potential duplicate
    /// vertex indices. Merges duplicates by summing weights and counts, then writes
    /// deduplicated results into `entries` (as `(vertex_index, total_weight)`) and
    /// `counts` (edge multiplicities), which are parallel vecs sorted by average
    /// weight (`total_weight / count`) ascending.
    ///
    /// Edge multiplicities exceeding `merge_limit` are capped: excess count is
    /// discarded while total weight is preserved (the per-edge weight increases).
    /// After the call, [`merged_counts()`](Self::merged_counts) returns
    /// `(vertex_index, discarded_count)` pairs for entries that were capped.
    ///
    /// Uses a sort-based path for small inputs (`<= SCATTER_THRESHOLD`) and a
    /// scatter-gather path for larger neighborhoods.
    pub fn dedup_ac2(
        &mut self,
        raw: &mut [Neighbor<T>],
        entries: &mut Vec<(u32, T)>,
        counts: &mut Vec<u32>,
        merge_limit: u32,
    ) {
        if raw.len() <= SCATTER_THRESHOLD {
            self.dedup_sort_small_ac2(raw, entries, counts, merge_limit);
        } else {
            self.dedup_scatter_ac2(raw, entries, counts, merge_limit);
        }
    }

    /// Apply merge limit: cap multi-edge counts, preserving total weight (AC2 only).
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

    /// Sort entries by average weight ascending, then by vertex index (AC2 only).
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
