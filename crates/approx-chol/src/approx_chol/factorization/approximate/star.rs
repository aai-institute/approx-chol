use super::ordering::{DegreeDeltas, DynamicOrdering};
use crate::graph::{AdjListGraph, EdgeCount, Neighbor};
use crate::types::{float_total_cmp, Real};

/// One entry of a deduped star: a neighbor, the weight its surviving copies carry
/// between them, and how many that is.
///
/// The copies are stored the way the graph stores them, so a single-copy layout
/// spends no space on a count it knows and no arithmetic on dividing by it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct StarEntry<T, C> {
    pub neighbor: u32,
    pub copies: C,
    pub weight: T,
}

/// A pivot's deduped neighborhood: one entry per unique neighbor, ordered as the
/// clique-tree path eliminates them.
pub(super) struct Star<T: Real, C> {
    entries: Vec<StarEntry<T, C>>,
    /// Staging for the sort of a layout whose copies differ, which needs one
    /// sortable element carrying both the key and the entry it belongs to. A
    /// single-copy star sorts in place and never touches this.
    sort_scratch: Vec<(T, StarEntry<T, C>)>,
}

impl<T: Real, C: EdgeCount> Star<T, C> {
    pub(super) fn new() -> Self {
        Self {
            entries: Vec::new(),
            sort_scratch: Vec::new(),
        }
    }

    /// Every neighbor at the same multiplicity — the shape the standalone samplers
    /// are handed.
    pub(super) fn uniform(entries: &[(u32, T)], copies: C) -> Self {
        let mut star = Self::new();
        for &(neighbor, weight) in entries {
            star.push(StarEntry {
                neighbor,
                copies,
                weight,
            });
        }
        star
    }

    fn clear(&mut self) {
        self.entries.clear();
    }

    fn push(&mut self, entry: StarEntry<T, C>) {
        self.entries.push(entry);
    }

    /// Keep at most `limit` copies of `copies` merged edges to `neighbor`, reporting
    /// the discards. One call per unique neighbor, so the cap needs no second pass.
    fn push_capped(
        &mut self,
        neighbor: u32,
        weight: T,
        copies: u32,
        limit: u32,
        discarded: &mut Vec<(u32, u32)>,
    ) {
        let (copies, dropped) = C::cap(copies, limit);
        if dropped > 0 {
            discarded.push((neighbor, dropped));
        }
        self.push(StarEntry {
            neighbor,
            copies,
            weight,
        });
    }

    pub(super) fn entries(&self) -> &[StarEntry<T, C>] {
        &self.entries
    }

    /// Accumulate the degree decrease each surviving neighbor experiences from the
    /// pivot's elimination (negative deltas).
    pub(super) fn accumulate_removal_delta(&self, deltas: &mut DegreeDeltas) {
        for entry in &self.entries {
            deltas.decrease(entry.neighbor, entry.copies.get());
        }
    }

    /// Ascending by the weight one copy carries, ties by neighbor index.
    ///
    /// A star whose copies all count one orders on weight alone; otherwise the
    /// quotient is precomputed, because cross-multiplying in the comparator can
    /// break transitivity under floating-point rounding.
    fn sort(&mut self) {
        if self.entries.len() <= 1 {
            return;
        }
        if C::SINGLE_COPY {
            self.entries.sort_unstable_by(|a, b| {
                float_total_cmp(&a.weight, &b.weight).then_with(|| a.neighbor.cmp(&b.neighbor))
            });
            return;
        }
        self.sort_scratch.clear();
        self.sort_scratch.reserve(self.entries.len());
        for entry in &self.entries {
            self.sort_scratch
                .push((entry.copies.per_copy(entry.weight), *entry));
        }
        self.sort_scratch.sort_unstable_by(|a, b| {
            float_total_cmp(&a.0, &b.0).then_with(|| a.1.neighbor.cmp(&b.1.neighbor))
        });
        for (slot, &(_, entry)) in self.entries.iter_mut().zip(&self.sort_scratch) {
            *slot = entry;
        }
    }
}

/// Apply the merge-compression degree decrease immediately (not batched through
/// [`DegreeDeltas`]), so it floors at zero *before* the step's fill/removal net
/// delta — matching the per-edge baseline, where a merge driving the estimate
/// below zero loses the excess rather than offsetting later fill.
fn apply_removed_copies(merged: &[(u32, u32)], ordering: &mut DynamicOrdering) {
    for &(u, n_merged) in merged {
        ordering.decrease(u as usize, n_merged);
    }
}

/// Builds the pivot's star, one elimination step at a time.
///
/// AC and AC2 are the same builder over different edge layouts: [`EdgeCount`] holds
/// everything that differs, so the copies an edge splits into reach the degree
/// buckets, the merge cap and the sampler as one number from one place.
pub(super) struct StarBuilder<T: Real, C: EdgeCount> {
    star: Star<T, C>,
    dedup: DedupWorkspace<T, C>,
    /// Copies each edge is eliminated on, which the split already decided: the cap
    /// no neighbor pair may keep more of.
    copies: u32,
}

impl<T: Real, C: EdgeCount> StarBuilder<T, C> {
    pub(super) fn new(n: usize, copies: u32) -> Self {
        Self {
            star: Star::new(),
            dedup: DedupWorkspace::new(n),
            copies,
        }
    }

    pub(super) fn build_star(
        &mut self,
        graph: &mut AdjListGraph<C, T>,
        v: usize,
        ordering: &mut DynamicOrdering,
    ) {
        self.dedup.collect(graph, v);
        self.dedup.dedup(&mut self.star, self.copies);
        apply_removed_copies(self.dedup.removed_copies(), ordering);
    }

    /// The star the last [`build_star`](Self::build_star) produced; its entries are
    /// empty when the pivot had no live neighbor left.
    pub(super) fn star(&self) -> &Star<T, C> {
        &self.star
    }
}

/// Neighborhoods with at most this many entries use sort-based dedup (O(d log d),
/// cache-friendly for small d). Larger neighborhoods use scatter-gather (O(d) via
/// indexed buffers, but with higher constant from random-access pattern).
const SCATTER_THRESHOLD: usize = 32;

/// Shared scratch for dedup variants. Both leave every per-vertex slot at zero
/// when they finish, so a zero `count` is also what marks a vertex unvisited —
/// no separate seen-set to keep in step with it.
struct DedupScratch<T: Real> {
    /// `scatter[idx]` accumulates weight for vertex `idx`.
    scatter: Vec<T>,
    /// `counts[idx]` sums [`EdgeCount::get`] over the raw neighbors at vertex
    /// `idx`. That is an occurrence count on the AC path only because a slim edge
    /// counts one — both paths accumulate the same quantity.
    counts: Vec<u32>,
    /// Tracks unique vertex indices seen in the current pass.
    unique: Vec<u32>,
    /// Number of vertices in the graph (for buffer sizing).
    n: usize,
}

impl<T: Real> DedupScratch<T> {
    fn new(n: usize) -> Self {
        Self {
            scatter: Vec::new(),
            counts: Vec::new(),
            unique: Vec::new(),
            n,
        }
    }

    /// Clear the pass state and size the per-vertex buffers.
    fn begin_pass(&mut self) {
        if self.scatter.len() < self.n {
            self.scatter.resize(self.n, T::zero());
            self.counts.resize(self.n, 0);
        }
        self.unique.clear();
    }

    /// Accumulate one raw neighbor, recording the vertex the first time it appears.
    #[inline]
    fn accumulate(&mut self, to: u32, weight: T, count: u32) {
        let idx = to as usize;
        if self.counts[idx] == 0 {
            self.unique.push(to);
        }
        self.scatter[idx] = self.scatter[idx] + weight;
        self.counts[idx] = self.counts[idx].saturating_add(count);
    }

    /// Visit each vertex the pass accumulated, in first-seen order, resetting its
    /// slots as it goes so the buffers are all-zero again when this returns.
    #[inline]
    fn drain_unique(&mut self, mut visit: impl FnMut(u32, T, u32)) {
        for index in 0..self.unique.len() {
            let vertex = self.unique[index];
            let idx = vertex as usize;
            visit(vertex, self.scatter[idx], self.counts[idx]);
            self.scatter[idx] = T::zero();
            self.counts[idx] = 0;
        }
    }
}

/// The pivot's neighborhood on its way from the graph to a deduped star: the raw
/// buffer it is read into, the per-vertex scratch that collapses it, and the report
/// of what that collapse cost each vertex's degree.
///
/// `raw` lives here rather than in the star builder because nothing outside this
/// workspace ever reads it — it is refilled by [`collect`](Self::collect) and
/// consumed by [`dedup`](Self::dedup) on the next line.
///
/// `removed_copies` stays outside [`DedupScratch`] so the scatter path can push to
/// it from inside [`DedupScratch::drain_unique`]'s closure; one struct would
/// borrow all of `self` for the call.
pub(super) struct DedupWorkspace<T: Real, C> {
    raw: Vec<Neighbor<T, C>>,
    scratch: DedupScratch<T>,
    /// Edge copies that left each vertex's degree this pass — duplicates collapsed
    /// into one entry, or multiplicity the merge cap discarded. One decrement
    /// either way, which is why one buffer carries both.
    removed_copies: Vec<(u32, u32)>,
}

impl<T: Real, C: EdgeCount> DedupWorkspace<T, C> {
    pub fn new(n: usize) -> Self {
        Self {
            raw: Vec::new(),
            scratch: DedupScratch::new(n),
            removed_copies: Vec::new(),
        }
    }

    /// `(vertex, copies removed)` from the last dedup call.
    pub fn removed_copies(&self) -> &[(u32, u32)] {
        &self.removed_copies
    }

    /// Read the pivot's live neighbors into `raw`, replacing the previous pass.
    fn collect(&mut self, graph: &AdjListGraph<C, T>, v: usize) {
        graph.live_neighbors(v, &mut self.raw);
    }

    /// Collapse the collected neighborhood into `star`, keeping at most `limit`
    /// copies per neighbor, and order it for elimination.
    ///
    /// The two paths differ only in how they find the duplicates; both cap and emit
    /// each unique neighbor exactly once, so neither can report a merge the other
    /// would not.
    pub(super) fn dedup(&mut self, star: &mut Star<T, C>, limit: u32) {
        self.removed_copies.clear();
        star.clear();
        if self.raw.len() <= SCATTER_THRESHOLD {
            self.dedup_by_sort(star, limit);
        } else {
            self.dedup_by_scatter(star, limit);
        }
        star.sort();
    }

    fn dedup_by_sort(&mut self, star: &mut Star<T, C>, limit: u32) {
        if self.raw.is_empty() {
            return;
        }
        self.raw.sort_unstable_by_key(|n| n.to);
        let first = self.raw[0];
        let mut run = (first.to, first.fill_weight, first.count.get());
        for neighbor in &self.raw[1..] {
            if neighbor.to == run.0 {
                run.1 = run.1 + neighbor.fill_weight;
                run.2 = run.2.saturating_add(neighbor.count.get());
            } else {
                star.push_capped(run.0, run.1, run.2, limit, &mut self.removed_copies);
                run = (neighbor.to, neighbor.fill_weight, neighbor.count.get());
            }
        }
        star.push_capped(run.0, run.1, run.2, limit, &mut self.removed_copies);
    }

    fn dedup_by_scatter(&mut self, star: &mut Star<T, C>, limit: u32) {
        self.scratch.begin_pass();
        for neighbor in &self.raw {
            self.scratch
                .accumulate(neighbor.to, neighbor.fill_weight, neighbor.count.get());
        }

        let removed_copies = &mut self.removed_copies;
        self.scratch.drain_unique(|vertex, weight, copies| {
            star.push_capped(vertex, weight, copies, limit, removed_copies);
        });
    }
}

#[cfg(test)]
mod tests;
