use crate::storage::Rows;
use crate::{Index, Real};

/// Every vertex once, components back to back.
pub(in crate::sgdd) struct Partition<I> {
    order: Vec<I>,
    ends: Vec<I>,
    /// Each vertex's index within its component, none when whole: extraction remaps without a per-component map.
    position: Option<Vec<I>>,
}

impl<I: Index> Partition<I> {
    /// The components of `C`'s bipartite graph, its rows numbered before its columns.
    pub(super) fn of_bipartite<T: Real>(c: &Rows<T, I>) -> Self {
        let (n_left, n_right) = (c.n_rows(), c.n_cols());
        // A row joins its columns; once they are one set, no cell joins anything new.
        let (mut columns, mut links) = (UnionFind::<I>::new(n_right), n_right.saturating_sub(1));
        'rows: for i in 0..n_left {
            let mut root = None;
            for (j, _) in c.row(i) {
                if links == 0 {
                    break 'rows;
                }
                let other = columns.find(j);
                root = match root {
                    Some(root) if root != other => {
                        links -= 1;
                        Some(columns.join(root, other))
                    }
                    _ => Some(other),
                };
            }
        }
        // Columns in one set and no empty row: connected without a union per row.
        if n_left > 0 && links == 0 && (0..n_left).all(|i| c.row_len(i) > 0) {
            return Self::whole(n_left + n_right);
        }
        let mut sets = UnionFind::new(n_left + n_right);
        for j in 0..n_right {
            sets.union(n_left + j, n_left + columns.find(j));
        }
        for i in 0..n_left {
            if let Some((j, _)) = c.row(i).next() {
                sets.union(i, n_left + j);
            }
        }
        sets.into_partition()
    }

    /// One component of `n > 0` vertices: every vertex is its own position.
    fn whole(n: usize) -> Self {
        Self {
            order: (0..n).map(I::from_index).collect(),
            ends: vec![I::from_index(n)],
            position: None,
        }
    }

    pub(in crate::sgdd) fn n(&self) -> usize {
        self.order.len()
    }

    /// `None` when whole, as every vertex is its own position.
    pub(super) fn position(&self) -> Option<&[I]> {
        self.position.as_deref()
    }

    pub(in crate::sgdd) fn components(&self) -> impl Iterator<Item = &[I]> + '_ {
        self.ends.iter().scan(0usize, |start, &end| {
            let component = &self.order[*start..end.index()];
            *start = end.index();
            Some(component)
        })
    }
}

/// Union by size with path compression.
pub(super) struct UnionFind<I> {
    parent: Vec<I>,
    size: Vec<I>,
}

impl<I: Index> UnionFind<I> {
    pub(super) fn new(n: usize) -> Self {
        Self {
            parent: (0..n).map(I::from_index).collect(),
            size: vec![I::one(); n],
        }
    }

    /// Path halving: every visited vertex skips to its grandparent.
    fn find(&mut self, mut i: usize) -> usize {
        while self.parent[i].index() != i {
            let grandparent = self.parent[self.parent[i].index()];
            self.parent[i] = grandparent;
            i = grandparent.index();
        }
        i
    }

    pub(super) fn union(&mut self, i: usize, j: usize) {
        let (a, b) = (self.find(i), self.find(j));
        if a != b {
            self.join(a, b);
        }
    }

    /// Links two distinct roots by size, returning the joined root.
    fn join(&mut self, mut a: usize, mut b: usize) -> usize {
        if self.size[a] < self.size[b] {
            std::mem::swap(&mut a, &mut b);
        }
        self.parent[b] = I::from_index(a);
        self.size[a] = self.size[a] + self.size[b];
        a
    }

    /// Whether every vertex is already in one set, which is the connected case.
    fn is_one_set(&mut self) -> bool {
        let n = self.parent.len();
        n > 0 && {
            let root = self.find(0);
            self.size[root].index() == n
        }
    }

    /// Components ordered by lowest vertex, members ascending.
    pub(super) fn into_partition(mut self) -> Partition<I> {
        let n = self.parent.len();
        // Connected, the common case, skips the counting sort: every vertex is its own position.
        if self.is_one_set() {
            return Partition::whole(n);
        }
        let mut label = vec![I::max_value(); n];
        let mut counts: Vec<I> = Vec::new();
        let component: Vec<I> = (0..n)
            .map(|v| {
                let root = self.find(v);
                if label[root] == I::max_value() {
                    label[root] = I::from_index(counts.len());
                    counts.push(I::zero());
                }
                let count = &mut counts[label[root].index()];
                *count = *count + I::one();
                label[root]
            })
            .collect();
        let ends: Vec<I> = (counts.iter())
            .scan(I::zero(), |end, &count| {
                *end = *end + count;
                Some(*end)
            })
            .collect();
        let mut next: Vec<I> = (ends.iter().zip(&counts))
            .map(|(&end, &count)| end - count)
            .collect();
        let (mut order, mut position) = (vec![I::zero(); n], vec![I::zero(); n]);
        for (v, &c) in component.iter().enumerate() {
            let c = c.index();
            let slot = &mut next[c];
            order[slot.index()] = I::from_index(v);
            position[v] = *slot - (ends[c] - counts[c]);
            *slot = *slot + I::one();
        }
        Partition {
            order,
            ends,
            position: Some(position),
        }
    }
}
