use crate::sgdd::forest::Forest;
use crate::storage::Rows;
use crate::{Index, Real};

/// Every vertex once, components back to back.
pub(in crate::sgdd) struct Partition<I> {
    pub(super) order: Vec<I>,
    pub(super) ends: Vec<I>,
    /// Each vertex's index within its component, none when whole: extraction remaps without a per-component map.
    pub(super) position: Option<Vec<I>>,
}

impl<I: Index> Partition<I> {
    /// The components of `C`'s bipartite graph, its rows numbered before its columns.
    pub(super) fn of_bipartite<T: Real>(c: &Rows<T, I>) -> Self {
        let (n_left, n_right) = (c.n_rows(), c.n_cols());
        // A row joins its columns; once they are one set, no cell joins anything new.
        let (mut columns, mut links) = (Forest::<I, ()>::new(n_right), n_right.saturating_sub(1));
        'rows: for i in 0..n_left {
            let mut root = None;
            for (j, _) in c.row(i) {
                if links == 0 {
                    break 'rows;
                }
                let (other, ()) = columns.find(j);
                root = match root {
                    Some(root) if root != other => {
                        links -= 1;
                        Some(columns.link(root, other, |_| ()).0)
                    }
                    _ => Some(other),
                };
            }
        }
        // Columns in one set and no empty row: connected without a union per row.
        if n_left > 0 && links == 0 && (0..n_left).all(|i| c.row_len(i) > 0) {
            return Self::whole(n_left + n_right);
        }
        let mut sets = Forest::new(n_left + n_right);
        for j in 0..n_right {
            sets.union(n_left + j, n_left + columns.find(j).0);
        }
        for i in 0..n_left {
            if let Some((j, _)) = c.row(i).next() {
                sets.union(i, n_left + j);
            }
        }
        sets.into_partition()
    }

    /// Each component's vertices.
    pub(super) fn components(&self) -> impl Iterator<Item = &[I]> {
        (self.ends.iter()).scan(0, |start, &end| {
            Some(&self.order[std::mem::replace(start, end.index())..end.index()])
        })
    }

    /// One component of `n > 0` vertices: every vertex is its own position.
    fn whole(n: usize) -> Self {
        Self {
            order: (0..n).map(I::from_index).collect(),
            ends: vec![I::from_index(n)],
            position: None,
        }
    }
}

impl<I: Index> Forest<I, ()> {
    /// Components ordered by lowest vertex, members ascending.
    pub(super) fn into_partition(mut self) -> Partition<I> {
        let n = self.len();
        // Connected, the common case, skips the counting sort: every vertex is its own position.
        let root = (n > 0).then(|| self.find(0).0);
        if root.is_some_and(|root| self.size(root).index() == n) {
            return Partition::whole(n);
        }
        let mut label = vec![I::max_value(); n];
        let mut counts: Vec<I> = Vec::new();
        let component: Vec<I> = (0..n)
            .map(|v| {
                let (root, ()) = self.find(v);
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
