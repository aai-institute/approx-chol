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
        for i in 0..n_left {
            let mut row = c.row(i).map(|(j, _)| j);
            // A row of one column joins nothing: its column is found only beside a second.
            let (Some(first), Some(mut j)) = (row.next(), row.next()) else {
                continue;
            };
            if links == 0 {
                break;
            }
            let mut root = columns.find(first).0;
            loop {
                let (other, ()) = columns.find(j);
                if root != other {
                    links -= 1;
                    root = columns.link(root, other, ()).0;
                }
                let Some(next) = row.next() else { break };
                j = next;
            }
        }
        // Columns in one set and no empty row: connected without a union per row.
        if n_left > 0 && links == 0 && (0..n_left).all(|i| c.row_len(i) > 0) {
            return Self::whole(n_left + n_right);
        }
        // A row's set is its first column's, an empty row its own; a column no row reaches stands alone.
        Self::of_sets(n_left + n_right, |v| match v < n_left {
            true => c
                .row(v)
                .next()
                .map_or(v, |(j, _)| n_left + columns.find(j).0),
            false => n_left + columns.find(v - n_left).0,
        })
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

    /// Components ordered by lowest vertex, members ascending, from each vertex's representative.
    fn of_sets(n: usize, mut root: impl FnMut(usize) -> usize) -> Self {
        let mut label = vec![I::max_value(); n];
        let mut counts: Vec<I> = Vec::new();
        let component: Vec<I> = (0..n)
            .map(|v| {
                let root = root(v);
                if label[root] == I::max_value() {
                    label[root] = I::from_index(counts.len());
                    counts.push(I::zero());
                }
                let count = &mut counts[label[root].index()];
                *count = *count + I::one();
                label[root]
            })
            .collect();
        if counts.len() == 1 {
            return Self::whole(n);
        }
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
        Self {
            order,
            ends,
            position: Some(position),
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
        Partition::of_sets(n, |v| self.find(v).0)
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::Xoshiro256PlusPlus;
    use rand::{RngExt, SeedableRng};

    use super::*;

    /// Read off the column forest, empty rows and columns alone, the partition is the one a union per cell finds.
    #[test]
    fn bipartite_components_with_empty_rows_and_columns_match_a_union_per_cell() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        for case in 0..400 {
            let (n_left, n_right) = (rng.random_range(0..9usize), rng.random_range(0..9usize));
            let groups = rng.random_range(1..4u32);
            let (mut ptrs, mut indices) = (vec![0], Vec::new());
            for i in 0..n_left {
                indices.extend((0..n_right as u32).filter(|&j| {
                    j % groups == i as u32 % groups
                        && rng.random_range(0..3) == 0
                        && j % 4 != case % 5
                }));
                ptrs.push(indices.len());
            }
            let values = vec![1.0_f64; indices.len()];
            let c = Rows::new(ptrs, indices, values, n_right);
            let mut sets = Forest::<u32, ()>::new(n_left + n_right);
            for i in 0..n_left {
                for (j, _) in c.row(i) {
                    sets.union(i, n_left + j);
                }
            }
            let (got, want) = (Partition::of_bipartite(&c), sets.into_partition());
            assert_eq!(got.order, want.order, "case {case}");
            assert_eq!(got.ends, want.ends, "case {case}");
            assert_eq!(got.position, want.position, "case {case}");
        }
    }
}
