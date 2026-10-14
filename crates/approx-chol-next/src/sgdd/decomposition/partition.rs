use crate::storage::Rows;
use crate::Real;

/// Every vertex once, components back to back.
pub(in crate::sgdd) struct Partition {
    order: Vec<u32>,
    ends: Vec<u32>,
    /// Each vertex's index within its component: extraction remaps without a per-component map.
    position: Vec<u32>,
}

impl Partition {
    /// The components of `C`'s bipartite graph, its rows numbered before its columns.
    pub(super) fn of_bipartite<T: Real>(c: &Rows<T>) -> Self {
        let (n_left, n_right) = (c.n_rows(), c.n_cols());
        // A row joins its columns; once they are one set, no cell joins anything new.
        let (mut columns, mut links) = (UnionFind::new(n_right), n_right.saturating_sub(1));
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
    pub(super) fn whole(n: usize) -> Self {
        let order: Vec<u32> = (0..n as u32).collect();
        Self {
            position: order.clone(),
            order,
            ends: vec![n as u32],
        }
    }

    pub(in crate::sgdd) fn n(&self) -> usize {
        self.position.len()
    }

    pub(super) fn position(&self) -> &[u32] {
        &self.position
    }

    pub(super) fn is_whole(&self) -> bool {
        self.ends.len() == 1
    }

    pub(in crate::sgdd) fn components(&self) -> impl Iterator<Item = &[u32]> + '_ {
        self.ends.iter().scan(0usize, |start, &end| {
            let component = &self.order[*start..end as usize];
            *start = end as usize;
            Some(component)
        })
    }
}

/// Union by size with path compression.
pub(super) struct UnionFind {
    parent: Vec<u32>,
    size: Vec<u32>,
}

impl UnionFind {
    pub(super) fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
            size: vec![1; n],
        }
    }

    /// Path halving: every visited vertex skips to its grandparent.
    pub(super) fn find(&mut self, mut i: usize) -> usize {
        while self.parent[i] as usize != i {
            let grandparent = self.parent[self.parent[i] as usize];
            self.parent[i] = grandparent;
            i = grandparent as usize;
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
    pub(super) fn join(&mut self, mut a: usize, mut b: usize) -> usize {
        if self.size[a] < self.size[b] {
            std::mem::swap(&mut a, &mut b);
        }
        self.parent[b] = a as u32;
        self.size[a] += self.size[b];
        a
    }

    /// Whether every vertex is already in one set, which is the connected case.
    fn is_one_set(&mut self) -> bool {
        let n = self.parent.len();
        n > 0 && {
            let root = self.find(0);
            self.size[root] as usize == n
        }
    }

    /// Components ordered by lowest vertex, members ascending.
    pub(super) fn into_partition(mut self) -> Partition {
        let n = self.parent.len();
        // Connected, the common case, skips the counting sort: every vertex is its own position.
        if self.is_one_set() {
            return Partition::whole(n);
        }
        let mut label = vec![u32::MAX; n];
        let mut counts: Vec<u32> = Vec::new();
        let component: Vec<u32> = (0..n)
            .map(|v| {
                let root = self.find(v);
                if label[root] == u32::MAX {
                    label[root] = counts.len() as u32;
                    counts.push(0);
                }
                counts[label[root] as usize] += 1;
                label[root]
            })
            .collect();
        let ends: Vec<u32> = (counts.iter())
            .scan(0, |end, &count| {
                *end += count;
                Some(*end)
            })
            .collect();
        let mut next: Vec<u32> = (ends.iter().zip(&counts))
            .map(|(end, count)| end - count)
            .collect();
        let (mut order, mut position) = (vec![0; n], vec![0; n]);
        for (v, &c) in component.iter().enumerate() {
            let slot = &mut next[c as usize];
            order[*slot as usize] = v as u32;
            position[v] = *slot - (ends[c as usize] - counts[c as usize]);
            *slot += 1;
        }
        Partition {
            order,
            ends,
            position,
        }
    }
}
