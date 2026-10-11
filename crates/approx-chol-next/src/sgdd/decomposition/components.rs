use num_traits::PrimInt;

use crate::input::{Canonical, CsrError, UpperCanonical};
use crate::Real;

/// Unions the pattern in the walk that checks symmetry and range.
pub(super) fn of_csr<T: Real, I: PrimInt>(
    csr: &Canonical<'_, T, I>,
) -> Result<Partition, CsrError> {
    let mut sets = UnionFind::new(csr.n());
    csr.walk(|i, j, _| sets.union(i, j))?;
    Ok(sets.into_partition())
}

/// Unions the pattern in the walk that checks range: upper storage has no mirror to check.
pub(super) fn of_upper<T: Real, I: PrimInt>(
    csr: &UpperCanonical<'_, T, I>,
) -> Result<Partition, CsrError> {
    let mut sets = UnionFind::new(csr.n());
    csr.walk(|i, j, _| sets.union(i, j))?;
    Ok(sets.into_partition())
}

/// Every vertex once, components back to back.
pub(super) struct Partition {
    order: Vec<u32>,
    ends: Vec<u32>,
    /// Each vertex's index within its component: extraction remaps without a per-component map.
    position: Vec<u32>,
}

impl Partition {
    pub(super) fn n(&self) -> usize {
        self.position.len()
    }

    pub(super) fn position(&self) -> &[u32] {
        &self.position
    }

    pub(super) fn is_whole(&self) -> bool {
        self.ends.len() == 1
    }

    pub(super) fn components(&self) -> impl Iterator<Item = &[u32]> + '_ {
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
    fn find(&mut self, mut i: usize) -> usize {
        while self.parent[i] as usize != i {
            let grandparent = self.parent[self.parent[i] as usize];
            self.parent[i] = grandparent;
            i = grandparent as usize;
        }
        i
    }

    pub(super) fn union(&mut self, i: usize, j: usize) {
        let (mut a, mut b) = (self.find(i), self.find(j));
        if a == b {
            return;
        }
        if self.size[a] < self.size[b] {
            std::mem::swap(&mut a, &mut b);
        }
        self.parent[b] = a as u32;
        self.size[a] += self.size[b];
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
            let order: Vec<u32> = (0..n as u32).collect();
            let ends = vec![n as u32];
            return Partition {
                position: order.clone(),
                order,
                ends,
            };
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
