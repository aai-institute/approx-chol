use num_traits::PrimInt;

use super::Biadjacency;
use crate::input::{Csr, CsrError};
use crate::Real;

/// Unions the pattern in the walk that checks symmetry and range.
pub(super) fn of_csr<T: Real, I: PrimInt>(csr: &Csr<'_, T, I>) -> Result<Partition, CsrError> {
    todo!()
}

pub(super) fn of_cross<T: Real>(c: &Biadjacency<T>) -> Partition {
    let mut sets = UnionFind::new(c.n());
    c.joins().for_each(|(i, j)| sets.union(i, j));
    sets.into_partition()
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
struct UnionFind {
    parent: Vec<u32>,
    size: Vec<u32>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        todo!()
    }

    fn union(&mut self, i: usize, j: usize) {
        todo!()
    }

    /// Components ordered by lowest vertex, members ascending.
    fn into_partition(self) -> Partition {
        todo!()
    }
}
