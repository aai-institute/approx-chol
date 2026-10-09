use num_traits::PrimInt;

use crate::input::{Csr, CsrError};
use crate::storage::Biadjacency;
use crate::Real;

pub(super) fn of_csr<T: Real, I: PrimInt>(csr: &Csr<'_, T, I>) -> Result<Partition, CsrError> {
    let mut sets = UnionFind::new(csr.n());
    csr.walk(|i, j| sets.union(i, j))?;
    Ok(sets.into_partition())
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
}

impl Partition {
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
