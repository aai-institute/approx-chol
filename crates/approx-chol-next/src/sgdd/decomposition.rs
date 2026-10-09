use num_traits::PrimInt;

use super::components::{self, Partition};
use super::{Component, Part};
use crate::input::{Columns, Csr, CsrError, Gram, Graph};
use crate::storage::Biadjacency;
use crate::Real;

/// One input split into components: what each looks like unreduced, and which parts provably float.
pub(super) struct Decomposition<'v, T, I> {
    input: Input<'v, T, I>,
    partition: Partition,
}

enum Input<'v, T, I> {
    /// A signless Laplacian: its diagonal is `|C|`'s row sums, so every part floats.
    Signless(Biadjacency<T>),
    /// A Gram, its diagonal given; floats when `fᵀGf ≤ 10u·fᵀDf`, summed as squares per column of `V`.
    Gram {
        columns: Columns<'v, T, I>,
        c: Biadjacency<T>,
        left: &'v [T],
        right: &'v [T],
    },
    /// Floats when each row's surplus is within the rounding its own additions can leave.
    Csr(Csr<'v, T, I>),
}

impl<'v, T: Real, I: PrimInt> Decomposition<'v, T, I> {
    /// Checks symmetry and finiteness in the walk that finds the components.
    pub(super) fn csr(csr: Csr<'v, T, I>) -> Result<Self, CsrError> {
        let partition = components::of_csr(&csr)?;
        Ok(Self {
            input: Input::Csr(csr),
            partition,
        })
    }

    pub(super) fn gram(gram: Gram<'v, 'v, T, I>) -> Self {
        let (left, right) = gram.diagonals();
        let columns = gram.columns();
        let c = columns.assemble();
        let partition = components::of_cross(&c);
        Self {
            input: Input::Gram {
                columns,
                c,
                left,
                right,
            },
            partition,
        }
    }

    pub(super) fn signless(graph: Graph<'v, T, I>) -> Self {
        let c = graph.columns().assemble();
        let partition = components::of_cross(&c);
        Self {
            input: Input::Signless(c),
            partition,
        }
    }

    pub(super) fn n(&self) -> usize {
        match &self.input {
            Input::Signless(c) | Input::Gram { c, .. } => c.n(),
            Input::Csr(csr) => csr.n(),
        }
    }

    /// A lone vertex is general, so a bipartite component always has both sides.
    pub(super) fn components(&self) -> impl Iterator<Item = Component<T>> + '_ {
        (self.partition.components()).map(|vertices| -> Component<T> { todo!() })
    }

    /// One pass over the input decides every part's ground, matching parts by their vertices.
    pub(super) fn float(&self, parts: &mut [Part<T>]) {
        match &self.input {
            Input::Signless(_) => parts.iter_mut().for_each(|part| part.m.float()),
            Input::Gram { .. } | Input::Csr(_) => todo!(),
        }
    }
}
