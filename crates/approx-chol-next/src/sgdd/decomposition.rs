use num_traits::PrimInt;

use super::components::{self, Partition};
use super::{Component, Part};
use crate::input::{Columns, Csr, CsrError, Gram, Graph};
use crate::storage::Biadjacency;
use crate::{Real, ReduceError};

/// One input split into components: what each looks like unreduced, and which parts provably float.
pub(super) struct Decomposition<'v, T, I> {
    n: usize,
    partition: Partition,
    /// A view's assembled `C`, taken whole by a lone component instead of copied block by block.
    c: Option<Biadjacency<T>>,
    input: Input<'v, T, I>,
}

/// What the floating test reads.
enum Input<'v, T, I> {
    /// A signless Laplacian: its diagonal sums each vertex's column weights, so a negative one fails as not GDD.
    Signless,
    /// A Gram, its diagonal given; floats when `fᵀGf ≤ 10u·fᵀDf`, summed as squares per column of `V`.
    Gram {
        columns: Columns<'v, T, I>,
        left: &'v [T],
        right: &'v [T],
    },
    /// Floats when each row's surplus is within the rounding its own additions can leave.
    Csr(Csr<'v, T, I>),
}

impl<'v, T: Real, I: PrimInt> Decomposition<'v, T, I> {
    /// Checks symmetry and range in the walk that finds the components.
    pub(super) fn csr(csr: Csr<'v, T, I>) -> Result<Self, CsrError> {
        Ok(Self {
            n: csr.n(),
            partition: components::of_csr(&csr)?,
            c: None,
            input: Input::Csr(csr),
        })
    }

    pub(super) fn gram(gram: Gram<'v, 'v, T, I>) -> Result<Self, ReduceError> {
        let (left, right) = gram.diagonals();
        let columns = gram.columns();
        let input = Input::Gram {
            columns,
            left,
            right,
        };
        Self::of_cross(columns.assemble()?, input)
    }

    pub(super) fn signless(graph: Graph<'v, T, I>) -> Result<Self, ReduceError> {
        Self::of_cross(graph.columns().assemble()?, Input::Signless)
    }

    fn of_cross(c: Biadjacency<T>, input: Input<'v, T, I>) -> Result<Self, ReduceError> {
        Ok(Self {
            n: c.n(),
            partition: components::of_cross(&c),
            c: Some(c),
            input,
        })
    }

    pub(super) fn n(&self) -> usize {
        self.n
    }

    /// A lone vertex is general, so a bipartite component always has both sides.
    pub(super) fn components(&mut self) -> impl Iterator<Item = Component<T>> + '_ {
        (self.partition.components()).map(|vertices| -> Component<T> { todo!() })
    }

    /// One pass over the input decides every part's ground, matching parts by their vertices.
    pub(super) fn float(&self, parts: &mut [Part<T>]) {
        match &self.input {
            Input::Signless => (parts.iter_mut()).for_each(|part| part.m.float()),
            Input::Gram { .. } | Input::Csr(_) => todo!(),
        }
    }
}
