use num_traits::PrimInt;

mod components;

use super::{Component, Part};
use crate::input::{Columns, Csr, CsrError, Gram, Graph};
use crate::storage::{CrossBlock, Rows};
use crate::{Real, ReduceError};
use components::Partition;

/// One input split into components: what each looks like unreduced, and which parts provably float.
pub(super) struct Decomposition<'v, T, I> {
    partition: Partition,
    input: Input<'v, T, I>,
}

/// The input, and what its parts' own rows cannot prove: a view's `C` is taken whole by a lone component.
enum Input<'v, T, I> {
    /// A signless Laplacian: zero surplus by shape, and a negative weight fails as not GDD.
    Signless(Option<Biadjacency<T>>),
    /// A Gram, its diagonal given; floats when `fᵀGf ≤ 10u·fᵀDf` per part, a straddling column adding `w(v f)²` to each side.
    Gram {
        c: Option<Biadjacency<T>>,
        columns: Columns<'v, T, I>,
        left: &'v [T],
        right: &'v [T],
    },
    /// Each component's rows decide its floating.
    Csr(Csr<'v, T, I>),
}

impl<'v, T: Real, I: PrimInt> Decomposition<'v, T, I> {
    /// Checks symmetry and range in the walk that finds the components.
    pub(super) fn csr(csr: Csr<'v, T, I>) -> Result<Self, CsrError> {
        Ok(Self {
            partition: components::of_csr(&csr)?,
            input: Input::Csr(csr),
        })
    }

    pub(super) fn gram(gram: Gram<'v, 'v, T, I>) -> Result<Self, ReduceError> {
        let (left, right) = gram.diagonals();
        let columns = gram.columns();
        let c = assemble(&columns)?;
        Ok(Self {
            partition: components::of_cross(&c),
            input: Input::Gram {
                c: Some(c),
                columns,
                left,
                right,
            },
        })
    }

    pub(super) fn signless(graph: Graph<'v, T, I>) -> Result<Self, ReduceError> {
        let c = assemble(&graph.columns())?;
        Ok(Self {
            partition: components::of_cross(&c),
            input: Input::Signless(Some(c)),
        })
    }

    pub(super) fn n(&self) -> usize {
        self.partition.n()
    }

    /// A lone vertex is general, so a bipartite component always has both sides.
    pub(super) fn components(&mut self) -> impl Iterator<Item = Component<T>> + '_ {
        (self.partition.components()).map(|vertices| -> Component<T> { todo!() })
    }

    /// What the parts' rows could not decide: a signless shape, or a Gram's columns in one pass.
    pub(super) fn float(&self, parts: &mut [Part<T>]) {
        match &self.input {
            Input::Signless(_) => (parts.iter_mut()).for_each(|part| part.m.float()),
            Input::Gram { .. } => todo!(),
            Input::Csr(_) => {}
        }
    }
}

/// `C`, accumulated in a dense table when it is small: within measured +54% without it.
fn assemble<T: Real, I: PrimInt>(
    columns: &Columns<'_, T, I>,
) -> Result<Biadjacency<T>, ReduceError> {
    todo!()
}

/// The biadjacency `C` of a factor-width-2 view, cancelled cells dropped: rows are `A`'s left vertices.
#[derive(Clone, Debug)]
struct Biadjacency<T> {
    c: Rows<T>,
}

impl<T: Real> Biadjacency<T> {
    /// Panics on a stored zero: a cancelled cell joins nothing.
    fn new(c: Rows<T>) -> Self {
        todo!()
    }

    fn n(&self) -> usize {
        self.c.n_rows() + self.c.n_cols()
    }

    /// Each stored cell as an edge of `A`: left row `i` is vertex `i`, right column `j` is `n_left + j`.
    fn joins(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        let n_left = self.c.n_rows();
        (0..n_left).flat_map(move |i| self.c.row(i).map(move |(j, _)| (i, n_left + j)))
    }

    /// The whole input as one component's block, moved, transposed only when the right side is larger.
    fn into_block(self) -> (Box<[u32]>, CrossBlock<T>) {
        todo!()
    }

    /// One component's block, its larger side as rows, remapped through `position`; returns its local order.
    fn block(&self, vertices: &[u32], position: &[u32]) -> (Box<[u32]>, CrossBlock<T>) {
        todo!()
    }
}
