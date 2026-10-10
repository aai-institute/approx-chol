use num_traits::PrimInt;

mod components;

use super::{Component, Proof, ScalingPolicy, Sgdd};
use crate::input::{Columns, Csr, CsrError, Gram, Graph, Observation};
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
    /// Floats when the signed alias `u` (`v_l u_l + v_r u_r = 0` along joining columns inside a part; grown in `f64` whatever `T`, exponent tracked, then scaled to `max|u| = 1`) has finite `Σ w (v_l u_l + v_r u_r)² ≤ 1e-13·Σ w (v_l² u_l² + v_r² u_r²)` in column order; a straddling column adds only its own side's square.
    Gram {
        c: Option<Biadjacency<T>>,
        columns: Columns<'v, T, I>,
        left: &'v [T],
        right: &'v [T],
    },
    /// Each part's own rows decide its floating.
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

    /// Dead components produce no part; a CSR's parts float on their own rows, a view's once every part is built.
    pub(super) fn reduce(mut self, policy: &ScalingPolicy) -> Result<Sgdd<T>, ReduceError> {
        let proof = match self.input {
            Input::Csr(_) => Proof::Rows,
            Input::Signless(_) | Input::Gram { .. } => Proof::Input,
        };
        let (mut parts, mut dead) = (Vec::new(), Vec::new());
        for a in self.components() {
            if a.is_dead() {
                dead.push(a.vertices[0]);
                continue;
            }
            parts.push(a.reduce(policy, proof)?);
        }
        match &self.input {
            Input::Signless(_) => (parts.iter_mut()).for_each(|part| part.m.float()),
            Input::Gram { .. } => todo!(),
            Input::Csr(_) => {}
        }
        Ok(Sgdd {
            n: self.partition.n(),
            parts,
            dead: dead.into_boxed_slice(),
        })
    }

    /// A lone vertex is general, so a bipartite component always has both sides.
    fn components(&mut self) -> impl Iterator<Item = Component<T>> + '_ {
        (self.partition.components()).map(|vertices| -> Component<T> { todo!() })
    }
}

/// `C`, accumulated in a dense table when it is small (+54% without it), unfiltered: a column that joins nothing adds an exact zero and zero cells drop; a fold of nonzero factors leaving `in_range` is `OutOfRange`.
fn assemble<T: Real, I: PrimInt>(
    columns: &Columns<'_, T, I>,
) -> Result<Biadjacency<T>, ReduceError> {
    todo!()
}

impl<T: Real> Observation<T> {
    /// Couples both rows, so the floating forest may grow along it; tested factor by factor so no product underflows.
    fn joins(&self) -> bool {
        self.weight != T::zero() && self.v_left != T::zero() && self.v_right != T::zero()
    }
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
