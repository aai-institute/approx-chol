use num_traits::PrimInt;

mod components;

use super::{Proof, ScalingPolicy, Sgdd, Unreduced};
use crate::input::{Columns, Csr, CsrError, Gram, Graph, Observation};
use crate::storage::{CrossBlock, Rows};
use crate::{Real, ReduceError};
use components::Partition;

/// One input split into components: what each looks like unreduced, and which components provably float.
pub(super) struct Decomposition<'v, T, I> {
    partition: Partition,
    input: Input<'v, T, I>,
}

/// The input, and what its components' own rows cannot prove: a view's `C` is taken whole by a lone component.
enum Input<'v, T, I> {
    /// A signless Laplacian: zero surplus by shape, and a negative weight fails as not GDD.
    Signless(Option<Biadjacency<T>>),
    /// Floats when the signed alias `u` (`v_l u_l + v_r u_r = 0` along joining columns inside a component, grown from one root as an `f64` mantissa and exponent: `m' = −(m_a m)/m_b`, `e' = e + e_a − e_b` over `frexp` of the loadings, so no ratio leaves range) leaves every vertex `i` with `Σ_{k∋i} (w p_k) p_k ≤ 1e-13·a_ii u_i²`, `p_k = v_l u_l + v_r u_r` scaled by `2^{−e_i}` and summed in `f64` in column order: per vertex, so a break on light vertices is not drowned by heavy ones; a straddling column's `p` is its own side's term.
    Gram {
        c: Option<Biadjacency<T>>,
        columns: Columns<'v, T, I>,
        left: &'v [T],
        right: &'v [T],
    },
    /// Each component's own rows decide its floating.
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

    /// Dead components are dropped; a CSR's float on their own rows, a view's once every one is reduced.
    pub(super) fn reduce(mut self, policy: &ScalingPolicy) -> Result<Sgdd<T>, ReduceError> {
        let proof = match self.input {
            Input::Csr(_) => Proof::Rows,
            Input::Signless(_) | Input::Gram { .. } => Proof::Input,
        };
        let (mut components, mut dead) = (Vec::new(), Vec::new());
        for a in self.components() {
            if a.is_dead() {
                dead.push(a.vertices[0]);
                continue;
            }
            components.push(a.reduce(policy, proof)?);
        }
        match &self.input {
            Input::Signless(_) => (components.iter_mut()).for_each(|component| component.m.float()),
            Input::Gram { .. } => todo!(),
            Input::Csr(_) => {}
        }
        Ok(Sgdd {
            n: self.partition.n(),
            components,
            dead: dead.into_boxed_slice(),
        })
    }

    /// A lone vertex is general, so a bipartite component always has both sides.
    fn components(&mut self) -> impl Iterator<Item = Unreduced<T>> + '_ {
        (self.partition.components()).map(|vertices| -> Unreduced<T> { todo!() })
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
