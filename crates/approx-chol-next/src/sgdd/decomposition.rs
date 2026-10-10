use num_traits::PrimInt;

mod components;

use super::{Component, Proof, ScalingPolicy, Sgdd, Unreduced};
use crate::input::{BipartiteGram, BipartiteGraph, Csr, CsrError, FactorWidth2, IntoColumns};
use crate::storage::{CrossBlock, Rows, UpperAdjacency};
use crate::{Real, ReduceError};
use components::Partition;

/// One input split into components: what each looks like unreduced, and which components provably float.
pub(super) struct Decomposition<'v, T, I> {
    partition: Partition,
    input: Input<'v, T, I>,
}

/// The input, and what its components' own rows cannot prove: an assembled `A` is taken whole by a lone component.
enum Input<'v, T, I> {
    /// Each component's own rows decide its floating.
    Csr(Csr<'v, T, I>),
    /// A signless bipartite Laplacian, its diagonal `C`'s signed row and column sums: singular by shape, so every component floats untested.
    Signless(Option<Biadjacency<T>>),
    /// `C` of `[D_l C; Cᵀ D_r]`, the sides' diagonals read from the Gram; a bipartite component takes the block elimination.
    BipartiteGram {
        c: Option<Biadjacency<T>>,
        gram: BipartiteGram<'v, T, I>,
    },
    /// A general factor-width-2 `A`, as its CSR would store it.
    Gram {
        a: Option<Assembled<T>>,
        v: FactorWidth2<'v, T, I>,
    },
}

impl<'v, T: Real, I: PrimInt> Decomposition<'v, T, I> {
    /// Checks symmetry and range in the walk that finds the components.
    pub(super) fn csr(csr: Csr<'v, T, I>) -> Result<Self, CsrError> {
        Ok(Self {
            partition: components::of_csr(&csr)?,
            input: Input::Csr(csr),
        })
    }

    pub(super) fn gram(v: FactorWidth2<'v, T, I>) -> Result<Self, ReduceError> {
        let a = assemble_general(v)?;
        Ok(Self {
            partition: components::of_general(&a),
            input: Input::Gram { a: Some(a), v },
        })
    }

    pub(super) fn bipartite_gram(gram: BipartiteGram<'v, T, I>) -> Result<Self, ReduceError> {
        let c = assemble_bipartite(gram, gram.n_left())?;
        Ok(Self {
            partition: components::of_cross(&c),
            input: Input::BipartiteGram { c: Some(c), gram },
        })
    }

    /// A negative weight leaves a diagonal below its row, so it fails as not GDD.
    pub(super) fn signless(graph: BipartiteGraph<'v, T, I>) -> Result<Self, ReduceError> {
        let c = assemble_bipartite(graph, graph.n_left())?;
        Ok(Self {
            partition: components::of_cross(&c),
            input: Input::Signless(Some(c)),
        })
    }

    /// Dead components are dropped; a CSR's float on their own rows, a view's once every one is reduced.
    pub(super) fn reduce(self, policy: &ScalingPolicy) -> Result<Sgdd<T>, ReduceError> {
        let Self {
            partition,
            mut input,
        } = self;
        let proof = match input {
            Input::Csr(_) => Proof::Rows,
            Input::Signless(_) | Input::BipartiteGram { .. } | Input::Gram { .. } => Proof::Input,
        };
        let (mut components, mut dead) = (Vec::new(), Vec::new());
        for a in input.components(&partition) {
            if a.is_dead() {
                dead.push(a.vertices[0]);
                continue;
            }
            components.push(a.reduce(policy, proof)?);
        }
        match &input {
            Input::Csr(_) => {}
            Input::Signless(_) => components.iter_mut().for_each(Component::float),
            Input::BipartiteGram { gram, .. } => float_certified(*gram, &mut components),
            Input::Gram { v, .. } => float_certified(*v, &mut components),
        }
        Ok(Sgdd {
            n: partition.n(),
            components,
            dead: dead.into_boxed_slice(),
        })
    }
}

impl<T: Real, I: PrimInt> Input<'_, T, I> {
    /// A lone vertex is general, so a bipartite component always has both sides.
    fn components<'s>(
        &'s mut self,
        partition: &'s Partition,
    ) -> impl Iterator<Item = Unreduced<T>> + 's {
        partition
            .components()
            .map(|vertices| -> Unreduced<T> { todo!() })
    }
}

/// Floats each folded component whose scaling meets `A_c`'s kernel to rounding: with `f = F·1` over its vertices and `p_k = f_a v_a + f_b v_b`, `Σ_k bilinear_k(p_k, p_k) ≤ 5ε Σ_k [bilinear_k(f_a v_a, f_a v_a) + bilinear_k(f_b v_b, f_b v_b)]`, `ε` that of `T`, summed in `f64` in column order over nonzero weights, as a zero-weight column is no part of `A`; a column straddling components adds its own end's square to both sides, a zero or non-finite sum grounds, and a frustrated component is nonsingular and never floats.
fn float_certified<T: Real>(v: impl IntoColumns<T>, components: &mut [Component<T>]) {
    todo!()
}

/// `C`, accumulated in a dense table when it is small (+54% without it), unfiltered: a column that couples nothing adds an exact zero and zero cells drop; a cell leaving `in_range` is `OutOfRange`.
fn assemble_bipartite<T: Real>(
    v: impl IntoColumns<T>,
    n_left: usize,
) -> Result<Biadjacency<T>, ReduceError> {
    todo!()
}

/// `A`'s strict upper triangle and diagonal, cells `bilinear(v_a, v_b)` summed by pair, zero cells dropped; a value leaving `in_range` is `OutOfRange`.
fn assemble_general<T: Real>(v: impl IntoColumns<T>) -> Result<Assembled<T>, ReduceError> {
    todo!()
}

/// A general factor-width-2 `A`, assembled: the pattern its components are found in.
struct Assembled<T> {
    adjacency: UpperAdjacency<T>,
    diagonal: Vec<T>,
}

/// The biadjacency `C` of a factor-width-2 view, cancelled cells dropped: rows are `A`'s left vertices.
#[derive(Clone, Debug)]
struct Biadjacency<T> {
    c: Rows<T>,
}

impl<T: Real> Biadjacency<T> {
    /// Panics on a stored zero: a cancelled cell couples nothing.
    fn new(c: Rows<T>) -> Self {
        todo!()
    }

    fn n(&self) -> usize {
        self.c.n_rows() + self.c.n_cols()
    }

    /// Each stored cell as an edge of `A`: left row `i` is vertex `i`, right column `j` is `n_left + j`.
    fn cells(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
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
