use num_traits::PrimInt;

mod components;

use super::{Proof, ScalingPolicy, Sgdd, Unreduced};
use crate::input::{BipartiteGram, BipartiteGraph, Columns, Csr, CsrError, FactorWidth2};
use crate::storage::{CrossBlock, Rows, UpperAdjacency};
use crate::{Real, ReduceError};
use components::Partition;

/// One input split into components: what each looks like unreduced, and who proves each singular.
pub(super) struct Decomposition<'v, T, I> {
    partition: Partition,
    input: Input<'v, T, I>,
    proofs: Proofs,
}

/// What components are extracted from: the caller's CSR, or a view's assembled `A`, taken whole by a lone component.
enum Input<'v, T, I> {
    Csr(Csr<'v, T, I>),
    /// `C` of `[D_l C; Cᵀ D_r]`; a bipartite component takes the block elimination.
    Bipartite {
        c: Option<Biadjacency<T>>,
        diagonal: Diagonal<'v, T>,
    },
    /// A general factor-width-2 `A`, as its CSR would store it.
    General(Option<Assembled<T>>),
}

/// A bipartite `A`'s diagonal: the sides' own, or a signless Laplacian's degrees, `|C|`'s row and column sums.
enum Diagonal<'v, T> {
    Sides(&'v [T], &'v [T]),
    Degrees,
}

/// Who proves each component singular, settled while the whole input is still in hand.
enum Proofs {
    /// Each component's rows, once scaled.
    Rows,
    /// One verdict per component of the partition.
    Verdicts(Box<[bool]>),
}

impl Proofs {
    fn of(&self, component: usize) -> Proof {
        match self {
            Self::Rows => Proof::Rows,
            Self::Verdicts(floats) => Proof::Input(floats[component]),
        }
    }
}

impl<'v, T: Real, I: PrimInt> Decomposition<'v, T, I> {
    /// Checks symmetry and range in the walk that finds the components.
    pub(super) fn csr(csr: Csr<'v, T, I>) -> Result<Self, CsrError> {
        Ok(Self {
            partition: components::of_csr(&csr)?,
            input: Input::Csr(csr),
            proofs: Proofs::Rows,
        })
    }

    pub(super) fn gram(v: FactorWidth2<'v, T, I>) -> Result<Self, ReduceError> {
        let columns = v.columns();
        let a = assemble_general(&columns)?;
        let partition = components::of_general(&a);
        let floats = certify(&columns, &partition, |i| a.diagonal[i]);
        Ok(Self {
            partition,
            input: Input::General(Some(a)),
            proofs: Proofs::Verdicts(floats),
        })
    }

    pub(super) fn bipartite_gram(gram: BipartiteGram<'v, T, I>) -> Result<Self, ReduceError> {
        let (left, right) = gram.diagonals();
        let columns = gram.columns();
        let c = assemble_bipartite(&columns)?;
        let partition = components::of_cross(&c);
        let n_left = left.len();
        let floats = certify(&columns, &partition, |i| match i.checked_sub(n_left) {
            None => left[i],
            Some(j) => right[j],
        });
        Ok(Self {
            partition,
            input: Input::Bipartite {
                c: Some(c),
                diagonal: Diagonal::Sides(left, right),
            },
            proofs: Proofs::Verdicts(floats),
        })
    }

    /// A signless bipartite Laplacian has zero surplus by shape, and a negative weight fails as not GDD.
    pub(super) fn signless(graph: BipartiteGraph<'v, T, I>) -> Result<Self, ReduceError> {
        let c = assemble_bipartite(&graph.columns())?;
        let partition = components::of_cross(&c);
        let floats = vec![true; partition.components().count()].into_boxed_slice();
        Ok(Self {
            partition,
            input: Input::Bipartite {
                c: Some(c),
                diagonal: Diagonal::Degrees,
            },
            proofs: Proofs::Verdicts(floats),
        })
    }

    /// Dead components are dropped; every other floats as its proof says.
    pub(super) fn reduce(self, policy: &ScalingPolicy) -> Result<Sgdd<T>, ReduceError> {
        let Self {
            partition,
            mut input,
            proofs,
        } = self;
        let (mut components, mut dead) = (Vec::new(), Vec::new());
        for (index, a) in input.components(&partition).enumerate() {
            if a.is_dead() {
                dead.push(a.vertices[0]);
                continue;
            }
            components.push(a.reduce(policy, proofs.of(index))?);
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

/// Per component, the Rayleigh certificate `uᵀAu ≤ 5ε·uᵀDu`, `D = diag(A)` and `ε` that of `T`: the signed alias `u` solves `v_a u_a + v_b u_b = 0` along a maximum spanning forest of `|bilinear(v_a, v_b)|` over coupling columns, grown in `f64` as `u = μ 2^η` with `(μ_b, δ) = frexp(−m(v_a) μ_a / m(v_b))`, `η_b = η_a + e(v_a) − e(v_b) + δ`, then shifted by one power of two to `max a_ii u_i² ∈ [1, 4)`; `uᵀAu = Σ_k bilinear_k(p_k, p_k)` with `p_k = v_a u_a + v_b u_b` formed on mantissas, a straddling column keeping its own end's term, summed in column order.
fn certify<T: Real, I: PrimInt>(
    columns: &Columns<'_, T, I>,
    partition: &Partition,
    diagonal: impl Fn(usize) -> T,
) -> Box<[bool]> {
    todo!()
}

/// `C`, accumulated in a dense table when it is small (+54% without it), unfiltered: a column that couples nothing adds an exact zero and zero cells drop; a cell leaving `in_range` is `OutOfRange`.
fn assemble_bipartite<T: Real, I: PrimInt>(
    columns: &Columns<'_, T, I>,
) -> Result<Biadjacency<T>, ReduceError> {
    todo!()
}

/// `A`'s strict upper triangle and diagonal, cells `bilinear(v_a, v_b)` summed by pair, zero cells dropped; a value leaving `in_range` is `OutOfRange`.
fn assemble_general<T: Real, I: PrimInt>(
    columns: &Columns<'_, T, I>,
) -> Result<Assembled<T>, ReduceError> {
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
