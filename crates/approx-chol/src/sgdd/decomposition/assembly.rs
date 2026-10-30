//! `A` assembled from `V`: its cells by layout, beside its diagonal or surplus, cut into components.

mod cells;

use num_traits::PrimInt;

use super::partition::{Partition, UnionFind};
use super::{Cells, LonePivot, Partitioned, PrincipalBlocks};
use crate::input::{BipartiteFactorWidth2, Column, FactorWidth2, Side};
use crate::sgdd::block::{Grounding, ReduceError};
use crate::sgdd::diagonal_in_range;
use crate::storage::{OffDiagonal, Rows, UpperAdjacency};
use crate::{narrow, widen, Index, Real};

/// How a shape's summed cells are stored, and its components found from them.
pub(in crate::sgdd) trait AdjacencyLayout {
    type Adjacency<T: Real, I: Index>: PrincipalBlocks<T, I, Block = (Box<[I]>, OffDiagonal<T, I>)>
        + Sync;

    /// The summed cells and components, `seen` visiting each column once and `pair` each nonzero cell with its two ends.
    fn sum<T: Real, J: PrimInt, I: Index>(
        self,
        v: FactorWidth2<'_, T, J>,
        seen: impl FnMut(&Column<T, I>),
        pair: impl FnMut(usize, usize, f64),
    ) -> Result<Partitioned<Self::Adjacency<T, I>, I>, ReduceError>;
}

/// A general `V`'s cells: `A`'s strict upper triangle.
#[derive(Clone, Copy, Debug)]
pub(in crate::sgdd) struct StrictUpperTriangle;

/// A bipartite `V`'s cells: the biadjacency `C` from the left part's rows.
#[derive(Clone, Copy, Debug)]
pub(in crate::sgdd) struct Biadjacency {
    n_left: usize,
    onto: Onto,
}

/// The side each bipartite block's Schur complement is onto; the other is eliminated first.
#[derive(Clone, Copy, Debug)]
pub(in crate::sgdd) enum Onto {
    /// The smaller, the right on a tie, so the larger is eliminated.
    Smaller,
    /// This side.
    Given(Side),
}

impl Biadjacency {
    /// `v`'s own left part, so the layout and the columns it sums never disagree; each block onto its smaller side.
    pub(in crate::sgdd) fn of<T: Real, J: PrimInt>(v: &FactorWidth2<'_, T, J>) -> Option<Self> {
        let onto = Onto::Smaller;
        v.n_left().map(|n_left| Self { n_left, onto })
    }

    /// `input`'s own left part, each block's complement onto `kept`.
    pub(in crate::sgdd) fn onto<T: Real, J: PrimInt>(
        input: &BipartiteFactorWidth2<'_, T, J>,
        kept: Side,
    ) -> Self {
        let n_left = input.n_left();
        let onto = Onto::Given(kept);
        Self { n_left, onto }
    }
}

/// `C` and the side each block's complement is onto.
pub(in crate::sgdd) struct Oriented<T, I> {
    c: Rows<T, I>,
    onto: Onto,
}

impl AdjacencyLayout for StrictUpperTriangle {
    type Adjacency<T: Real, I: Index> = UpperAdjacency<T, I>;

    // Out of line, the Laplacian entry measured 3–4% slower: the same work, laid out worse.
    #[inline(always)]
    fn sum<T: Real, J: PrimInt, I: Index>(
        self,
        v: FactorWidth2<'_, T, J>,
        seen: impl FnMut(&Column<T, I>),
        mut pair: impl FnMut(usize, usize, f64),
    ) -> Result<Partitioned<UpperAdjacency<T, I>, I>, ReduceError> {
        let mut sets = UnionFind::<I>::new(v.n());
        let rows = cells::upper::<T, J, I>(v, seen, |i, j, cell| {
            sets.union(i, j);
            pair(i, j, cell);
        })?;
        Ok(Partitioned {
            a: UpperAdjacency::new(rows),
            partition: sets.into_partition(),
        })
    }
}

impl AdjacencyLayout for Biadjacency {
    type Adjacency<T: Real, I: Index> = Oriented<T, I>;

    fn sum<T: Real, J: PrimInt, I: Index>(
        self,
        v: FactorWidth2<'_, T, J>,
        seen: impl FnMut(&Column<T, I>),
        pair: impl FnMut(usize, usize, f64),
    ) -> Result<Partitioned<Oriented<T, I>, I>, ReduceError> {
        let c = cells::cross(v, self.n_left, seen, pair)?;
        let partition = Partition::of_bipartite(&c);
        let onto = self.onto;
        let a = Oriented { c, onto };
        Ok(Partitioned { a, partition })
    }
}

/// A factor-width-2 `A`, assembled: general, or bipartite as `C`.
pub(in crate::sgdd) struct Assembled<T, C> {
    adjacency: C,
    diagonal: Vec<T>,
}

/// `A = L(C) + diag(σ)`, dominant by shape: no diagonal is stored, each pivot is `Σ_j |c_ij| + σ_i`.
pub(in crate::sgdd) struct Dominant<'a, T, C> {
    adjacency: C,
    surplus: Option<&'a [T]>,
}

/// An incidence `V`'s summed cells beside `σ`, and their components.
type IncidenceCells<'a, T, I, S> =
    Partitioned<Dominant<'a, T, <S as AdjacencyLayout>::Adjacency<T, I>>, I>;

/// A valued `V`'s summed cells beside `A`'s diagonal, and their components.
type ValueCells<T, I, S> = Partitioned<Assembled<T, <S as AdjacencyLayout>::Adjacency<T, I>>, I>;

/// Each pivot `Σ_j |c_ij| + σ_i`, screened in the assembly's own sums; `A`'s diagonal never stored.
pub(in crate::sgdd) fn assemble_incidence<'a, T: Real, J: PrimInt, I: Index, S: AdjacencyLayout>(
    cells: S,
    v: FactorWidth2<'a, T, J>,
) -> Result<IncidenceCells<'a, T, I, S>, ReduceError> {
    let mut abs = vec![0.0_f64; v.n()];
    let a = cells.sum(
        v,
        |_| {},
        |i, j, c: f64| {
            (abs[i], abs[j]) = (abs[i] + c.abs(), abs[j] + c.abs());
        },
    )?;
    let surplus = v.surplus();
    let s = |i: usize| surplus.map_or(0.0, |s| widen(s[i]));
    if let Some(vertex) = (0..abs.len()).find(|&i| !diagonal_in_range(narrow::<T>(abs[i] + s(i)))) {
        return Err(ReduceError::OutOfRange { vertex });
    }
    let Partitioned { a, partition } = a;
    let a = Dominant {
        adjacency: a,
        surplus,
    };
    Ok(Partitioned { a, partition })
}

/// `A`'s cells, its diagonal `Σ_k w_k v_ik²` and its components, summed in one column pass.
pub(in crate::sgdd) fn assemble_values<T: Real, J: PrimInt, I: Index, S: AdjacencyLayout>(
    cells: S,
    v: FactorWidth2<'_, T, J>,
) -> Result<ValueCells<T, I, S>, ReduceError> {
    let mut sums = vec![0.0; v.n()];
    let a = cells.sum::<T, J, I>(v, |column| add_squares(&mut sums, column), |_, _, _| {})?;
    assembled(a, sums)
}

/// Adds `w v²` at both ends, so a row of only underflowed terms fails the range screen.
#[inline(always)]
fn add_squares<T: Real, I: Index>(sums: &mut [f64], column: &Column<T, I>) {
    // Two straight-line ends: a loop over them costs signed_F 7–10% and bip_sparse 18–20%.
    sums[column.a.index()] += column.square(widen(column.v_a));
    sums[column.b.index()] += column.square(widen(column.v_b));
}

/// The diagonal is each sum as `T` stores it; a nonzero one cast to zero underflowed, so it is out of range.
fn assembled<T: Real, I, C>(
    Partitioned { a, partition }: Partitioned<C, I>,
    sums: Vec<f64>,
) -> Result<Partitioned<Assembled<T, C>, I>, ReduceError> {
    let diagonal = (sums.into_iter().enumerate())
        .map(|(i, sum)| match narrow::<T>(sum) {
            d if sum != 0.0 && d == T::zero() => Err(ReduceError::OutOfRange { vertex: i }),
            d => Ok(d),
        })
        .collect::<Result<_, _>>()?;
    let a = Assembled {
        adjacency: a,
        diagonal,
    };
    Ok(Partitioned { a, partition })
}

impl<T: Real, C> LonePivot<T> for Assembled<T, C> {
    fn lone(&self, v: usize) -> T {
        self.diagonal[v]
    }
}

impl<T: Real, C> LonePivot<T> for Dominant<'_, T, C> {
    fn lone(&self, v: usize) -> T {
        self.surplus.map_or(T::zero(), |s| s[v])
    }
}

/// The upper triangle, cancelled cells dropped.
impl<T: Real, I: Index> PrincipalBlocks<T, I> for UpperAdjacency<T, I> {
    type Block = (Box<[I]>, OffDiagonal<T, I>);

    fn whole(self, vertices: &[I]) -> Self::Block {
        (vertices.into(), OffDiagonal::General(self))
    }

    fn component(&self, vertices: &[I], position: &[I]) -> Self::Block {
        let mut ptrs = Vec::with_capacity(vertices.len() + 1);
        ptrs.push(0);
        let entries = vertices.iter().map(|&v| self.upper_len(v.index())).sum();
        let (mut indices, mut values) = (Vec::with_capacity(entries), Vec::with_capacity(entries));
        for &v in vertices {
            for (j, &x) in self.upper(v.index()) {
                indices.push(position[j]);
                values.push(x);
            }
            ptrs.push(indices.len());
        }
        let rows = Rows::new(ptrs, indices, values, vertices.len());
        (
            vertices.into(),
            OffDiagonal::General(UpperAdjacency::new(rows)),
        )
    }
}

impl<T: Real, I: Index, C> PrincipalBlocks<T, I> for Dominant<'_, T, C>
where
    C: PrincipalBlocks<T, I, Block = (Box<[I]>, OffDiagonal<T, I>)>,
{
    type Block = (Box<[I]>, OffDiagonal<T, I>, Grounding<T>);

    fn whole(self, vertices: &[I]) -> Self::Block {
        let (vertices, off_diagonal) = self.adjacency.whole(vertices);
        let grounding = surplus_at(self.surplus, &vertices);
        (vertices, off_diagonal, grounding)
    }

    fn component(&self, vertices: &[I], position: &[I]) -> Self::Block {
        let (vertices, off_diagonal) = self.adjacency.component(vertices, position);
        let grounding = surplus_at(self.surplus, &vertices);
        (vertices, off_diagonal, grounding)
    }
}

fn surplus_at<T: Real, I: Index>(surplus: Option<&[T]>, vertices: &[I]) -> Grounding<T> {
    surplus.map_or_else(Grounding::ungrounded, |s| Grounding::of(at(s, vertices)))
}

/// `values` in `vertices`' order.
fn at<T: Copy, I: Index>(values: &[T], vertices: &[I]) -> Vec<T> {
    vertices.iter().map(|&v| values[v.index()]).collect()
}

/// The biadjacency `C`, cancelled cells dropped: rows are `A`'s left vertices.
impl<T: Real, I: Index> PrincipalBlocks<T, I> for Oriented<T, I> {
    type Block = (Box<[I]>, OffDiagonal<T, I>);

    fn whole(self, vertices: &[I]) -> Self::Block {
        let (lefts, rights) = vertices.split_at(self.c.n_rows());
        sides(lefts, rights, self.c, self.onto)
    }

    fn component(&self, vertices: &[I], position: &[I]) -> Self::Block {
        let c = &self.c;
        let n_left = c.n_rows();
        let (lefts, rights) = vertices.split_at(vertices.partition_point(|&v| v.index() < n_left));
        let mut ptrs = Vec::with_capacity(lefts.len() + 1);
        ptrs.push(0);
        let entries = lefts.iter().map(|&i| c.row_len(i.index())).sum();
        let (mut indices, mut values) = (Vec::with_capacity(entries), Vec::with_capacity(entries));
        for &i in lefts {
            for (j, &x) in c.row(i.index()) {
                indices.push(position[n_left + j] - I::from_index(lefts.len()));
                values.push(x);
            }
            ptrs.push(indices.len());
        }
        sides(
            lefts,
            rights,
            Rows::new(ptrs, indices, values, rights.len()),
            self.onto,
        )
    }
}

impl<T: Real, I: Index, C> PrincipalBlocks<T, I> for Assembled<T, C>
where
    C: PrincipalBlocks<T, I, Block = (Box<[I]>, OffDiagonal<T, I>)>,
{
    type Block = Cells<T, I>;

    fn whole(self, vertices: &[I]) -> Self::Block {
        let (vertices, off_diagonal) = self.adjacency.whole(vertices);
        let diagonal = at(&self.diagonal, &vertices);
        (vertices, off_diagonal, diagonal)
    }

    fn component(&self, vertices: &[I], position: &[I]) -> Self::Block {
        let (vertices, off_diagonal) = self.adjacency.component(vertices, position);
        let diagonal = at(&self.diagonal, &vertices);
        (vertices, off_diagonal, diagonal)
    }
}

/// The side opposite `onto`'s eliminated first, `C` transposed when it is the right one.
fn sides<T: Real, I: Index>(
    lefts: &[I],
    rights: &[I],
    c: Rows<T, I>,
    onto: Onto,
) -> (Box<[I]>, OffDiagonal<T, I>) {
    let left_first = match onto {
        Onto::Smaller => lefts.len() >= rights.len(),
        Onto::Given(kept) => kept == Side::Right,
    };
    match left_first {
        true => ([lefts, rights].concat().into(), OffDiagonal::Bipartite(c)),
        false => {
            let c = c.transpose();
            ([rights, lefts].concat().into(), OffDiagonal::Bipartite(c))
        }
    }
}
