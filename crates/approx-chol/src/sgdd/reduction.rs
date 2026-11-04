//! Each block reduced on its own; folded Grams share one column pass.

mod gram;
mod graph;

use num_traits::PrimInt;
use rayon::prelude::*;

use super::block::{Congruent, ReduceError, Reduced};
use super::congruence::{Congruence, Selection};
use super::decomposition::{
    of_csr, of_upper, AdjacencyLayout, Biadjacency, Lone, Partitioned, RowSums,
    StrictUpperTriangle, Summands, SymmetricBlock,
};
use super::scaling::{self, ScaleVector, ScalingPolicy};
use super::{diagonal_in_range, Complement, Reduction};
use crate::input::{BipartiteFactorWidth2, Canonical, Gram, Input, Repr, Side, Stored, Values};
use crate::storage::{narrow, widen, Adjacency, Balance, Grounding, Index, Real, Sum2, WddCsr};
use crate::Error;

/// The reduced blocks, each off-diagonal in the shape layout `S` stores.
type ShapedBy<T, I, S> = Summands<Reduced<T, I, <S as AdjacencyLayout>::Shape<T, I>>, I, T>;

impl<T: Real, I: Index> Reduction<T, I> {
    /// The view's irreducible blocks, each reduced; `policy` scales those not already dominant.
    pub(crate) fn reduce<J: PrimInt + Sync>(
        input: Input<'_, T, J>,
        policy: ScalingPolicy,
    ) -> Result<Self, Error> {
        Ok(match input.repr() {
            // Symmetry is checked in the walk that unions the pattern; upper storage has no mirror.
            Repr::Csr(csr, Stored::Whole) => of_csr(Canonical::of(csr)?)?.reduce(&policy)?,
            Repr::Csr(csr, Stored::UpperTriangle) => {
                of_upper(Canonical::of(csr)?)?.reduce(&policy)?
            }
            Repr::Gram(v) => Self::of_gram(v, &policy)?,
        })
    }

    fn of_gram<J: PrimInt>(v: Gram<'_, T, J>, policy: &ScalingPolicy) -> Result<Self, ReduceError> {
        Ok(match Biadjacency::of(&v) {
            Some(layout) => from_gram(layout, v, policy)?.into_reduction(),
            None => from_gram(StrictUpperTriangle, v, policy)?.into_reduction(),
        })
    }
}

/// An incidence `V` is dominant by shape; values are scaled under `policy`.
fn from_gram<T: Real, J: PrimInt, I: Index, S: AdjacencyLayout>(
    cells: S,
    v: Gram<'_, T, J>,
    policy: &ScalingPolicy,
) -> Result<ShapedBy<T, I, S>, ReduceError> {
    match v.values() {
        Values::Incidence { surplus } => graph::from_incidence(cells, v, surplus),
        Values::Valued { .. } => gram::from_values(cells, v, policy),
    }
}

impl<T: Real, I: Index> Complement<T, I> {
    /// As [`Reduction::reduce`], each block's side opposite `kept` eliminated first.
    pub(crate) fn reduce_onto<J: PrimInt>(
        input: BipartiteFactorWidth2<'_, T, J>,
        kept: Side,
        policy: ScalingPolicy,
    ) -> Result<Self, ReduceError> {
        let range = input.vertices(kept);
        let layout = Biadjacency::onto(&input, kept);
        let Summands { blocks, lone, dead } = from_gram(layout, input.gram(), &policy)?;
        let local = |v: usize| I::from_index(v - range.start);
        let on_kept = |v: I| range.contains(&v.index()).then(|| local(v.index()));
        Ok(Self {
            blocks: (blocks.into_par_iter())
                .map(|block| block.onto_kept(local))
                .collect(),
            lone: (lone.into_iter())
                .filter_map(|lone| {
                    let vertex = on_kept(lone.vertex)?;
                    Some(Lone { vertex, ..lone })
                })
                .collect(),
            dead: dead.into_iter().filter_map(on_kept).collect(),
        })
    }
}

impl<T: Real, J: PrimInt + Sync, I: Index> Partitioned<Canonical<'_, T, J>, I> {
    /// Scales and signs each block's `A_c`, closing its surplus from the same rows.
    fn reduce(self, policy: &ScalingPolicy) -> Result<Reduction<T, I>, ReduceError> {
        self.extract(|a| {
            let relax = |a: &SymmetricBlock<T, I, WddCsr<T, I>>| {
                scaling::relax(&a.adjacency.off_diagonal, &a.rows.diagonal, policy)
            };
            let block = match a.reduce(policy, relax)? {
                Signed::Folded(folded) => {
                    let singularity = folded.singularity();
                    folded.settle(singularity, &RowSurplus)?
                }
                Signed::Frustrated(block) => block,
            };
            Ok(block)
        })
        .map(|components| components.into_summands().into_reduction())
    }
}

impl<T: Real, I: Index, P: Adjacency<T, I>> SymmetricBlock<T, I, P> {
    /// `M = F A_c F` with `F = S R`, formed once with its rows, which decide the report; `relax` runs only past the row test, and a frustrated `A_c` takes `S = I` and grounds on its rows at once.
    fn reduce(
        self,
        policy: &ScalingPolicy,
        relax: impl FnOnce(&Self) -> Option<scaling::Relaxation<T>>,
    ) -> Result<Signed<T, I, P>, ReduceError> {
        let RowSums { diagonal, abs } = &self.rows;
        let in_range = |(&d, &s): (&T, &T)| diagonal_in_range(d) && diagonal_in_range(s);
        if let Some(i) = diagonal.iter().zip(abs).position(|row| !in_range(row)) {
            let vertex = self.adjacency.vertices.vertex(i);
            return Err(ReduceError::OutOfRange { vertex });
        }
        if let Some(i) = diagonal.iter().position(|&d| d <= T::zero()) {
            let vertex = self.adjacency.vertices.vertex(i);
            return Err(ReduceError::NonpositiveDiagonal { vertex });
        }
        let vertex = self.adjacency.vertices.vertex(0);
        let relaxation = match diagonal
            .iter()
            .zip(abs)
            .all(|(&d, &s)| policy.dominates(d, s))
        {
            true => None,
            false => Some(relax(&self).ok_or(ReduceError::Unscalable { vertex })?),
        };
        let folding = folding_signs(&self.adjacency.off_diagonal);
        let folded = folding.is_some();
        let signs = folding.unwrap_or(Congruence::Identity);
        let vector = (relaxation.as_ref()).map_or(ScaleVector::Identity, |r| r.vector);
        let (m, rows) = match relaxation {
            None => Congruent::signed(self, signs),
            Some(r) => Congruent::scaled(self, signs, &r.scales)
                .ok_or(ReduceError::Unscalable { vertex })?,
        };
        let deficits = (rows.diagonal.iter().zip(&rows.abs)).map(|(&d, &s)| widen(s) - widen(d));
        let clamped = vector.worst_past_band(deficits, &rows.diagonal, m.off_diagonal());
        match folded {
            true => Ok(Signed::Folded(Folded {
                m,
                rows,
                vector,
                clamped,
            })),
            false => {
                let grounding = surplus(&rows).flushed(&rows.diagonal, scaling::ratio_band::<T>());
                let grounding = grounding.map_err(|i| out_of_range(&m, i))?;
                let report = vector.report(clamped);
                Ok(Signed::Frustrated(m.block(
                    grounding,
                    Balance::Frustrated,
                    report,
                )?))
            }
        }
    }
}

/// Only a folded `M` can still take a surplus from `V`.
enum Signed<T, I, P> {
    Folded(Folded<T, I, P>),
    Frustrated(Reduced<T, I, P>),
}

/// A folded `M` with its rows and the bands they sit in, its surplus still to be decided.
struct Folded<T, I, P> {
    m: Congruent<T, I, P>,
    rows: RowSums<T>,
    vector: ScaleVector,
    /// The worst `(Σ_j|m_ij| − m_ii)/m_ii` clamped past its row's band.
    clamped: f64,
}

/// `M`'s surplus `M1` and what tells a real one from rounding: a CSR `A` has only `M`'s rows, a Gram also `V`.
trait Surplus<T: Real, I: Index> {
    /// Whether residues alone float a scaled block, which then has no certificate to fall back on.
    const RESIDUES_FLOAT_SCALED: bool;
    /// Each row's residue `(M1)_i`.
    fn residues<'a, P: Adjacency<T, I>>(
        &'a self,
        folded: &'a Folded<T, I, P>,
    ) -> impl Iterator<Item = f64> + Clone + 'a;
    /// Whether a row states a ground, which no residue test can float.
    fn states_ground<P: Adjacency<T, I>>(&self, folded: &Folded<T, I, P>) -> bool;
    /// Whether the block's cycles all close within `√ε`; asked last, as it may run a pass over `V`.
    fn certifies(&self, vertices: &Selection<I>) -> bool;
    /// `M`'s grounds once it does not float.
    fn grounds<P: Adjacency<T, I>>(&self, folded: &Folded<T, I, P>) -> Grounding<T>;
}

/// Read from `M`'s rows: `d_i − Σ_j |m_ij|`.
struct RowSurplus;

impl<T: Real, I: Index> Surplus<T, I> for RowSurplus {
    const RESIDUES_FLOAT_SCALED: bool = true;

    fn residues<'a, P: Adjacency<T, I>>(
        &'a self,
        folded: &'a Folded<T, I, P>,
    ) -> impl Iterator<Item = f64> + Clone + 'a {
        let RowSums { diagonal, abs } = &folded.rows;
        (diagonal.iter().zip(abs)).map(|(&d, &s)| widen(d) - widen(s))
    }

    fn states_ground<P: Adjacency<T, I>>(&self, _: &Folded<T, I, P>) -> bool {
        false
    }

    fn certifies(&self, _: &Selection<I>) -> bool {
        false
    }

    fn grounds<P: Adjacency<T, I>>(
        &self,
        Folded { m, rows, .. }: &Folded<T, I, P>,
    ) -> Grounding<T> {
        match surplus(rows) {
            grounding if grounding.is_grounded() => grounding,
            // An uncompensated `f64` row sum can round up to `d_i` from degree 67 (#269).
            _ => {
                let sums = m.off_diagonal().compensated_abs_row_sums();
                let residues = (rows.diagonal.iter().zip(sums))
                    .map(|(&d, s)| narrow((widen(d) - s).max(0.0)))
                    .collect();
                Grounding::of(residues)
            }
        }
    }
}

impl<T: Real, I: Index, P: Adjacency<T, I>> Folded<T, I, P> {
    /// The one float-or-ground rule: in the singular band, no stated ground, and residues or a certificate float `M`.
    fn settle<S: Surplus<T, I>>(
        self,
        singularity: Singularity,
        surplus: &S,
    ) -> Result<Reduced<T, I, P>, ReduceError> {
        let dominant = matches!(self.vector, ScaleVector::Identity);
        let floats = !surplus.states_ground(&self)
            && singularity.within_band::<T>()
            && (((dominant || S::RESIDUES_FLOAT_SCALED) && self.floats(surplus.residues(&self)))
                || surplus.certifies(self.m.vertices()));
        let report = self.vector.report(self.clamped);
        if floats {
            return (self.m).block(Grounding::ungrounded(), Balance::Balanced, report);
        }
        let grounding = surplus.grounds(&self);
        // The first ground, which flushing all of them would lose from a definite `M`.
        let Some(first) = (grounding.surplus()).and_then(|s| s.iter().position(|&s| s > T::zero()))
        else {
            let vertex = self.m.vertices().vertex(0);
            return Err(ReduceError::Unscalable { vertex });
        };
        let Self { m, rows, .. } = self;
        match grounding.flushed(&rows.diagonal, scaling::ratio_band::<T>()) {
            Ok(grounding) if grounding.is_grounded() => {
                m.block(grounding, Balance::Balanced, report)
            }
            Ok(_) => Err(out_of_range(&m, first)),
            Err(i) => Err(out_of_range(&m, i)),
        }
    }

    /// From `M`'s rows, `1ᵀM1 = tr M − Σ|offdiag(M)|`, so `tr M` cannot overflow.
    fn singularity(&self) -> Singularity {
        let epsilon = widen(T::epsilon());
        let diagonal = Sum2::of(&self.rows.diagonal, |_, d| epsilon * widen(d));
        let trace = diagonal.total();
        // `tr M` rounded on its own drops a surplus below its ulp, which the Gram path keeps.
        let ones = (self.m.off_diagonal())
            .scaled_abs_into(diagonal, -epsilon)
            .total();
        Singularity { ones, trace }
    }

    /// Whether no row's surplus outweighs its coupling or passes its band.
    fn floats(&self, surplus: impl Iterator<Item = f64> + Clone) -> bool {
        let RowSums { diagonal, abs } = &self.rows;
        if (surplus.clone().zip(abs)).any(|(s, &abs)| s > widen(abs)) {
            return false;
        }
        let excess = (surplus.zip(diagonal)).map(|(s, &d)| s - self.clamped * widen(d));
        (self.vector).worst_past_band(excess, diagonal, self.m.off_diagonal()) == 0.0
    }
}

/// Row `i` of `m` holds a ground out of range.
fn out_of_range<T: Real, I: Index, P>(m: &Congruent<T, I, P>, i: usize) -> ReduceError {
    ReduceError::OutOfRange {
        vertex: m.vertices().vertex(i),
    }
}

/// Each row's residue, deficits clamped to zero.
fn surplus<T: Real>(RowSums { diagonal, abs }: &RowSums<T>) -> Grounding<T> {
    Grounding::of(
        (diagonal.iter().zip(abs))
            .map(|(&d, &s)| (d - s).max(T::zero()))
            .collect(),
    )
}

/// `ε·1ᵀM1` and `ε·tr M`, both `ε`-scaled so neither overflows: how near `M` is to singular.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Singularity {
    pub(crate) ones: f64,
    pub(crate) trace: f64,
}

impl Singularity {
    /// `1ᵀM1 ≤ SINGULAR_BAND ε tr M` bounds `M`'s least eigenvalue relative to its diagonal (Rayleigh).
    const SINGULAR_BAND: f64 = 5.0;

    /// Whether `1ᵀM1` is within the singular band of `tr M`: needed, not enough, for `M` to float.
    pub(crate) fn within_band<T: Real>(self) -> bool {
        self.ones <= Self::SINGULAR_BAND * widen(T::epsilon()) * self.trace
    }
}

/// The signs `S` folding `A_c` into a Z-matrix, `I` when it already is one; `None` when frustrated.
fn folding_signs<T: Real, I: Index>(off_diagonal: &impl Adjacency<T, I>) -> Option<Congruence<T>> {
    if off_diagonal.nonpositive() {
        return Some(Congruence::Identity);
    }
    if let Some(cross) = off_diagonal.bipartite() {
        if cross.positive() {
            return Some(Congruence::NegateFrom(cross.n_eliminated()));
        }
    }
    let mut forest = ParityForest::<I>::new(off_diagonal.n());
    for (i, row) in off_diagonal.upper_rows() {
        for (j, &a) in row {
            // A positive entry folds only across opposite signs.
            forest.join(i, j, a > T::zero())?;
        }
    }
    Some(forest.signs())
}

/// Union-find whose links carry whether the two ends' signs differ.
struct ParityForest<I> {
    parent: Vec<I>,
    /// Parity to the parent; `false` at a root.
    odd: Vec<bool>,
    size: Vec<I>,
}

impl<I: Index> ParityForest<I> {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).map(I::from_index).collect(),
            odd: vec![false; n],
            size: vec![I::one(); n],
        }
    }

    /// Root and parity to it, halving the path.
    fn find(&mut self, mut x: usize) -> (usize, bool) {
        let mut parity = false;
        while self.parent[x].index() != x {
            let up = self.parent[x].index();
            self.odd[x] ^= self.odd[up];
            self.parent[x] = self.parent[up];
            parity ^= self.odd[x];
            x = self.parent[x].index();
        }
        (x, parity)
    }

    /// Each vertex negated when odd to its root.
    fn signs<T: Real>(mut self) -> Congruence<T> {
        let sign = |negated: bool| if negated { -T::one() } else { T::one() };
        Congruence::Signs(
            (0..self.parent.len())
                .map(|i| sign(self.find(i).1))
                .collect(),
        )
    }

    /// `None` when `i` and `j` are already joined at the other parity.
    fn join(&mut self, i: usize, j: usize, differ: bool) -> Option<()> {
        let ((ri, pi), (rj, pj)) = (self.find(i), self.find(j));
        if ri == rj {
            return (pi ^ pj == differ).then_some(());
        }
        let (big, small) = match self.size[ri] >= self.size[rj] {
            true => (ri, rj),
            false => (rj, ri),
        };
        self.parent[small] = I::from_index(big);
        self.odd[small] = pi ^ pj ^ differ;
        self.size[big] = self.size[big] + self.size[small];
        Some(())
    }
}
