//! Each block reduced on its own; folded Grams share one column pass.

mod gram;
mod graph;

use num_traits::PrimInt;

use super::block::{Balance, Grounding, IrreducibleBlock, ReduceError};
use super::congruence::{Congruence, ScaledFactors};
use super::decomposition::{
    of_csr, of_upper, AdjacencyLayout, Biadjacency, Cells, Partitioned, StrictUpperTriangle,
};
use super::scaling::{self, ScaleVector, ScalingPolicy, ScalingReport};
use super::Sgdd;
use crate::input::{BipartiteFactorWidth2, Canonical, FactorWidth2, Input, Laplacian};
use crate::storage::OffDiagonal;
use crate::{diagonal_in_range, widen, Error, Index, Real};

impl<T: Real, I: Index> Sgdd<T, I> {
    /// The view's irreducible blocks, each reduced; `policy` scales those not already dominant.
    pub(crate) fn reduce<J: PrimInt + Sync>(
        input: Input<'_, T, J>,
        policy: ScalingPolicy,
    ) -> Result<Self, Error> {
        Ok(match input {
            // Symmetry is checked in the walk that unions the pattern; upper storage has no mirror.
            Input::Csr(csr) => of_csr(Canonical::of(csr)?)?.reduce(&policy)?,
            Input::UpperCsr(csr) => of_upper(Canonical::of(csr.0)?)?.reduce(&policy)?,
            Input::Laplacian(Laplacian(v))
            | Input::FactorWidth2(v)
            | Input::BipartiteFactorWidth2(BipartiteFactorWidth2(v)) => match Biadjacency::of(&v) {
                Some(layout) => Self::from_gram(layout, v, &policy)?,
                None => Self::from_gram(StrictUpperTriangle, v, &policy)?,
            },
        })
    }

    /// An incidence `V` is dominant by shape; values are scaled under `policy`.
    fn from_gram<J: PrimInt, S: AdjacencyLayout>(
        cells: S,
        v: FactorWidth2<'_, T, J>,
        policy: &ScalingPolicy,
    ) -> Result<Self, ReduceError> {
        match v.is_incidence() {
            true => Self::from_incidence(cells, v),
            false => Self::from_values(cells, v, policy),
        }
    }
}

impl<T: Real, J: PrimInt + Sync, I: Index> Partitioned<Canonical<'_, T, J>, I> {
    /// Scales and signs each block's `A_c`, closing its surplus from the same rows.
    fn reduce(self, policy: &ScalingPolicy) -> Result<Sgdd<T, I>, ReduceError> {
        self.extract(|cells| {
            let a = SymmetricBlock::new(cells);
            let relax = |a: &SymmetricBlock<T, I>| {
                scaling::relax(&a.off_diagonal, &a.rows.diagonal, policy)
            };
            let block = match a.reduce(policy, relax)? {
                Signed::Folded(m, rows) => {
                    let grounding = rows.folded_surplus(&m.off_diagonal);
                    m.block(Balance::Balanced(grounding))?
                }
                Signed::Frustrated(block) => block,
            };
            Ok(block)
        })
        .map(Sgdd::new)
    }
}

/// One block `A_c` as the input gave it, with its rows' `Σ_j |a_ij|` and entry counts.
struct SymmetricBlock<T, I> {
    /// In `A`'s numbering, in local order.
    vertices: Box<[I]>,
    off_diagonal: OffDiagonal<T, I>,
    rows: RowSums<T>,
}

impl<T: Real, I: Index> SymmetricBlock<T, I> {
    fn new((vertices, off_diagonal, diagonal): Cells<T, I>) -> Self {
        assert_eq!(vertices.len(), off_diagonal.n(), "one vertex per row");
        assert_eq!(diagonal.len(), off_diagonal.n(), "one diagonal per row");
        Self {
            vertices,
            rows: RowSums {
                diagonal,
                abs: off_diagonal.abs_row_sums(),
            },
            off_diagonal,
        }
    }

    /// `M = F A_c F` with `F = S R`, formed once with its rows, which decide the report; `relax` runs only past the row test, and a frustrated `A_c` takes `S = I` and grounds on its rows at once.
    fn reduce(
        self,
        policy: &ScalingPolicy,
        relax: impl FnOnce(&Self) -> Option<scaling::Relaxation<T>>,
    ) -> Result<Signed<T, I>, ReduceError> {
        let RowSums { diagonal, abs } = &self.rows;
        let in_range = |(&d, &s): (&T, &T)| diagonal_in_range(d) && diagonal_in_range(s);
        if let Some(i) = diagonal.iter().zip(abs).position(|row| !in_range(row)) {
            let vertex = self.vertices[i].index();
            return Err(ReduceError::OutOfRange { vertex });
        }
        if let Some(i) = diagonal.iter().position(|&d| d <= T::zero()) {
            let vertex = self.vertices[i].index();
            return Err(ReduceError::NonpositiveDiagonal { vertex });
        }
        let vertex = self.vertices[0].index();
        let relaxation = match diagonal
            .iter()
            .zip(abs)
            .all(|(&d, &s)| policy.dominates(d, s))
        {
            true => None,
            false => Some(relax(&self).ok_or(ReduceError::Unscalable { vertex })?),
        };
        let folding = folding_signs(&self.off_diagonal);
        let folded = folding.is_some();
        let signs = folding.unwrap_or(Congruence::Identity);
        let (m, rows, vector) = match relaxation {
            // Signs keep every magnitude, so `A_c`'s rows are `M`'s.
            None => {
                let m = Congruent::new(self.vertices, self.off_diagonal, signs);
                (m, self.rows, ScaleVector::Identity)
            }
            Some((scales, vector)) => {
                let factors: Box<[T]> = (scales.iter().enumerate())
                    .map(|(i, &scale)| signs.factor(i) * scale)
                    .collect();
                let (off_diagonal, abs) = (self.off_diagonal.scaled_abs_row_sums(&factors))
                    .ok_or(ReduceError::Unscalable { vertex })?;
                let diagonal = (self.rows.diagonal.iter().zip(&scales[..]))
                    .map(|(&a, &f)| f * a * f)
                    .collect();
                let m = Congruent {
                    vertices: self.vertices,
                    congruence: Congruence::Scaled(ScaledFactors::new(factors)),
                    off_diagonal,
                    report: None,
                };
                (m, RowSums { diagonal, abs }, vector)
            }
        };
        let report = vector.report(&rows.diagonal, &rows.abs, &m.off_diagonal);
        let m = Congruent { report, ..m };
        match folded {
            true => Ok(Signed::Folded(m, rows)),
            false => Ok(Signed::Frustrated(
                m.block(Balance::Frustrated(rows.surplus()))?,
            )),
        }
    }
}

/// Only a folded `M` can still take a surplus from `V`.
enum Signed<T, I> {
    Folded(Congruent<T, I>, RowSums<T>),
    Frustrated(IrreducibleBlock<T, I>),
}

/// `M = F · A_c · F` over its vertices, its surplus still to be decided.
struct Congruent<T, I> {
    vertices: Box<[I]>,
    congruence: Congruence<T>,
    off_diagonal: OffDiagonal<T, I>,
    report: Option<ScalingReport>,
}

impl<T: Real, I: Index> Congruent<T, I> {
    /// `S A_c S`, unscaled.
    fn new(vertices: Box<[I]>, off_diagonal: OffDiagonal<T, I>, signs: Congruence<T>) -> Self {
        let off_diagonal = match signs.factors(off_diagonal.n()) {
            Some(s) => off_diagonal.signed(&s),
            None => off_diagonal,
        };
        Self {
            vertices,
            congruence: signs,
            off_diagonal,
            report: None,
        }
    }

    fn block(self, balance: Balance<T>) -> Result<IrreducibleBlock<T, I>, ReduceError> {
        let Self {
            vertices,
            congruence,
            off_diagonal,
            report,
        } = self;
        IrreducibleBlock::new(vertices, congruence, off_diagonal, balance, report)
    }
}

/// `M`'s diagonal and `Σ_j |m_ij|`: the surplus when no shape or `V` gives it.
struct RowSums<T> {
    diagonal: Vec<T>,
    abs: Vec<T>,
}

impl<T: Real> RowSums<T> {
    /// A folded `M`'s row sums are `M1`, so `1ᵀM1 = tr M − Σ|offdiag(M)|`, both compensated.
    fn folded_surplus<I: Index>(self, off_diagonal: &OffDiagonal<T, I>) -> Grounding<T> {
        let trace = crate::compensated_sum(self.diagonal.iter().map(|&d| widen(d)));
        let ones = trace - off_diagonal.abs_total();
        match within_band::<T>(ones, trace) {
            true => Grounding::ungrounded(),
            false => self.surplus(),
        }
    }

    /// Each row's residue, deficits clamped to zero.
    fn surplus(self) -> Grounding<T> {
        let Self { mut diagonal, abs } = self;
        (diagonal.iter_mut().zip(&abs)).for_each(|(d, &s)| *d = (*d - s).max(T::zero()));
        Grounding::of(diagonal)
    }
}

/// `1ᵀM1 ≤ SINGULAR_BAND ε tr M` bounds `M`'s least eigenvalue relative to its diagonal (Rayleigh), so `M` floats.
const SINGULAR_BAND: f64 = 5.0;

/// Whether `ones = 1ᵀM1` is within the band of `trace = tr M`, so `M` is singular to working precision.
pub(crate) fn within_band<T: Real>(ones: f64, trace: f64) -> bool {
    ones <= SINGULAR_BAND * widen(T::epsilon()) * trace
}

/// The signs `S` folding `A_c` into a Z-matrix, `I` when it already is one; `None` when frustrated.
fn folding_signs<T: Real, I: Index>(off_diagonal: &OffDiagonal<T, I>) -> Option<Congruence<T>> {
    if off_diagonal.nonpositive() {
        return Some(Congruence::Identity);
    }
    if let OffDiagonal::Bipartite(cross) = off_diagonal {
        if cross.positive() {
            return Some(Congruence::NegateFrom(cross.n_rows()));
        }
    }
    // Each stored entry once: a bipartite block searches `C` alone, as `Cᵀ` costs a transpose.
    let (rows, offset) = off_diagonal.stored();
    let mut forest = ParityForest::<I>::new(off_diagonal.n());
    for i in 0..rows.n_rows() {
        for (j, &a) in rows.row(i) {
            // A positive entry folds only across opposite signs.
            forest.join(i, offset + j, a > T::zero())?;
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
