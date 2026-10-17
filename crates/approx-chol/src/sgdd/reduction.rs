//! Each block reduced on its own; folded Grams share one column pass.

mod csr;
mod gram;
mod graph;

use std::cell::LazyCell;

use num_traits::PrimInt;

use super::block::{Balance, Grounding, IrreducibleBlock, ReduceError, Sgdd};
use super::congruence::{Congruence, Scales};
use super::decomposition::{Cells, Components};
use super::scaling::{self, ScaleVector, ScalingPolicy, ScalingReport, RATIO_BAND};
use super::signature;
use crate::input::Input;
use crate::storage::OffDiagonal;
use crate::{diagonal_in_range, widen, Error, Index, Real};

impl<T: Real, I: Index> Sgdd<T, I> {
    /// The view's irreducible blocks, each reduced; `policy` scales those not already dominant.
    pub(crate) fn reduce<J: PrimInt + Sync>(
        input: Input<'_, T, J>,
        policy: ScalingPolicy,
    ) -> Result<Self, Error> {
        Ok(match input {
            Input::Csr(csr) => Self::from_csr(csr, policy)?,
            Input::UpperCsr(csr) => Self::from_upper_csr(csr, policy)?,
            Input::Laplacian(laplacian) => Self::try_from(laplacian)?,
            Input::SignedGraph(graph) => Self::try_from(graph)?,
            Input::Gram(v) => Self::from_gram(v, policy)?,
            Input::BipartiteGram(gram) => Self::from_bipartite_gram(gram, policy)?,
        })
    }
}

fn into_sgdd<T: Real, I: Index>(
    (partition, live): Components<IrreducibleBlock<T, I>, I>,
) -> Sgdd<T, I> {
    Sgdd::new(partition.n(), live)
}

/// One block `A_c` as the input gave it, with its rows' `Σ_j |a_ij|` and entry counts.
struct SymmetricBlock<T, I> {
    /// In `A`'s numbering, in local order.
    vertices: Box<[I]>,
    off_diagonal: OffDiagonal<T, I>,
    diagonal: Vec<T>,
    /// Summed in `f64`: a long `f32` row is not undercounted.
    abs_row_sums: Vec<T>,
}

impl<T: Real, I: Index> SymmetricBlock<T, I> {
    /// `None` when dead: components join only nonzeros, so a vertex with nothing to solve is alone with a zero diagonal.
    fn live((vertices, off_diagonal, diagonal): Cells<T, I>) -> Option<Self> {
        assert_eq!(vertices.len(), off_diagonal.n(), "one vertex per row");
        assert_eq!(diagonal.len(), off_diagonal.n(), "one diagonal per row");
        (diagonal != [T::zero()]).then(|| Self {
            vertices,
            abs_row_sums: off_diagonal.abs_row_sums(),
            off_diagonal,
            diagonal,
        })
    }

    /// `V` proves a scaling exists; a zero diagonal beside a cell is its terms underflowing.
    fn reduce_gram(
        self,
        policy: &ScalingPolicy,
        nulls: &LazyCell<Vec<f64>, impl FnOnce() -> Vec<f64>>,
    ) -> Result<Signed<T, I>, ReduceError> {
        if let Some(i) = self.diagonal.iter().position(|&d| d == T::zero()) {
            let vertex = self.vertices[i].index();
            return Err(ReduceError::OutOfRange { vertex });
        }
        self.reduce(policy, |a| {
            let u = (a.vertices.iter()).map(|&v| nulls[v.index()]);
            (u.clone().all(f64::is_finite))
                .then(|| scaling::null_scaling(u, &a.diagonal))
                .flatten()
                .or_else(|| scaling::relax(&a.off_diagonal, &a.diagonal, policy))
        })
    }

    /// `M = F A_c F` with `F = S R`, formed once with its rows, which decide the report; `relax` runs only past the row test, and a frustrated `A_c` takes `S = I` and grounds on its rows at once.
    fn reduce(
        self,
        policy: &ScalingPolicy,
        relax: impl FnOnce(&Self) -> Option<scaling::Relaxation<T>>,
    ) -> Result<Signed<T, I>, ReduceError> {
        if let Some(i) = self.diagonal.iter().position(|&d| !diagonal_in_range(d)) {
            let vertex = self.vertices[i].index();
            return Err(ReduceError::OutOfRange { vertex });
        }
        if let Some(i) = self
            .abs_row_sums
            .iter()
            .position(|&s| !diagonal_in_range(s))
        {
            let vertex = self.vertices[i].index();
            return Err(ReduceError::OutOfRange { vertex });
        }
        if let Some(i) = self.diagonal.iter().position(|&d| d <= T::zero()) {
            let vertex = self.vertices[i].index();
            return Err(ReduceError::NonpositiveDiagonal { vertex });
        }
        let vertex = self.vertices[0].index();
        let mut rows = self.diagonal.iter().zip(&self.abs_row_sums);
        let relaxation = match rows.all(|(&d, &s)| policy.dominates(d, s)) {
            true => None,
            false => Some(relax(&self).ok_or(ReduceError::Unscalable { vertex })?),
        };
        let folding = signature::folding_signs(&self.off_diagonal);
        let folded = folding.is_some();
        let signs = folding.unwrap_or(Congruence::Identity);
        let (m, rows) = match relaxation {
            None => {
                // Signs keep every magnitude, so `A_c`'s rows are `M`'s.
                let (diagonal, abs) = (&self.diagonal, &self.abs_row_sums);
                let deficit =
                    clamped_deficit(diagonal, abs, &self.off_diagonal, ScaleVector::Identity);
                let mut m = Congruent::new(self.vertices, self.off_diagonal, signs);
                m.report = policy.report(ScalingReport::unscaled(), deficit);
                let rows = RowSums {
                    diagonal: self.diagonal,
                    abs: self.abs_row_sums,
                };
                (m, rows)
            }
            Some(relaxation) => {
                let factors = signs.times(&relaxation.scales);
                let (off_diagonal, abs) = (self.off_diagonal.scaled_abs_row_sums(&factors))
                    .ok_or(ReduceError::Unscalable { vertex })?;
                let diagonal: Vec<T> = (self.diagonal.iter().zip(&relaxation.scales[..]))
                    .map(|(&a, &f)| f * a * f)
                    .collect();
                let deficit = clamped_deficit(&diagonal, &abs, &off_diagonal, relaxation.vector);
                let m = Congruent {
                    vertices: self.vertices,
                    congruence: Congruence::Scaled(Scales::new(factors)),
                    off_diagonal,
                    report: policy.report(relaxation.shortfall(), deficit),
                };
                (m, RowSums { diagonal, abs })
            }
        };
        match folded {
            true => Ok(Signed::Folded(m, rows)),
            false => Ok(Signed::Frustrated(m.frustrated(rows.surplus())?)),
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

    fn frustrated(self, grounding: Grounding<T>) -> Result<IrreducibleBlock<T, I>, ReduceError> {
        let balance = Balance::Frustrated(grounding);
        IrreducibleBlock::new(
            self.vertices,
            self.congruence,
            self.off_diagonal,
            balance,
            self.report,
        )
    }

    fn balanced(self, grounding: Grounding<T>) -> Result<IrreducibleBlock<T, I>, ReduceError> {
        let balance = Balance::Balanced(grounding);
        IrreducibleBlock::new(
            self.vertices,
            self.congruence,
            self.off_diagonal,
            balance,
            self.report,
        )
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
        match ones <= SINGULAR_BAND * widen(T::epsilon()) * trace {
            true => Grounding::ungrounded(),
            false => self.surplus(),
        }
    }

    /// Each row's residue, deficits clamped to zero.
    fn surplus(self) -> Grounding<T> {
        let rows = self.diagonal.iter().zip(&self.abs);
        Grounding::of(rows.map(|(&d, &s)| (d - s).max(T::zero())).collect())
    }
}

/// `max_i (Σ_j|m_ij| − m_ii)/m_ii` past row `i`'s band, which its `k_i` entries and the scale vector set.
fn clamped_deficit<T: Real, I: Index>(
    diagonal: &[T],
    abs_row_sums: &[T],
    off_diagonal: &OffDiagonal<T, I>,
    vector: ScaleVector,
) -> f64 {
    let epsilon = widen(T::epsilon());
    let deficits =
        (diagonal.iter().zip(abs_row_sums)).map(|(&d, &s)| (widen(s) - widen(d)) / widen(d));
    // Degrees cost a pass over the entries, so only a row past the floor pays for them.
    if deficits
        .clone()
        .all(|deficit| deficit <= RATIO_BAND * epsilon)
    {
        return 0.0;
    }
    (deficits.zip(off_diagonal.degrees()).zip(diagonal))
        .filter(|&((deficit, k), &d)| deficit > vector.row_band(k, widen(d), epsilon))
        .fold(0.0, |worst, ((deficit, _), _)| worst.max(deficit))
}

/// `1ᵀM1 ≤ SINGULAR_BAND ε tr M` bounds `M`'s least eigenvalue relative to its diagonal (Rayleigh), so `M` floats.
const SINGULAR_BAND: f64 = 5.0;
