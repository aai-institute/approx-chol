//! Each block reduced on its own; folded Grams share one column pass.

mod csr;
mod gram;
mod graph;

use std::cell::LazyCell;

use super::block::{Balance, Grounding, IrreducibleBlock, ReduceError, Sgdd};
use super::congruence::Congruence;
use super::decomposition::{Cells, Partition};
use super::scaling::{self, Existence, Scaling, ScalingPolicy};
use super::signature;
use crate::storage::OffDiagonal;
use crate::{diagonal_in_range, widen, Real};

fn into_sgdd<T: Real>(
    (partition, live, dead): (Partition, Vec<IrreducibleBlock<T>>, Vec<u32>),
) -> Sgdd<T> {
    Sgdd::new(partition.n(), live, dead)
}

/// One block `A_c` as the input gave it, with its rows' `Σ_j |a_ij|` and entry counts.
struct SymmetricBlock<T> {
    /// In `A`'s numbering, in local order.
    vertices: Box<[u32]>,
    off_diagonal: OffDiagonal<T>,
    diagonal: Vec<T>,
    /// Summed in `f64`: a long `f32` row is not undercounted.
    abs_row_sums: Vec<T>,
}

impl<T: Real> From<Cells<T>> for SymmetricBlock<T> {
    fn from((vertices, off_diagonal, diagonal): Cells<T>) -> Self {
        Self::new(vertices, off_diagonal, diagonal)
    }
}

impl<T: Real> SymmetricBlock<T> {
    /// Panics unless one vertex and one diagonal per row.
    fn new(vertices: Box<[u32]>, off_diagonal: OffDiagonal<T>, diagonal: Vec<T>) -> Self {
        assert_eq!(vertices.len(), off_diagonal.n(), "one vertex per row");
        assert_eq!(diagonal.len(), off_diagonal.n(), "one diagonal per row");
        Self {
            vertices,
            abs_row_sums: off_diagonal.abs_row_sums(),
            off_diagonal,
            diagonal,
        }
    }

    /// Components join only nonzeros, so a vertex with nothing to solve is alone with a zero diagonal.
    fn is_dead(&self) -> bool {
        self.diagonal == [T::zero()]
    }

    /// Rows prove nothing: relaxed past the row test.
    fn scale(&self, policy: &ScalingPolicy) -> Result<Scaling<T>, ReduceError> {
        self.scaling(policy, Existence::Unknown, || {
            scaling::relax(&self.off_diagonal, &self.diagonal, policy)
        })
    }

    /// `V` proves a scaling exists; a zero diagonal beside a cell is its terms underflowing.
    fn scale_gram(
        &self,
        policy: &ScalingPolicy,
        nulls: &LazyCell<Vec<f64>, impl FnOnce() -> Vec<f64>>,
    ) -> Result<Scaling<T>, ReduceError> {
        if let Some(i) = self.diagonal.iter().position(|&d| d == T::zero()) {
            let vertex = self.vertices[i];
            return Err(ReduceError::OutOfRange { vertex });
        }
        self.scaling(policy, Existence::Guaranteed, || {
            let u: Vec<f64> = (self.vertices.iter()).map(|&v| nulls[v as usize]).collect();
            (u.iter().all(|x| x.is_finite()))
                .then(|| scaling::null_scaling(&u, &self.diagonal))
                .flatten()
                .or_else(|| scaling::relax(&self.off_diagonal, &self.diagonal, policy))
        })
    }

    /// Screens the diagonal and row sums; `relax` runs only past the row test.
    fn scaling(
        &self,
        policy: &ScalingPolicy,
        existence: Existence,
        relax: impl FnOnce() -> Option<scaling::Relaxation<T>>,
    ) -> Result<Scaling<T>, ReduceError> {
        if let Some(i) = self.diagonal.iter().position(|&d| !diagonal_in_range(d)) {
            let vertex = self.vertices[i];
            return Err(ReduceError::OutOfRange { vertex });
        }
        if let Some(i) = self
            .abs_row_sums
            .iter()
            .position(|&s| !diagonal_in_range(s))
        {
            let vertex = self.vertices[i];
            return Err(ReduceError::OutOfRange { vertex });
        }
        if let Some(i) = self.diagonal.iter().position(|&d| d <= T::zero()) {
            let vertex = self.vertices[i];
            return Err(ReduceError::NonpositiveDiagonal { vertex });
        }
        let vertex = self.vertices[0];
        let mut rows = self.diagonal.iter().zip(&self.abs_row_sums);
        if rows.all(|(&d, &s)| policy.dominates(d, s)) {
            return Ok(Scaling::AlreadyDominant);
        }
        let relaxation = relax().ok_or(ReduceError::Unscalable { vertex })?;
        Ok(Scaling::Scaled(relaxation, existence))
    }

    /// `M = F A_c F` with `F = S R`; a frustrated `A_c` takes `S = I` and grounds on its rows at once.
    fn sign(self, scaling: Scaling<T>, policy: &ScalingPolicy) -> Result<Signed<T>, ReduceError> {
        match signature::folding_signs(&self.off_diagonal) {
            Some(signs) => {
                let (m, rows) = self.congruent(signs, scaling, policy)?;
                Ok(Signed::Folded(m, rows))
            }
            None => {
                let (m, rows) = self.congruent(Congruence::Identity, scaling, policy)?;
                Ok(Signed::Frustrated(m.frustrated(rows.surplus())?))
            }
        }
    }

    /// `M` formed once with its rows, which decide the report.
    fn congruent(
        self,
        signs: Congruence<T>,
        scaling: Scaling<T>,
        policy: &ScalingPolicy,
    ) -> Result<(Congruent<T>, RowSums<T>), ReduceError> {
        match scaling {
            Scaling::AlreadyDominant => {
                // Signs keep every magnitude, so `A_c`'s rows are `M`'s.
                let rows = RowSums {
                    diagonal: self.diagonal,
                    abs: self.abs_row_sums,
                };
                Ok((
                    Congruent::new(self.vertices, self.off_diagonal, signs),
                    rows,
                ))
            }
            Scaling::Scaled(relaxation, existence) => {
                let vertex = self.vertices[0];
                let factors = signs.times(&relaxation.scales);
                let (off_diagonal, abs) = (self.off_diagonal.scaled_abs_row_sums(&factors))
                    .ok_or(ReduceError::Unscalable { vertex })?;
                let diagonal: Vec<T> = (self.diagonal.iter().zip(&relaxation.scales[..]))
                    .map(|(&a, &f)| f * a * f)
                    .collect();
                let deficit = clamped_deficit(&diagonal, &abs);
                let report = policy.report(&relaxation, existence, deficit);
                let m = Congruent {
                    vertices: self.vertices,
                    congruence: Congruence::Scaled(factors, report),
                    off_diagonal,
                };
                let rows = RowSums { diagonal, abs };
                Ok((m, rows))
            }
        }
    }
}

/// Only a folded `M` can still take a surplus from `V`.
enum Signed<T> {
    Folded(Congruent<T>, RowSums<T>),
    Frustrated(IrreducibleBlock<T>),
}

/// `M = F · A_c · F` over its vertices, its surplus still to be decided.
struct Congruent<T> {
    vertices: Box<[u32]>,
    congruence: Congruence<T>,
    off_diagonal: OffDiagonal<T>,
}

impl<T: Real> Congruent<T> {
    /// `S A_c S`, unscaled.
    fn new(vertices: Box<[u32]>, off_diagonal: OffDiagonal<T>, signs: Congruence<T>) -> Self {
        let off_diagonal = match signs.factors(off_diagonal.n()) {
            Some(s) => off_diagonal.signed(&s),
            None => off_diagonal,
        };
        Self {
            vertices,
            congruence: signs,
            off_diagonal,
        }
    }

    fn frustrated(self, grounding: Grounding<T>) -> Result<IrreducibleBlock<T>, ReduceError> {
        let balance = Balance::Frustrated(grounding);
        IrreducibleBlock::new(self.vertices, self.congruence, self.off_diagonal, balance)
    }

    fn balanced(self, grounding: Grounding<T>) -> Result<IrreducibleBlock<T>, ReduceError> {
        let balance = Balance::Balanced(grounding);
        IrreducibleBlock::new(self.vertices, self.congruence, self.off_diagonal, balance)
    }
}

/// `M`'s diagonal and `Σ_j |m_ij|`: the surplus when no shape or `V` gives it.
struct RowSums<T> {
    diagonal: Vec<T>,
    abs: Vec<T>,
}

impl<T: Real> RowSums<T> {
    /// A folded `M`'s row sums are `M1`, so `1ᵀM1 = tr M − Σ|offdiag(M)|`, both compensated.
    fn folded_surplus(self, off_diagonal: &OffDiagonal<T>) -> Grounding<T> {
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

/// `max_i (Σ_j|m_ij| − m_ii)/m_ii` past the entries' rounding: what clamping adds.
fn clamped_deficit<T: Real>(diagonal: &[T], abs_row_sums: &[T]) -> f64 {
    let band = SINGULAR_BAND * widen(T::epsilon());
    (diagonal.iter().zip(abs_row_sums))
        .map(|(&d, &s)| (widen(s) - widen(d)) / widen(d))
        .filter(|&deficit| deficit > band)
        .fold(0.0, f64::max)
}

/// `1ᵀM1 ≤ SINGULAR_BAND ε tr M` bounds `M`'s least eigenvalue relative to its diagonal (Rayleigh), so `M` floats.
const SINGULAR_BAND: f64 = 5.0;
