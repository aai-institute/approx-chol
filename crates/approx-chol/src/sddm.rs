mod csr;

use crate::types::Real;
use crate::{CsrError, Error};

/// The ground's diagonal; one definition, so the check and ingestion sum it identically.
fn total<T: Real>(surplus: &[T]) -> T {
    surplus.iter().fold(T::zero(), |sum, &s| sum + s)
}

/// One definition, so the check and ingestion sum each weighted degree identically.
#[inline]
fn add_edge<T: Real>(degrees: &mut [T], row: usize, col: usize, weight: T) {
    degrees[row] = degrees[row] + weight;
    degrees[col] = degrees[col] + weight;
}

/// `L(G)` stored as `G`'s strict upper adjacency with positive weights; the diagonal is implied.
pub(crate) struct Laplacian<T> {
    row_ptrs: Vec<u32>,
    neighbors: Vec<u32>,
    weights: Vec<T>,
}

impl<T> Laplacian<T> {
    pub(crate) fn n(&self) -> usize {
        self.row_ptrs.len() - 1
    }

    #[inline]
    pub(crate) fn row(&self, i: usize) -> (&[u32], &[T]) {
        let (from, to) = (self.row_ptrs[i] as usize, self.row_ptrs[i + 1] as usize);
        (&self.neighbors[from..to], &self.weights[from..to])
    }
}

/// A symmetric diagonally dominant matrix with non-positive off-diagonals: Laplacian plus surplus.
pub(crate) enum Sddm<T> {
    Laplacian(Laplacian<T>),
    /// Grounded wherever surplus is positive; a component without any still floats.
    Grounded(Grounded<T>),
}

impl<T> Sddm<T> {
    pub(crate) fn n(&self) -> usize {
        self.laplacian().n()
    }

    pub(crate) fn laplacian(&self) -> &Laplacian<T> {
        match self {
            Self::Laplacian(laplacian) => laplacian,
            Self::Grounded(grounded) => &grounded.laplacian,
        }
    }
}

impl<T: Real> Sddm<T> {
    /// Grounded exactly where some surplus is positive; an all-zero surplus stays a bare Laplacian.
    fn with_surplus(laplacian: Laplacian<T>, surplus: Vec<T>) -> Result<Self, Error> {
        // The ground's degree, which the approximate arm sums when it eliminates the ground.
        let ground = total(&surplus);
        if !ground.is_finite() {
            return Err(Error::SurplusOverflow);
        }
        // Every surplus is zero or positive, so a positive total means some vertex is grounded.
        if ground == T::zero() {
            return Ok(Self::Laplacian(laplacian));
        }
        if laplacian.n() >= u32::MAX as usize {
            return Err(Error::InvalidCsr(
                CsrError::MatrixDimensionExceedsIndexType {
                    n: laplacian.n().saturating_add(1),
                },
            ));
        }
        Ok(Self::Grounded(Grounded { laplacian, surplus }))
    }

    /// Summed in the checks' order, so every entry is finite; `visit` spares a caller its own edge pass.
    pub(crate) fn diagonal(&self, mut visit: impl FnMut(usize, u32)) -> Vec<T> {
        let laplacian = self.laplacian();
        let n = laplacian.n();
        // Room for the ground's diagonal.
        let mut diagonal = Vec::with_capacity(n + 1);
        diagonal.resize(n, T::zero());
        for row in 0..n {
            let (neighbors, weights) = laplacian.row(row);
            for (&col, &weight) in neighbors.iter().zip(weights) {
                add_edge(&mut diagonal, row, col as usize, weight);
                visit(row, col);
            }
        }
        if let Self::Grounded(grounded) = self {
            for (d, &s) in diagonal.iter_mut().zip(&grounded.surplus) {
                *d = *d + s;
            }
            diagonal.push(total(&grounded.surplus));
        }
        diagonal
    }
}

pub(crate) struct Grounded<T> {
    laplacian: Laplacian<T>,
    surplus: Vec<T>,
}

impl<T> Grounded<T> {
    /// Diagonal excess over edge weights, only what survives the summation-noise floor.
    pub(crate) fn surplus(&self) -> &[T] {
        &self.surplus
    }
}
