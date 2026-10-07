mod csr;
mod laplacian;

pub(crate) use laplacian::Laplacian;
use laplacian::Summed;

use crate::types::Real;
use crate::{CsrError, Error};

/// An admission threshold, measured to keep uniformly scaled solves at unit-scale quality (#163).
fn floor<T: Real>() -> T {
    T::min_positive_value() / T::epsilon()
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
    /// Owns every sum check, whoever stored the rows; an all-zero surplus stays a bare Laplacian.
    fn with_surplus(summed: Summed<T>, surplus: Vec<T>) -> Result<Self, Error> {
        // A ground edge's weight, so it clears the floor every stored entry clears.
        if let Some(row) = surplus.iter().position(|&s| s > T::zero() && s < floor()) {
            return Err(Error::MagnitudeTooSmall { entry: (row, row) });
        }
        // The ground's degree, which the approximate arm sums when it eliminates the ground.
        let ground = surplus.iter().fold(T::zero(), |sum, &s| sum + s);
        if !ground.is_finite() {
            return Err(Error::SurplusOverflow);
        }
        if let Some(row) = summed.first_non_finite_diagonal(&surplus) {
            return Err(Error::NonFiniteRow { row });
        }
        let laplacian = summed.into_laplacian();
        if ground != T::zero() && laplacian.n() >= u32::MAX as usize {
            return Err(Error::InvalidCsr(
                CsrError::MatrixDimensionExceedsIndexType {
                    n: laplacian.n().saturating_add(1),
                },
            ));
        }
        // Every surplus is zero or positive, so a positive total means some vertex is grounded.
        Ok(if ground == T::zero() {
            Self::Laplacian(laplacian)
        } else {
            Self::Grounded(Grounded { laplacian, surplus })
        })
    }

    /// Summed in [`Summed`]'s order, so every entry is the checked one; `visit` spares a caller its own edge pass.
    pub(crate) fn diagonal(&self, mut visit: impl FnMut(usize, u32)) -> Vec<T> {
        let laplacian = self.laplacian();
        let n = laplacian.n();
        // Room for the ground's diagonal.
        let mut diagonal = Vec::with_capacity(n + 1);
        diagonal.resize(n, T::zero());
        for row in 0..n {
            let (neighbors, weights) = laplacian.row(row);
            for (&col, &weight) in neighbors.iter().zip(weights) {
                diagonal[row] = diagonal[row] + weight;
                diagonal[col as usize] = diagonal[col as usize] + weight;
                visit(row, col);
            }
        }
        if let Self::Grounded(grounded) = self {
            for (d, &s) in diagonal.iter_mut().zip(&grounded.surplus) {
                *d = *d + s;
            }
            diagonal.push(grounded.surplus.iter().fold(T::zero(), |sum, &s| sum + s));
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
