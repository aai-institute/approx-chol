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
        // Every surplus is zero or positive, so a positive total means some vertex is grounded.
        if ground == T::zero() {
            return Ok(Self::Laplacian(laplacian));
        }
        // A grounded component's ground slot is named by a `u32` after its vertices.
        let n = laplacian.n() + 1;
        if u32::try_from(n).is_err() {
            return Err(Error::InvalidCsr(
                CsrError::MatrixDimensionExceedsIndexType { n },
            ));
        }
        Ok(Self::Grounded(Grounded { laplacian, surplus }))
    }

    /// Each row's upper entries, then its surplus; a diagonal arrives as summands, in [`Summed`]'s order.
    #[inline]
    pub(crate) fn entries(
        &self,
        rows: impl Iterator<Item = usize> + Clone,
        mut entry: impl FnMut(usize, usize, T),
    ) {
        let laplacian = self.laplacian();
        for row in rows.clone() {
            let (neighbors, weights) = laplacian.row(row);
            for (&col, &weight) in neighbors.iter().zip(weights) {
                let col = col as usize;
                entry(row, col, -weight);
                entry(row, row, weight);
                entry(col, col, weight);
            }
        }
        if let Self::Grounded(grounded) = self {
            for row in rows {
                entry(row, row, grounded.surplus[row]);
            }
        }
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
