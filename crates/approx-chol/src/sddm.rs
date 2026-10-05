use crate::{CsrError, Error};
use num_traits::Float;

/// A weighted graph's Laplacian `L(G)`, stored as `G`'s strict upper adjacency: row `i`
/// lists its neighbors `j > i` in ascending order, each with a finite weight `w > 0`.
/// Below `u32::MAX` vertices, so a ground vertex still has an index.
///
/// The matrix entry at `(i, j)` is `-w`; the diagonal is never stored.
#[derive(Debug, Clone)]
pub struct Laplacian<T = f64> {
    row_ptrs: Vec<u32>,
    neighbors: Vec<u32>,
    weights: Vec<T>,
}

impl<T: Float> Laplacian<T> {
    /// Validate and take ownership of a strict upper adjacency.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidCsr`] for malformed `row_ptrs`, `u32::MAX` or more vertices, or an
    /// out-of-range neighbor,
    /// [`Error::NotStrictlyUpper`], [`Error::UnsortedNeighbors`], [`Error::NonFiniteValue`]
    /// or [`Error::NonPositiveWeight`] for the entry that breaks the invariant.
    pub fn new(row_ptrs: Vec<u32>, neighbors: Vec<u32>, weights: Vec<T>) -> Result<Self, Error> {
        let Some(&end) = row_ptrs.last() else {
            return Err(Error::InvalidCsr(CsrError::RowPtrsLenMismatch {
                expected: 1,
                got: 0,
            }));
        };
        if row_ptrs[0] != 0 {
            return Err(Error::InvalidCsr(CsrError::RowPtrsMustStartAtZero {
                got: row_ptrs[0] as usize,
            }));
        }
        if neighbors.len() != weights.len() {
            return Err(Error::InvalidCsr(CsrError::ColIndicesValuesLenMismatch {
                col_indices_len: neighbors.len(),
                values_len: weights.len(),
            }));
        }
        if end as usize != neighbors.len() {
            return Err(Error::InvalidCsr(CsrError::RowPtrsEndMismatchNnz {
                row_ptr_end: end as usize,
                nnz: neighbors.len(),
            }));
        }
        let n = row_ptrs.len() - 1;
        if n >= u32::MAX as usize {
            return Err(Error::InvalidCsr(
                CsrError::MatrixDimensionExceedsIndexType { n },
            ));
        }
        for (row, bounds) in row_ptrs.windows(2).enumerate() {
            if bounds[0] > bounds[1] {
                return Err(Error::InvalidCsr(CsrError::RowPtrsNotNonDecreasing {
                    row,
                    prev: bounds[0] as usize,
                    next: bounds[1] as usize,
                }));
            }
            let (from, to) = (bounds[0] as usize, bounds[1] as usize);
            let mut previous = row;
            for position in from..to {
                let col = neighbors[position] as usize;
                if col >= n {
                    return Err(Error::InvalidCsr(CsrError::ColumnIndexOutOfBounds {
                        position,
                        col,
                        n,
                    }));
                }
                if col <= row {
                    return Err(Error::NotStrictlyUpper { edge: (row, col) });
                }
                if position > from && col <= previous {
                    return Err(Error::UnsortedNeighbors { row });
                }
                previous = col;
                let weight = weights[position];
                if !weight.is_finite() {
                    return Err(Error::NonFiniteValue { position });
                }
                if weight <= T::zero() {
                    return Err(Error::NonPositiveWeight { edge: (row, col) });
                }
            }
        }
        Ok(Self {
            row_ptrs,
            neighbors,
            weights,
        })
    }
}

impl<T> Laplacian<T> {
    /// Built by a caller that already holds the invariants [`new`](Self::new) checks.
    pub(crate) fn trusted(row_ptrs: Vec<u32>, neighbors: Vec<u32>, weights: Vec<T>) -> Self {
        Self {
            row_ptrs,
            neighbors,
            weights,
        }
    }

    /// Number of vertices.
    pub fn n(&self) -> usize {
        self.row_ptrs.len() - 1
    }

    /// Row pointers into [`neighbors`](Self::neighbors), length `n + 1`.
    pub fn row_ptrs(&self) -> &[u32] {
        &self.row_ptrs
    }

    /// Each row's neighbors above the diagonal, ascending.
    pub fn neighbors(&self) -> &[u32] {
        &self.neighbors
    }

    /// Edge weights, parallel to [`neighbors`](Self::neighbors).
    pub fn weights(&self) -> &[T] {
        &self.weights
    }

    /// Row `i`'s neighbors above the diagonal and their weights.
    #[inline]
    pub(crate) fn row(&self, i: usize) -> (&[u32], &[T]) {
        let (from, to) = (self.row_ptrs[i] as usize, self.row_ptrs[i + 1] as usize);
        (&self.neighbors[from..to], &self.weights[from..to])
    }
}

/// A symmetric diagonally dominant matrix with non-positive off-diagonals: a graph's
/// Laplacian, alone or with surplus on its diagonal.
#[derive(Debug, Clone)]
pub enum Sddm<T = f64> {
    /// Floating in every connected component.
    Laplacian(Laplacian<T>),
    /// Grounded wherever surplus is positive.
    Grounded(Grounded<T>),
}

impl<T> Sddm<T> {
    /// Number of vertices.
    pub fn n(&self) -> usize {
        self.laplacian().n()
    }

    /// The off-diagonal part.
    pub fn laplacian(&self) -> &Laplacian<T> {
        match self {
            Self::Laplacian(laplacian) => laplacian,
            Self::Grounded(grounded) => &grounded.laplacian,
        }
    }
}

impl<T> From<Laplacian<T>> for Sddm<T> {
    fn from(laplacian: Laplacian<T>) -> Self {
        Self::Laplacian(laplacian)
    }
}

impl<T> From<Grounded<T>> for Sddm<T> {
    fn from(grounded: Grounded<T>) -> Self {
        Self::Grounded(grounded)
    }
}

/// `L(G) + diag(surplus)` with surplus somewhere; a connected component without any
/// still floats.
#[derive(Debug, Clone)]
pub struct Grounded<T = f64> {
    laplacian: Laplacian<T>,
    surplus: Vec<T>,
}

impl<T: Float> Grounded<T> {
    /// # Errors
    ///
    /// [`Error::SurplusLengthMismatch`] unless there is one surplus per vertex,
    /// [`Error::InvalidSurplus`] for one that is negative or not finite, and
    /// [`Error::NoSurplus`] when every one is zero, which is a bare [`Laplacian`].
    pub fn new(laplacian: Laplacian<T>, surplus: Vec<T>) -> Result<Self, Error> {
        if surplus.len() != laplacian.n() {
            return Err(Error::SurplusLengthMismatch {
                expected: laplacian.n(),
                got: surplus.len(),
            });
        }
        let mut grounded = false;
        for (vertex, &s) in surplus.iter().enumerate() {
            if !(s.is_finite() && s >= T::zero()) {
                return Err(Error::InvalidSurplus { vertex });
            }
            grounded |= s > T::zero();
        }
        if !grounded {
            return Err(Error::NoSurplus);
        }
        Ok(Self { laplacian, surplus })
    }
}

impl<T> Grounded<T> {
    /// `surplus` is one non-negative entry per vertex, at least one positive.
    pub(crate) fn trusted(laplacian: Laplacian<T>, surplus: Vec<T>) -> Self {
        Self { laplacian, surplus }
    }

    /// Number of vertices.
    pub fn n(&self) -> usize {
        self.laplacian.n()
    }

    /// The off-diagonal part.
    pub fn laplacian(&self) -> &Laplacian<T> {
        &self.laplacian
    }

    /// Each vertex's diagonal excess over its edge weights; on the CSR path, what
    /// survived the summation-noise floor.
    pub fn surplus(&self) -> &[T] {
        &self.surplus
    }
}
