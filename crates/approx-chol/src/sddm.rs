use crate::{CsrError, CsrRef, GroundedError, LaplacianError};
use num_traits::Float;

/// Each vertex's diagonal, summed edge by edge in row order onto a starting value: the
/// order the exact arm sums it in, so a finite one here is finite there.
pub(crate) struct Diagonal<T>(Vec<T>);

impl<T: Float> Diagonal<T> {
    pub(crate) fn starting_at(start: Vec<T>) -> Self {
        Self(start)
    }

    #[inline]
    pub(crate) fn add(&mut self, row: usize, col: usize, weight: T) {
        self.0[row] = self.0[row] + weight;
        self.0[col] = self.0[col] + weight;
    }

    pub(crate) fn first_non_finite(&self) -> Option<usize> {
        self.0.iter().position(|value| !value.is_finite())
    }

    /// For a caller that learns the surplus only after summing the edges.
    pub(crate) fn first_non_finite_with(&self, surplus: &[T]) -> Option<usize> {
        self.0
            .iter()
            .zip(surplus)
            .position(|(&value, &s)| !(value + s).is_finite())
    }
}

fn validate_surplus<T: Float>(
    laplacian: &Laplacian<T>,
    surplus: &[T],
) -> Result<(), GroundedError> {
    if surplus.len() != laplacian.n() {
        return Err(GroundedError::LengthMismatch {
            expected: laplacian.n(),
            got: surplus.len(),
        });
    }
    if let Some(vertex) = surplus
        .iter()
        .position(|&s| !(s.is_finite() && s >= T::zero()))
    {
        return Err(GroundedError::InvalidSurplus { vertex });
    }
    if !surplus_sums_finitely(surplus) {
        return Err(GroundedError::SurplusOverflow);
    }
    if let Some(vertex) = laplacian.diagonal(surplus.to_vec()).first_non_finite() {
        return Err(GroundedError::DiagonalOverflow { vertex });
    }
    Ok(())
}

/// The sum is a ground vertex's degree.
pub(crate) fn surplus_sums_finitely<T: Float>(surplus: &[T]) -> bool {
    surplus
        .iter()
        .fold(T::zero(), |sum, &s| sum + s)
        .is_finite()
}

/// A weighted graph's Laplacian `L(G)`, stored as `G`'s strict upper adjacency: row `i`
/// lists its neighbors `j > i` in ascending order, each with a finite weight `w > 0`.
/// Below `u32::MAX` vertices, so a ground vertex still has an index, and every weighted
/// degree is finite.
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
    /// [`LaplacianError::Structure`] when the arrays are not a CSR matrix, else the
    /// variant naming the first entry or vertex that breaks the invariant.
    pub fn new(
        row_ptrs: Vec<u32>,
        neighbors: Vec<u32>,
        weights: Vec<T>,
    ) -> Result<Self, LaplacianError> {
        let Some(n) = row_ptrs.len().checked_sub(1) else {
            return Err(LaplacianError::Structure(CsrError::RowPtrsLenMismatch {
                expected: 1,
                got: 0,
            }));
        };
        let dimension = u32::try_from(n)
            .ok()
            .filter(|&n| n < u32::MAX)
            .ok_or(LaplacianError::TooManyVertices { n })?;
        CsrRef::new(&row_ptrs, &neighbors, &weights, dimension)
            .map_err(LaplacianError::Structure)?;
        let mut diagonal = Diagonal::starting_at(vec![T::zero(); n]);
        for (row, bounds) in row_ptrs.windows(2).enumerate() {
            let (from, to) = (bounds[0] as usize, bounds[1] as usize);
            let mut previous = row;
            for position in from..to {
                let col = neighbors[position] as usize;
                if col <= row {
                    return Err(LaplacianError::NotStrictlyUpper { edge: (row, col) });
                }
                if position > from && col <= previous {
                    return Err(LaplacianError::UnsortedNeighbors { row });
                }
                previous = col;
                let weight = weights[position];
                if !(weight.is_finite() && weight > T::zero()) {
                    return Err(LaplacianError::InvalidWeight { edge: (row, col) });
                }
                diagonal.add(row, col, weight);
            }
        }
        if let Some(vertex) = diagonal.first_non_finite() {
            return Err(LaplacianError::DegreeOverflow { vertex });
        }
        Ok(Self {
            row_ptrs,
            neighbors,
            weights,
        })
    }
}

impl<T: Float> Laplacian<T> {
    /// `start` plus each vertex's weighted degree.
    fn diagonal(&self, start: Vec<T>) -> Diagonal<T> {
        let mut diagonal = Diagonal::starting_at(start);
        for row in 0..self.n() {
            let (neighbors, weights) = self.row(row);
            for (&col, &weight) in neighbors.iter().zip(weights) {
                diagonal.add(row, col as usize, weight);
            }
        }
        diagonal
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

impl<T: Float> Sddm<T> {
    /// `L + diag(surplus)`, which is the bare Laplacian when every surplus is zero.
    ///
    /// # Errors
    ///
    /// What [`Grounded::new`] reports, except [`GroundedError::NoSurplus`].
    pub fn with_surplus(laplacian: Laplacian<T>, surplus: Vec<T>) -> Result<Self, GroundedError> {
        validate_surplus(&laplacian, &surplus)?;
        Ok(Self::trusted(laplacian, surplus))
    }

    /// The one place the variant is chosen; the caller has checked every sum.
    pub(crate) fn trusted(laplacian: Laplacian<T>, surplus: Vec<T>) -> Self {
        if surplus.iter().any(|&s| s > T::zero()) {
            Grounded { laplacian, surplus }.into()
        } else {
            laplacian.into()
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
/// still floats. Every diagonal entry and the surplus total are finite.
#[derive(Debug, Clone)]
pub struct Grounded<T = f64> {
    laplacian: Laplacian<T>,
    surplus: Vec<T>,
}

impl<T: Float> Grounded<T> {
    /// # Errors
    ///
    /// [`GroundedError::LengthMismatch`] unless there is one surplus per vertex,
    /// [`GroundedError::InvalidSurplus`] for one that is negative or not finite,
    /// [`GroundedError::SurplusOverflow`] or [`GroundedError::DiagonalOverflow`] for a sum
    /// that is not finite, and
    /// [`GroundedError::NoSurplus`] when every one is zero, which is a bare [`Laplacian`].
    pub fn new(laplacian: Laplacian<T>, surplus: Vec<T>) -> Result<Self, GroundedError> {
        match Sddm::with_surplus(laplacian, surplus)? {
            Sddm::Grounded(grounded) => Ok(grounded),
            Sddm::Laplacian(_) => Err(GroundedError::NoSurplus),
        }
    }
}

impl<T> Grounded<T> {
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
