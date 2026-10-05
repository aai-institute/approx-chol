mod csr;

use crate::{CsrError, CsrRef, GroundedError, LaplacianError};
use num_traits::Float;

/// Each vertex's weighted degree, summed edge by edge in row order.
struct Degrees<T>(Vec<T>);

impl<T: Float> Degrees<T> {
    fn zeros(n: usize) -> Self {
        Self(vec![T::zero(); n])
    }

    #[inline]
    fn add(&mut self, row: usize, col: usize, weight: T) {
        self.0[row] = self.0[row] + weight;
        self.0[col] = self.0[col] + weight;
    }
}

/// Strictly upper and ascending fold into one bound: a column must exceed the row's last.
#[inline]
fn accepts<T: Float>(last: usize, col: usize, weight: T) -> bool {
    col > last && weight > T::zero() && weight.is_finite()
}

/// Names what [`accepts`] refused; `last` is the row itself until the row holds an edge.
#[cold]
fn rejection(row: usize, last: usize, col: usize) -> LaplacianError {
    if col <= row {
        LaplacianError::NotStrictlyUpper { edge: (row, col) }
    } else if col <= last {
        LaplacianError::UnsortedNeighbors { row }
    } else {
        LaplacianError::InvalidWeight { edge: (row, col) }
    }
}

/// A vertex whose weighted degree is not finite.
struct DegreeOverflow {
    vertex: usize,
}

/// Which sum attaching surplus left non-finite.
enum SumOverflow {
    Diagonal { vertex: usize },
    Total,
}

/// The only way to a [`Laplacian`]: every degree is summed as an edge arrives. Edges are
/// checked by whoever holds the argument for them, so appending stays inside this module.
struct UpperRows<T> {
    row_ptrs: Vec<u32>,
    neighbors: Vec<u32>,
    weights: Vec<T>,
    degrees: Degrees<T>,
    row: usize,
}

impl<T: Float> UpperRows<T> {
    fn with_capacity(n: usize, nnz: usize) -> Self {
        let mut row_ptrs = Vec::with_capacity(n + 1);
        row_ptrs.push(0);
        Self {
            row_ptrs,
            neighbors: Vec::with_capacity(nnz),
            weights: Vec::with_capacity(nnz),
            degrees: Degrees::zeros(n),
            row: 0,
        }
    }

    /// The caller's rows are canonical and its weights already proven positive and finite.
    #[inline]
    fn push(&mut self, col: u32, weight: T) {
        self.neighbors.push(col);
        self.weights.push(weight);
        self.degrees.add(self.row, col as usize, weight);
    }

    fn end_row(&mut self) {
        self.row_ptrs.push(self.neighbors.len() as u32);
        self.row += 1;
    }

    /// Checks arrays already shaped as CSR in place, so adopting them copies nothing.
    fn adopt(
        row_ptrs: Vec<u32>,
        neighbors: Vec<u32>,
        weights: Vec<T>,
    ) -> Result<Checked<T>, LaplacianError> {
        let n = row_ptrs.len() - 1;
        let mut degrees = Degrees::zeros(n);
        for (row, bounds) in row_ptrs.windows(2).enumerate() {
            let mut last = row;
            for position in bounds[0] as usize..bounds[1] as usize {
                let (col, weight) = (neighbors[position] as usize, weights[position]);
                if !accepts(last, col, weight) {
                    return Err(rejection(row, last, col));
                }
                last = col;
                degrees.add(row, col, weight);
            }
        }
        Self {
            row_ptrs,
            neighbors,
            weights,
            degrees,
            row: n,
        }
        .finish()
        .map_err(|DegreeOverflow { vertex }| LaplacianError::DegreeOverflow { vertex })
    }

    fn finish(self) -> Result<Checked<T>, DegreeOverflow> {
        if let Some(vertex) = self.degrees.0.iter().position(|degree| !degree.is_finite()) {
            return Err(DegreeOverflow { vertex });
        }
        Ok(Checked {
            laplacian: Laplacian {
                row_ptrs: self.row_ptrs,
                neighbors: self.neighbors,
                weights: self.weights,
            },
            degrees: self.degrees,
        })
    }
}

/// A [`Laplacian`] with the degrees it was checked with, so attaching surplus sums no edge twice.
struct Checked<T> {
    laplacian: Laplacian<T>,
    degrees: Degrees<T>,
}

impl<T: Float> Checked<T> {
    /// Re-sums the degrees of a Laplacian whose own were dropped after it was checked.
    fn of(laplacian: Laplacian<T>) -> Self {
        let mut degrees = Degrees::zeros(laplacian.n());
        for row in 0..laplacian.n() {
            let (neighbors, weights) = laplacian.row(row);
            for (&col, &weight) in neighbors.iter().zip(weights) {
                degrees.add(row, col as usize, weight);
            }
        }
        Self { laplacian, degrees }
    }

    fn into_laplacian(self) -> Laplacian<T> {
        self.laplacian
    }

    /// `surplus` is one finite, non-negative entry per vertex; the one place the variant is chosen.
    fn with_surplus(self, surplus: Vec<T>) -> Result<Sddm<T>, SumOverflow> {
        let total = surplus.iter().fold(T::zero(), |sum, &s| sum + s);
        if !total.is_finite() {
            return Err(SumOverflow::Total);
        }
        if let Some(vertex) = self
            .degrees
            .0
            .iter()
            .zip(&surplus)
            .position(|(&degree, &s)| !(degree + s).is_finite())
        {
            return Err(SumOverflow::Diagonal { vertex });
        }
        Ok(if surplus.iter().any(|&s| s > T::zero()) {
            Grounded {
                laplacian: self.laplacian,
                surplus,
            }
            .into()
        } else {
            self.laplacian.into()
        })
    }
}

/// A weighted graph's Laplacian `L(G)`, stored as `G`'s strict upper adjacency: row `i`
/// lists its neighbors `j > i` in ascending order, each with a finite weight `w > 0`.
/// Below `u32::MAX` vertices, so a ground vertex still has an index, and every weighted
/// degree is finite.
///
/// The matrix entry at `(i, j)` is `-w`; the diagonal is never stored.
#[derive(Debug, Clone, PartialEq)]
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
        UpperRows::adopt(row_ptrs, neighbors, weights).map(Checked::into_laplacian)
    }
}

impl<T> Laplacian<T> {
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
#[derive(Debug, Clone, PartialEq)]
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
        Checked::of(laplacian)
            .with_surplus(surplus)
            .map_err(|overflow| match overflow {
                SumOverflow::Diagonal { vertex } => GroundedError::DiagonalOverflow { vertex },
                SumOverflow::Total => GroundedError::SurplusOverflow,
            })
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
#[derive(Debug, Clone, PartialEq)]
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
