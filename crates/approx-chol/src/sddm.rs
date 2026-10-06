mod csr;

use crate::{CsrRef, GroundedError, LaplacianError};
use num_traits::Float;

/// An admission threshold, measured to keep uniformly scaled solves at unit-scale quality (#163).
fn floor<T: Float>() -> T {
    T::min_positive_value() / T::epsilon()
}

/// Each vertex's weighted degree, summed edge by edge in row order: checks and consumers share it.
struct Degrees<T>(Vec<T>);

impl<T: Float> Degrees<T> {
    /// Room for the ground's diagonal, which [`plus`](Self::plus) appends.
    fn zeros(n: usize) -> Self {
        let mut degrees = Vec::with_capacity(n + 1);
        degrees.resize(n, T::zero());
        Self(degrees)
    }

    #[inline]
    fn add(&mut self, row: usize, col: usize, weight: T) {
        self.0[row] = self.0[row] + weight;
        self.0[col] = self.0[col] + weight;
    }

    /// `visit` sees each edge on the way, so a caller walking them anyway needs no pass of its own.
    fn of(laplacian: &Laplacian<T>, mut visit: impl FnMut(usize, u32)) -> Self {
        let mut degrees = Self::zeros(laplacian.n());
        for row in 0..laplacian.n() {
            let (neighbors, weights) = laplacian.row(row);
            for (&col, &weight) in neighbors.iter().zip(weights) {
                degrees.add(row, col as usize, weight);
                visit(row, col);
            }
        }
        degrees
    }

    fn first_non_finite(&self) -> Option<usize> {
        self.0.iter().position(|degree| !degree.is_finite())
    }

    /// Each vertex's diagonal, then the ground's when there is surplus: its total.
    fn plus(self, surplus: Option<&[T]>) -> Vec<T> {
        let mut diagonal = self.0;
        let Some(surplus) = surplus else {
            return diagonal;
        };
        let total = surplus.iter().fold(T::zero(), |sum, &s| sum + s);
        for (d, &s) in diagonal.iter_mut().zip(surplus) {
            *d = *d + s;
        }
        diagonal.push(total);
        diagonal
    }
}

/// Strictly upper and ascending fold into one bound: a column must exceed the row's last.
#[inline]
fn accepts<T: Float>(last: usize, col: usize, weight: T) -> bool {
    col > last && weight >= floor() && weight.is_finite()
}

/// Names what [`accepts`] refused; `last` is the row itself until the row holds an edge.
#[cold]
fn rejection<T: Float>(row: usize, last: usize, col: usize, weight: T) -> LaplacianError {
    if col <= row {
        LaplacianError::NotStrictlyUpper { edge: (row, col) }
    } else if col <= last {
        LaplacianError::UnsortedNeighbors { row }
    } else if weight > T::zero() && weight.is_finite() {
        LaplacianError::WeightTooSmall { edge: (row, col) }
    } else {
        LaplacianError::InvalidWeight { edge: (row, col) }
    }
}

/// A vertex whose weighted degree is not finite.
struct DegreeOverflow {
    vertex: usize,
}

/// Which diagonal attaching surplus left out of range; the ground's is the surplus total.
enum DiagonalFault {
    Overflow { vertex: usize },
    BelowFloor { vertex: usize },
    GroundOverflow,
}

/// Builds a [`Laplacian`] row by row, summing degrees as edges arrive, which callers have checked.
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

    /// The caller's rows are canonical and its weights already proven in range.
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

    fn finish(self) -> Result<Checked<T>, DegreeOverflow> {
        let laplacian = Laplacian {
            row_ptrs: self.row_ptrs,
            neighbors: self.neighbors,
            weights: self.weights,
        };
        Checked::new(laplacian, self.degrees)
    }
}

/// A [`Laplacian`] with the degrees it was checked with, so attaching surplus sums no edge twice.
struct Checked<T> {
    laplacian: Laplacian<T>,
    degrees: Degrees<T>,
}

impl<T: Float> Checked<T> {
    /// The one place a [`Laplacian`] comes into being, so none skips the degree check.
    fn new(laplacian: Laplacian<T>, degrees: Degrees<T>) -> Result<Self, DegreeOverflow> {
        match degrees.first_non_finite() {
            Some(vertex) => Err(DegreeOverflow { vertex }),
            None => Ok(Self { laplacian, degrees }),
        }
    }

    /// Checks arrays already shaped as CSR in place, so adopting them copies nothing.
    fn adopt(
        row_ptrs: Vec<u32>,
        neighbors: Vec<u32>,
        weights: Vec<T>,
    ) -> Result<Self, LaplacianError> {
        let mut degrees = Degrees::zeros(row_ptrs.len() - 1);
        for (row, bounds) in row_ptrs.windows(2).enumerate() {
            let mut last = row;
            for position in bounds[0] as usize..bounds[1] as usize {
                let (col, weight) = (neighbors[position] as usize, weights[position]);
                if !accepts(last, col, weight) {
                    return Err(rejection(row, last, col, weight));
                }
                last = col;
                degrees.add(row, col, weight);
            }
        }
        let laplacian = Laplacian {
            row_ptrs,
            neighbors,
            weights,
        };
        Self::new(laplacian, degrees)
            .map_err(|DegreeOverflow { vertex }| LaplacianError::DegreeOverflow { vertex })
    }

    /// Re-sums the degrees of a Laplacian whose own were dropped after it was checked.
    fn of(laplacian: Laplacian<T>) -> Self {
        let degrees = Degrees::of(&laplacian, |_, _| {});
        Self { laplacian, degrees }
    }

    /// `surplus` is one non-negative, zero-or-normal entry per vertex; the one place the variant is chosen.
    fn with_surplus(self, surplus: Vec<T>) -> Result<Sddm<T>, DiagonalFault> {
        let diagonal = self.degrees.plus(Some(&surplus));
        let (&total, vertices) = diagonal
            .split_last()
            .expect("the ground's diagonal is last");
        if !total.is_finite() {
            return Err(DiagonalFault::GroundOverflow);
        }
        for (vertex, &d) in vertices.iter().enumerate() {
            if !d.is_finite() {
                return Err(DiagonalFault::Overflow { vertex });
            }
            if d > T::zero() && d < floor() {
                return Err(DiagonalFault::BelowFloor { vertex });
            }
        }
        // Every surplus is zero or positive, so a positive total means some vertex is grounded.
        Ok(if total > T::zero() {
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

/// `L(G)` stored as `G`'s strict upper adjacency with positive weights; the diagonal is implied.
#[derive(Debug, Clone, PartialEq)]
pub struct Laplacian<T = f64> {
    row_ptrs: Vec<u32>,
    neighbors: Vec<u32>,
    weights: Vec<T>,
}

impl<T: Float> Laplacian<T> {
    /// Errors with [`LaplacianError`] naming the first entry or vertex that breaks the invariant.
    pub fn new(
        row_ptrs: Vec<u32>,
        neighbors: Vec<u32>,
        weights: Vec<T>,
    ) -> Result<Self, LaplacianError> {
        // Empty arrays reach `validated`, which names them.
        let n = row_ptrs.len().saturating_sub(1);
        let dimension = u32::try_from(n)
            .ok()
            .filter(|&n| n < u32::MAX)
            .ok_or(LaplacianError::TooManyVertices { n })?;
        CsrRef::validated(&row_ptrs, &neighbors, &weights, dimension)
            .map_err(LaplacianError::Structure)?;
        Checked::adopt(row_ptrs, neighbors, weights).map(|checked| checked.laplacian)
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

/// A symmetric diagonally dominant matrix with non-positive off-diagonals: Laplacian plus surplus.
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
    /// Each vertex's diagonal as its checks summed it, so every entry is finite; the ground's last.
    pub(crate) fn diagonal(&self, visit: impl FnMut(usize, u32)) -> Vec<T> {
        let surplus = match self {
            Self::Laplacian(_) => None,
            Self::Grounded(grounded) => Some(grounded.surplus()),
        };
        Degrees::of(self.laplacian(), visit).plus(surplus)
    }

    /// `L + diag(surplus)`, bare when all zero; errors as [`Grounded::new`] save `NoSurplus`.
    pub fn with_surplus(laplacian: Laplacian<T>, surplus: Vec<T>) -> Result<Self, GroundedError> {
        if surplus.len() != laplacian.n() {
            return Err(GroundedError::LengthMismatch {
                expected: laplacian.n(),
                got: surplus.len(),
            });
        }
        // Conversion's noise floor only ever emits normal surplus, so typed input admits no more.
        if let Some(vertex) = surplus
            .iter()
            .position(|&s| !(s == T::zero() || (s.is_normal() && s > T::zero())))
        {
            return Err(GroundedError::InvalidSurplus { vertex });
        }
        Checked::of(laplacian)
            .with_surplus(surplus)
            .map_err(|fault| match fault {
                DiagonalFault::Overflow { vertex } => GroundedError::DiagonalOverflow { vertex },
                DiagonalFault::BelowFloor { vertex } => GroundedError::DiagonalTooSmall { vertex },
                DiagonalFault::GroundOverflow => GroundedError::SurplusOverflow,
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

/// `L(G) + diag(surplus)` with surplus somewhere; a component without any still floats.
#[derive(Debug, Clone, PartialEq)]
pub struct Grounded<T = f64> {
    laplacian: Laplacian<T>,
    surplus: Vec<T>,
}

impl<T: Float> Grounded<T> {
    /// Errors with [`GroundedError`], including [`GroundedError::NoSurplus`] for all-zero surplus.
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

    /// Diagonal excess over edge weights; from CSR, only what survives the summation-noise floor.
    pub fn surplus(&self) -> &[T] {
        &self.surplus
    }
}
