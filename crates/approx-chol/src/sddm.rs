mod csr;

use crate::Error;
use num_traits::Float;

/// An admission threshold, measured to keep uniformly scaled solves at unit-scale quality (#163).
fn floor<T: Float>() -> T {
    T::min_positive_value() / T::epsilon()
}

/// The ground's diagonal; one definition, so the check and ingestion sum it identically.
fn total<T: Float>(surplus: &[T]) -> T {
    surplus.iter().fold(T::zero(), |sum, &s| sum + s)
}

/// Each vertex's weighted degree, summed edge by edge in row order, so checks and ingestion agree.
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

    /// Each vertex's diagonal, then the ground's when there is surplus.
    fn plus(self, surplus: Option<&[T]>) -> Vec<T> {
        let mut diagonal = self.0;
        if let Some(surplus) = surplus {
            for (d, &s) in diagonal.iter_mut().zip(surplus) {
                *d = *d + s;
            }
            diagonal.push(total(surplus));
        }
        diagonal
    }
}

/// Builds a [`Laplacian`] row by row from edges the caller has checked.
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

    /// Judges each row's sums in order, `verdict` turning its off-diagonal sum into surplus, so errors keep row order.
    fn finish(
        self,
        mut surplus: Vec<T>,
        mut verdict: impl FnMut(usize, T) -> Result<T, Error>,
    ) -> Result<Sddm<T>, Error> {
        for (row, (s, &degree)) in surplus.iter_mut().zip(&self.degrees.0).enumerate() {
            if !degree.is_finite() {
                return Err(Error::NonFiniteRow { row });
            }
            *s = verdict(row, *s)?;
            let diagonal = degree + *s;
            if !diagonal.is_finite() {
                return Err(Error::NonFiniteRow { row });
            }
            if diagonal > T::zero() && diagonal < floor() {
                return Err(Error::MagnitudeTooSmall { entry: (row, row) });
            }
        }
        let ground = total(&surplus);
        if !ground.is_finite() {
            return Err(Error::SurplusOverflow);
        }
        if ground > T::zero() && ground < floor() {
            return Err(Error::SurplusTooSmall);
        }
        let laplacian = Laplacian {
            row_ptrs: self.row_ptrs,
            neighbors: self.neighbors,
            weights: self.weights,
        };
        // Every surplus is zero or positive, so a positive total means some vertex is grounded.
        Ok(if ground > T::zero() {
            Sddm::Grounded(Grounded { laplacian, surplus })
        } else {
            Sddm::Laplacian(laplacian)
        })
    }
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

    /// Row `i`'s neighbors above the diagonal and their weights.
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

impl<T: Float> Sddm<T> {
    /// Each vertex's diagonal as its checks summed it, so every entry is finite; the ground's last.
    pub(crate) fn diagonal(&self, visit: impl FnMut(usize, u32)) -> Vec<T> {
        let surplus = match self {
            Self::Laplacian(_) => None,
            Self::Grounded(grounded) => Some(grounded.surplus()),
        };
        Degrees::of(self.laplacian(), visit).plus(surplus)
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
