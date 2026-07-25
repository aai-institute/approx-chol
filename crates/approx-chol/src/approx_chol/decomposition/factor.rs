//! The [`Factor`] decomposition and its solve API.

use super::sequence::EliminationSequence;
#[cfg(feature = "serde")]
use super::FactorError;
use crate::{DenseFailure, Error};
use core::fmt;

#[cfg(test)]
mod tests;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "T: serde::Serialize",
        deserialize = "T: serde::de::DeserializeOwned"
    ))
)]
#[derive(Clone, Debug)]
pub(crate) enum SingleFactor<T> {
    Approx {
        n: usize,
        sequence: EliminationSequence<T>,
    },
    Dense {
        n: usize,
        m: usize,
        lower: Vec<T>,
    },
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "T: serde::Serialize",
        deserialize = "T: serde::de::DeserializeOwned"
    ))
)]
#[derive(Clone, Debug)]
pub(crate) struct ComponentFactor<T> {
    pub(crate) vertices: Vec<u32>,
    pub(crate) factor: SingleFactor<T>,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "T: serde::Serialize",
        deserialize = "T: serde::de::DeserializeOwned"
    ))
)]
#[derive(Clone, Debug)]
enum FactorStorage<T> {
    Single(SingleFactor<T>),
    Blocks(Vec<ComponentFactor<T>>),
}

/// Exact or approximate Cholesky decomposition of an SDDM matrix.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(
        bound(
            serialize = "T: serde::Serialize",
            deserialize = "T: serde::de::DeserializeOwned"
        ),
        try_from = "FactorData<T>"
    )
)]
#[derive(Clone, Debug)]
pub struct Factor<T = f64> {
    pub(crate) n: usize,
    pub(crate) original_n: usize,
    storage: FactorStorage<T>,
}

#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
#[serde(bound(deserialize = "T: serde::de::DeserializeOwned"))]
struct FactorData<T> {
    n: usize,
    original_n: usize,
    storage: FactorStorage<T>,
}

#[cfg(feature = "serde")]
impl<T> TryFrom<FactorData<T>> for Factor<T> {
    type Error = FactorError;

    fn try_from(data: FactorData<T>) -> Result<Self, Self::Error> {
        let factor = Self {
            n: data.n,
            original_n: data.original_n,
            storage: data.storage,
        };
        factor.validate_structure()?;
        Ok(factor)
    }
}

#[cfg(feature = "serde")]
impl<T> Factor<T> {
    fn validate_structure(&self) -> Result<(), FactorError> {
        if self.original_n > self.n {
            return Err(FactorError::OriginalDimExceedsInternal {
                original_n: self.original_n,
                n: self.n,
            });
        }
        fn validate_single<T>(factor: &SingleFactor<T>) -> Result<(), FactorError> {
            match factor {
                SingleFactor::Approx { n, sequence } => sequence.validate_for_dim(*n),
                SingleFactor::Dense { n, m, lower } => {
                    if m.checked_mul(*m) != Some(lower.len()) || *m != n.saturating_sub(1) {
                        return Err(FactorError::DenseLengthInvalid {
                            n: *n,
                            len: lower.len(),
                        });
                    }
                    Ok(())
                }
            }
        }
        match &self.storage {
            FactorStorage::Single(factor) => {
                let factor_n = match factor {
                    SingleFactor::Approx { n, .. } | SingleFactor::Dense { n, .. } => *n,
                };
                if factor_n != self.n {
                    return Err(FactorError::SingleDimensionMismatch);
                }
                validate_single(factor)
            }
            FactorStorage::Blocks(components) => {
                let mut seen = vec![false; self.n];
                for component in components {
                    if let Some(window) = component
                        .vertices
                        .windows(2)
                        .find(|window| window[0] >= window[1])
                    {
                        return Err(FactorError::ComponentVertexInvalid {
                            vertex: window[1] as usize,
                        });
                    }
                    let component_n = match &component.factor {
                        SingleFactor::Approx { n, .. } | SingleFactor::Dense { n, .. } => *n,
                    };
                    if component.vertices.len() != component_n {
                        return Err(FactorError::ComponentDimensionMismatch);
                    }
                    for &vertex in &component.vertices {
                        let vertex = vertex as usize;
                        if vertex >= self.n || seen[vertex] {
                            return Err(FactorError::ComponentVertexInvalid { vertex });
                        }
                        seen[vertex] = true;
                    }
                    validate_single(&component.factor)?;
                }
                if let Some(vertex) = seen.iter().position(|&is_seen| !is_seen) {
                    return Err(FactorError::ComponentVertexInvalid { vertex });
                }
                Ok(())
            }
        }
    }
}

/// Errors returned by fallible [`Factor`] solve methods.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolveError {
    /// Right-hand side length exceeds the solvable (original) dimension.
    RhsLengthExceedsFactor {
        /// Provided RHS length.
        rhs_len: usize,
        /// Maximum accepted RHS length.
        factor_dim: usize,
    },
    /// Work buffer is smaller than the internal factor dimension.
    WorkBufferTooSmall {
        /// Provided work length.
        work_len: usize,
        /// Required factor dimension.
        factor_dim: usize,
    },
}

impl fmt::Display for SolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RhsLengthExceedsFactor {
                rhs_len,
                factor_dim,
            } => write!(
                f,
                "rhs length {rhs_len} exceeds original matrix dimension {factor_dim}"
            ),
            Self::WorkBufferTooSmall {
                work_len,
                factor_dim,
            } => write!(
                f,
                "work buffer too small: got {work_len}, need at least {factor_dim}"
            ),
        }
    }
}

impl std::error::Error for SolveError {}

impl<T> SingleFactor<T>
where
    T: num_traits::Float + Send + Sync + 'static,
{
    pub(crate) fn approx(n: usize, sequence: EliminationSequence<T>) -> Self {
        Self::Approx { n, sequence }
    }

    pub(crate) fn dense(
        n: usize,
        mut matrix: Vec<T>,
        pivot_vertices: &[u32],
    ) -> Result<Self, Error> {
        let m = pivot_vertices.len();
        debug_assert_eq!(matrix.len(), m * m);
        for col in 0..m {
            let mut diagonal = matrix[col * m + col];
            for k in 0..col {
                let value = matrix[col * m + k];
                diagonal = diagonal - value * value;
            }
            if !diagonal.is_finite() {
                return Err(Error::DenseFactorizationFailed {
                    vertex: pivot_vertices[col] as usize,
                    failure: DenseFailure::NonFinitePivot,
                });
            }
            if diagonal <= T::zero() {
                return Err(Error::DenseFactorizationFailed {
                    vertex: pivot_vertices[col] as usize,
                    failure: DenseFailure::NonPositivePivot,
                });
            }
            let pivot = diagonal.sqrt();
            if !pivot.is_finite() {
                return Err(Error::DenseFactorizationFailed {
                    vertex: pivot_vertices[col] as usize,
                    failure: DenseFailure::NonFinitePivot,
                });
            }
            let inverse = T::one() / pivot;
            if !inverse.is_finite() {
                return Err(Error::DenseFactorizationFailed {
                    vertex: pivot_vertices[col] as usize,
                    failure: DenseFailure::NonFiniteReciprocal,
                });
            }
            matrix[col * m + col] = pivot;
            for row in col + 1..m {
                let mut value = matrix[row * m + col];
                for k in 0..col {
                    value = value - matrix[row * m + k] * matrix[col * m + k];
                }
                matrix[row * m + col] = value * inverse;
            }
        }
        Ok(Self::Dense {
            n,
            m,
            lower: matrix,
        })
    }

    fn n(&self) -> usize {
        match self {
            Self::Approx { n, .. } | Self::Dense { n, .. } => *n,
        }
    }

    fn n_steps(&self) -> usize {
        match self {
            Self::Approx { sequence, .. } => sequence.n_steps(),
            Self::Dense { m, .. } => *m,
        }
    }

    fn solve_raw(&self, values: &mut [T]) {
        match self {
            Self::Approx { sequence, .. } => {
                for index in 0..sequence.n_steps() {
                    let step = sequence.step(index);
                    step.apply_forward(values, sequence.inv_diagonal[index]);
                }
                for index in (0..sequence.n_steps()).rev() {
                    sequence.step(index).apply_backward(values);
                }
            }
            Self::Dense { m, lower, .. } => {
                let m = *m;
                for row in 0..m {
                    let mut value = values[row];
                    for col in 0..row {
                        value = value - lower[row * m + col] * values[col];
                    }
                    values[row] = value / lower[row * m + row];
                }
                for row in (0..m).rev() {
                    let mut value = values[row];
                    for col in row + 1..m {
                        value = value - lower[col * m + row] * values[col];
                    }
                    values[row] = value / lower[row * m + row];
                }
                values[m..self.n()].fill(T::zero());
            }
        }
    }

    fn solve_recovered(&self, values: &mut [T], original_n: usize) {
        match self {
            Self::Dense { .. } => self.solve_raw(values),
            Self::Approx { .. } if self.n() > original_n => {
                let aux = original_n;
                values[aux] = -values[..aux]
                    .iter()
                    .fold(T::zero(), |sum, &value| sum + value);
                self.solve_raw(values);
                let ground = values[aux];
                for value in &mut values[..aux] {
                    *value = *value - ground;
                }
            }
            Self::Approx { .. } => {
                self.solve_raw(values);
                if original_n > 0 {
                    let count = num_traits::cast::<usize, T>(original_n).unwrap();
                    let mean = values[..original_n]
                        .iter()
                        .fold(T::zero(), |sum, &value| sum + value)
                        / count;
                    for value in &mut values[..original_n] {
                        *value = *value - mean;
                    }
                }
            }
        }
    }
}

impl<T> Factor<T>
where
    T: num_traits::Float + Send + Sync + 'static,
{
    pub(crate) fn single(original_n: usize, factor: SingleFactor<T>) -> Self {
        Self {
            n: factor.n(),
            original_n,
            storage: FactorStorage::Single(factor),
        }
    }

    pub(crate) fn blocks(n: usize, original_n: usize, components: Vec<ComponentFactor<T>>) -> Self {
        Self {
            n,
            original_n,
            storage: FactorStorage::Blocks(components),
        }
    }

    /// Dimension of the original input matrix.
    pub fn original_n(&self) -> usize {
        self.original_n
    }

    /// Internal factor dimension, including a possible ground vertex.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Number of approximate elimination steps or exact dense pivots.
    pub fn n_steps(&self) -> usize {
        match &self.storage {
            FactorStorage::Single(factor) => factor.n_steps(),
            FactorStorage::Blocks(components) => components
                .iter()
                .map(|component| component.factor.n_steps())
                .sum(),
        }
    }

    fn validate(&self, b_len: Option<usize>, work_len: usize) -> Result<(), SolveError> {
        if let Some(rhs_len) = b_len {
            if rhs_len > self.original_n {
                return Err(SolveError::RhsLengthExceedsFactor {
                    rhs_len,
                    factor_dim: self.original_n,
                });
            }
        }
        if work_len < self.n {
            return Err(SolveError::WorkBufferTooSmall {
                work_len,
                factor_dim: self.n,
            });
        }
        Ok(())
    }

    fn solve_kernel(&self, b: &[T], work: &mut [T]) {
        work[..self.n].fill(T::zero());
        work[..b.len()].copy_from_slice(b);
        match &self.storage {
            FactorStorage::Single(factor) => factor.solve_recovered(work, self.original_n),
            FactorStorage::Blocks(components) => {
                Self::solve_components(components, work, |component, local| {
                    let local_original_n = component
                        .vertices
                        .partition_point(|&vertex| (vertex as usize) < self.original_n);
                    component.factor.solve_recovered(local, local_original_n);
                });
            }
        }
    }

    fn solve_components(
        components: &[ComponentFactor<T>],
        values: &mut [T],
        mut solve: impl FnMut(&ComponentFactor<T>, &mut [T]),
    ) {
        let max_n = components
            .iter()
            .map(|component| component.factor.n())
            .max()
            .unwrap_or(0);
        let mut local = vec![T::zero(); max_n];
        for component in components {
            let local = &mut local[..component.factor.n()];
            for (local_index, &global) in component.vertices.iter().enumerate() {
                local[local_index] = values[global as usize];
            }
            solve(component, local);
            for (local_index, &global) in component.vertices.iter().enumerate() {
                values[global as usize] = local[local_index];
            }
        }
    }

    /// Solve `M x = b`, returning a newly allocated solution.
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, SolveError> {
        let mut work = vec![T::zero(); self.n];
        self.solve_into(b, &mut work)?;
        work.truncate(self.original_n);
        Ok(work)
    }

    /// Solve `M x = b` into a caller-provided work buffer.
    pub fn solve_into(&self, b: &[T], work: &mut [T]) -> Result<(), SolveError> {
        self.validate(Some(b.len()), work.len())?;
        self.solve_kernel(b, work);
        Ok(())
    }

    /// Apply the stored factors directly without gauge recovery.
    pub fn solve_in_place(&self, values: &mut [T]) -> Result<(), SolveError> {
        self.validate(None, values.len())?;
        match &self.storage {
            FactorStorage::Single(factor) => factor.solve_raw(values),
            FactorStorage::Blocks(components) => {
                Self::solve_components(components, values, |component, local| {
                    component.factor.solve_raw(local);
                });
            }
        }
        Ok(())
    }
}
