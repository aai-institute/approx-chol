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
pub(crate) enum BlockFactor<T> {
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
pub(crate) struct Block<T> {
    pub(crate) start: u32,
    /// Every block's Laplacian is singular, so one variable must be pinned.
    pub(crate) anchor: u32,
    pub(crate) ground: Option<u32>,
    pub(crate) factor: BlockFactor<T>,
}

impl<T> Block<T> {
    fn ground(&self) -> Option<usize> {
        self.ground.map(|ground| ground as usize)
    }
}

/// Block-contiguous order to input order, held as the cycle decomposition of
/// the forward map so both directions are in-place rotations needing no scratch.
/// Fixed points are omitted; `starts` bounds each cycle's run in `cycles`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub(crate) struct Permutation {
    cycles: Vec<u32>,
    starts: Vec<u32>,
}

impl Permutation {
    /// `forward[i]` is the input vertex at permuted position `i`; `None` if identity.
    pub(crate) fn from_forward(forward: &[u32]) -> Option<Self> {
        if forward.iter().enumerate().all(|(i, &v)| i as u32 == v) {
            return None;
        }
        let mut visited = vec![false; forward.len()];
        let mut cycles = Vec::new();
        let mut starts = vec![0u32];
        for position in 0..forward.len() {
            if visited[position] || forward[position] as usize == position {
                continue;
            }
            let mut current = position;
            while !visited[current] {
                visited[current] = true;
                cycles.push(current as u32);
                current = forward[current] as usize;
            }
            starts.push(cycles.len() as u32);
        }
        Some(Self { cycles, starts })
    }

    fn cycle_slices(&self) -> impl Iterator<Item = &[u32]> {
        self.starts
            .windows(2)
            .map(|bounds| &self.cycles[bounds[0] as usize..bounds[1] as usize])
    }

    /// `values[i] <- values[forward[i]]`
    fn gather<T: Copy>(&self, values: &mut [T]) {
        for cycle in self.cycle_slices() {
            let first = values[cycle[0] as usize];
            for window in cycle.windows(2) {
                values[window[0] as usize] = values[window[1] as usize];
            }
            values[cycle[cycle.len() - 1] as usize] = first;
        }
    }

    /// `values[forward[i]] <- values[i]`
    fn scatter<T: Copy>(&self, values: &mut [T]) {
        for cycle in self.cycle_slices() {
            let last = values[cycle[cycle.len() - 1] as usize];
            for window in cycle.windows(2).rev() {
                values[window[1] as usize] = values[window[0] as usize];
            }
            values[cycle[0] as usize] = last;
        }
    }
}

/// A block whose exact factorization failed, so it was factored approximately.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExactFallback {
    /// Pivot vertex that failed, in input numbering.
    pub vertex: usize,
    /// Why the pivot failed.
    pub failure: DenseFailure,
}

/// Exact or approximate Cholesky decomposition of an SDDM matrix.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(
        bound(
            serialize = "T: serde::Serialize",
            deserialize = "T: serde::de::DeserializeOwned + num_traits::Float"
        ),
        try_from = "FactorData<T>"
    )
)]
#[derive(Clone, Debug)]
pub struct Factor<T = f64> {
    pub(crate) n: usize,
    pub(crate) original_n: usize,
    permutation: Option<Permutation>,
    blocks: Vec<Block<T>>,
    exact_fallbacks: Vec<ExactFallback>,
}

#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
#[serde(bound(deserialize = "T: serde::de::DeserializeOwned"))]
struct FactorData<T> {
    n: usize,
    original_n: usize,
    permutation: Option<Permutation>,
    blocks: Vec<Block<T>>,
    exact_fallbacks: Vec<ExactFallback>,
}

#[cfg(feature = "serde")]
impl<T: num_traits::Float> TryFrom<FactorData<T>> for Factor<T> {
    type Error = FactorError;

    fn try_from(data: FactorData<T>) -> Result<Self, Self::Error> {
        let factor = Self {
            n: data.n,
            original_n: data.original_n,
            permutation: data.permutation,
            blocks: data.blocks,
            exact_fallbacks: data.exact_fallbacks,
        };
        factor.validate_structure()?;
        Ok(factor)
    }
}

#[cfg(feature = "serde")]
impl<T: num_traits::Float> Factor<T> {
    fn validate_structure(&self) -> Result<(), FactorError> {
        if self.original_n > self.n {
            return Err(FactorError::OriginalDimExceedsInternal {
                original_n: self.original_n,
                n: self.n,
            });
        }
        // Tiling [0, n) is what stops a corrupted range aliasing another block.
        let mut expected_start = 0usize;
        for block in &self.blocks {
            let block_n = block.factor.n();
            if block.start as usize != expected_start || block_n == 0 {
                return Err(FactorError::BlockRangeInvalid {
                    start: block.start as usize,
                    n: block_n,
                });
            }
            if block.anchor as usize >= block_n {
                return Err(FactorError::BlockAnchorInvalid {
                    anchor: block.anchor as usize,
                    n: block_n,
                });
            }
            if let Some(ground) = block.ground {
                if ground as usize >= block_n {
                    return Err(FactorError::BlockGroundInvalid {
                        ground: ground as usize,
                        n: block_n,
                    });
                }
            }
            block.factor.validate()?;
            expected_start += block_n;
        }
        if expected_start != self.n {
            return Err(FactorError::BlockRangeInvalid {
                start: expected_start,
                n: self.n,
            });
        }
        if let Some(permutation) = &self.permutation {
            permutation.validate_for_dim(self.n)?;
        }
        Ok(())
    }
}

#[cfg(feature = "serde")]
impl Permutation {
    fn validate_for_dim(&self, n: usize) -> Result<(), FactorError> {
        if self.starts.first() != Some(&0)
            || self.starts.last() != Some(&(self.cycles.len() as u32))
        {
            return Err(FactorError::PermutationInvalid { position: 0 });
        }
        if self.starts.windows(2).any(|bounds| bounds[0] >= bounds[1]) {
            return Err(FactorError::PermutationInvalid { position: 0 });
        }
        // `from_forward` omits fixed points; `gather` would silently ignore one.
        if self.cycle_slices().any(|cycle| cycle.len() < 2) {
            return Err(FactorError::PermutationInvalid { position: 0 });
        }
        let mut seen = vec![false; n];
        for &position in &self.cycles {
            let position = position as usize;
            if position >= n || seen[position] {
                return Err(FactorError::PermutationInvalid { position });
            }
            seen[position] = true;
        }
        Ok(())
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

impl<T> BlockFactor<T> {
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
}

#[cfg(feature = "serde")]
impl<T: num_traits::Float> BlockFactor<T> {
    fn validate(&self) -> Result<(), FactorError> {
        match self {
            Self::Approx { n, sequence } => sequence.validate_for_dim(*n),
            Self::Dense { n, m, lower } => {
                if m.checked_mul(*m) != Some(lower.len()) || *m != n.saturating_sub(1) {
                    return Err(FactorError::DenseLengthInvalid {
                        n: *n,
                        len: lower.len(),
                    });
                }
                // `solve_raw` divides by each diagonal entry twice per row.
                for index in 0..*m {
                    let pivot = lower[index * m + index];
                    if !pivot.is_finite() || pivot <= T::zero() {
                        return Err(FactorError::DensePivotInvalid { index });
                    }
                }
                Ok(())
            }
        }
    }
}

impl<T> BlockFactor<T>
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

    /// Solves the block system, leaving `values[pin]` equal to zero, where `pin`
    /// is the ground vertex if this block has one and the anchor otherwise. Both
    /// backends satisfy this, so the raw result never depends on which ran.
    fn apply_anchored(&self, values: &mut [T], anchor: usize, ground: Option<usize>) {
        // Every block is a singular Laplacian, so it solves only a zero-sum
        // right-hand side; both arms put `values` in that range.
        match ground {
            // The exact embedding of `M x = b` as `L_aug [x; 0] = [b; -sum b]`.
            Some(ground) => {
                let mut sum = T::zero();
                for (index, &value) in values.iter().enumerate() {
                    if index != ground {
                        sum = sum + value;
                    }
                }
                values[ground] = -sum;
            }
            // No ground vertex to absorb the null-space component, so project it
            // out; an inconsistent right-hand side then gives least squares.
            None => Self::project_zero_mean(values),
        }
        match self {
            // The factor is already of the anchor-deleted submatrix, and
            // `solve_raw` zeroes the anchor slot.
            Self::Dense { m, .. } => {
                debug_assert_eq!(ground.unwrap_or(anchor), *m);
                self.solve_raw(values);
            }
            Self::Approx { .. } => {
                self.solve_raw(values);
                let pinned = values[ground.unwrap_or(anchor)];
                for value in values.iter_mut() {
                    *value = *value - pinned;
                }
            }
        }
    }

    fn solve_recovered(&self, values: &mut [T], anchor: usize, ground: Option<usize>) {
        self.apply_anchored(values, anchor, ground);
        // Pinning the ground vertex already gives the SDDM solution; a floating
        // block is determined only up to a constant, fixed here to zero mean.
        if ground.is_none() {
            Self::project_zero_mean(values);
        }
    }

    fn project_zero_mean(values: &mut [T]) {
        if values.is_empty() {
            return;
        }
        let count = num_traits::cast::<usize, T>(values.len()).unwrap();
        let mean = values.iter().fold(T::zero(), |sum, &value| sum + value) / count;
        for value in values.iter_mut() {
            *value = *value - mean;
        }
    }
}

impl<T> Factor<T>
where
    T: num_traits::Float + Send + Sync + 'static,
{
    /// An empty `forward` means the blocks already sit in input order.
    pub(crate) fn from_blocks(
        n: usize,
        original_n: usize,
        forward: &[u32],
        blocks: Vec<Block<T>>,
        exact_fallbacks: Vec<ExactFallback>,
    ) -> Self {
        debug_assert!(forward.is_empty() || forward.len() == n);
        Self {
            n,
            original_n,
            permutation: Permutation::from_forward(forward),
            blocks,
            exact_fallbacks,
        }
    }

    pub(crate) fn empty(original_n: usize) -> Self {
        Self {
            n: 0,
            original_n,
            permutation: None,
            blocks: Vec::new(),
            exact_fallbacks: Vec::new(),
        }
    }

    /// Blocks that [`Backend::ExactBelow`](crate::Backend::ExactBelow) selected for
    /// exact Cholesky but whose factorization failed, so they were factored
    /// approximately instead. Empty for any other backend.
    ///
    /// A non-empty result means the input was not positive definite within the
    /// tolerance ingestion accepted it under; the factor is still usable as a
    /// preconditioner.
    pub fn exact_fallbacks(&self) -> &[ExactFallback] {
        &self.exact_fallbacks
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
        self.blocks.iter().map(|block| block.factor.n_steps()).sum()
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
        self.for_each_block(work, |block, values| {
            block
                .factor
                .solve_recovered(values, block.anchor as usize, block.ground());
        });
    }

    fn for_each_block(&self, values: &mut [T], mut solve: impl FnMut(&Block<T>, &mut [T])) {
        let values = &mut values[..self.n];
        if let Some(permutation) = &self.permutation {
            permutation.gather(values);
        }
        for block in &self.blocks {
            let start = block.start as usize;
            solve(block, &mut values[start..start + block.factor.n()]);
        }
        if let Some(permutation) = &self.permutation {
            permutation.scatter(values);
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

    /// Solve in place, skipping the zero-mean canonicalization that [`Self::solve`]
    /// applies to floating blocks.
    ///
    /// Each block is left anchored: one variable per block is pinned to zero — the
    /// ground vertex where augmentation added one, the un-eliminated vertex
    /// otherwise. For an SDDM input that pins the ground vertex, so the result
    /// already solves `M x = b` and matches [`Self::solve_into`]. For a Laplacian
    /// input the result differs from [`Self::solve_into`] by a constant per block.
    /// Neither depends on [`Backend`](crate::Backend).
    pub fn solve_in_place(&self, values: &mut [T]) -> Result<(), SolveError> {
        self.validate(None, values.len())?;
        self.for_each_block(values, |block, values| {
            block
                .factor
                .apply_anchored(values, block.anchor as usize, block.ground());
        });
        Ok(())
    }
}
