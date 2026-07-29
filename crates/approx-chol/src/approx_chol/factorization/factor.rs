use super::anchor::Anchor;
use super::block::Block;
use super::permutation::Permutation;
#[cfg(any(feature = "serde", test))]
use super::FactorError;
use core::fmt;

#[cfg(test)]
mod tests;

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
/// Exact or approximate Cholesky decomposition of an SDDM matrix.
pub struct Factor<T = f64> {
    original_n: usize,
    permutation: Option<Permutation>,
    blocks: Vec<Block<T>>,
    #[cfg_attr(feature = "serde", serde(default))]
    fallbacks: Vec<Fallback>,
}

#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
#[serde(bound(deserialize = "T: serde::de::DeserializeOwned"))]
struct FactorData<T> {
    original_n: usize,
    permutation: Option<Permutation>,
    blocks: Vec<Block<T>>,
    #[serde(default)]
    fallbacks: Vec<Fallback>,
}

#[cfg(feature = "serde")]
impl<T: num_traits::Float> TryFrom<FactorData<T>> for Factor<T> {
    type Error = FactorError;

    fn try_from(data: FactorData<T>) -> Result<Self, Self::Error> {
        let factor = Self {
            original_n: data.original_n,
            permutation: data.permutation,
            blocks: data.blocks,
            fallbacks: data.fallbacks,
        };
        factor.validate_structure()?;
        Ok(factor)
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Why a block [`Backend::ExactBelow`](crate::Backend::ExactBelow) claimed was factored approximately.
pub enum Fallback {
    /// Dense elimination reached a pivot it could not use.
    InvalidPivot(crate::UnusablePivot),
    /// The dense copy would not fit in memory; never fatal, whatever [`ExactFailure`](crate::ExactFailure) says.
    WillNotFit {
        /// Variables the block solves for, so the copy is `dim * dim` scalars.
        dim: usize,
    },
}

impl fmt::Display for Fallback {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPivot(pivot) => write!(f, "{pivot}"),
            Self::WillNotFit { dim } => write!(f, "{dim} variables do not fit in memory"),
        }
    }
}

#[cfg(any(feature = "serde", test))]
impl<T: num_traits::Float> Factor<T> {
    fn validate_structure(&self) -> Result<(), FactorError> {
        let n = self.n();
        // A Ground anchor overwrites its block's last entry with `-sum`, so a second
        // one silently solves a different system.
        let grounded = self.ground_blocks();
        if grounded > 1 {
            return Err(FactorError::MultipleGroundBlocks { grounded });
        }
        let mut covered = 0usize;
        for block in &self.blocks {
            block.cholesky.validate_for_dim(block.dim)?;
            covered = covered
                .checked_add(block.dim.total())
                .ok_or(FactorError::BlockDimsDoNotCoverFactor { covered, n })?;
        }
        if covered != n {
            return Err(FactorError::BlockDimsDoNotCoverFactor { covered, n });
        }
        if let Some(permutation) = &self.permutation {
            permutation.validate_for_dim(n)?;
        }
        Ok(())
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
/// Errors returned by fallible [`Factor`] solve methods.
pub enum SolveError {
    /// Right-hand side longer than the solvable (original) dimension.
    RhsLengthExceedsFactor {
        /// Provided RHS length.
        rhs_len: usize,
        /// Maximum accepted RHS length.
        factor_dim: usize,
    },
    /// Work buffer smaller than the factor dimension.
    WorkBufferTooSmall {
        /// Provided work length.
        work_len: usize,
        /// Factor dimension (`Factor::n()`).
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

impl<T> Factor<T> {
    /// Blocks routed to exact Cholesky and factored approximately anyway.
    pub fn fallbacks(&self) -> &[Fallback] {
        &self.fallbacks
    }

    /// Dimension of the original input matrix.
    pub fn original_n(&self) -> usize {
        self.original_n
    }

    /// Internal factor dimension, including a possible ground vertex.
    #[inline]
    pub fn n(&self) -> usize {
        self.original_n + self.ground_blocks()
    }

    /// Nothing else records that the ground vertex exists, and at most one block can
    /// hold it.
    fn ground_blocks(&self) -> usize {
        self.blocks
            .iter()
            .filter(|block| block.anchor == Anchor::Ground)
            .count()
    }
}

impl<T> Factor<T>
where
    T: num_traits::Float + Send + Sync + 'static,
{
    pub(crate) fn from_blocks(
        original_n: usize,
        permutation: Option<Permutation>,
        blocks: Vec<Block<T>>,
        fallbacks: Vec<Fallback>,
    ) -> Self {
        let factor = Self {
            original_n,
            permutation,
            blocks,
            fallbacks,
        };
        #[cfg(any(feature = "serde", test))]
        debug_assert_eq!(factor.validate_structure(), Ok(()));
        factor
    }

    pub(crate) fn empty(original_n: usize) -> Self {
        Self::from_blocks(original_n, None, Vec::new(), Vec::new())
    }

    /// Total elimination steps across all blocks.
    pub fn n_steps(&self) -> usize {
        self.blocks.iter().map(Block::n_steps).sum()
    }

    #[inline]
    fn validate_work(&self, work: &[T]) -> Result<(), SolveError> {
        if work.len() < self.n() {
            return Err(SolveError::WorkBufferTooSmall {
                work_len: work.len(),
                factor_dim: self.n(),
            });
        }
        Ok(())
    }

    fn solve_kernel(&self, b: &[T], work: &mut [T]) {
        work[..b.len()].copy_from_slice(b);
        work[b.len()..self.n()].fill(T::zero());
        self.for_each_block(work, Block::solve_canonical);
    }

    fn for_each_block(&self, values: &mut [T], mut solve: impl FnMut(&Block<T>, &mut [T])) {
        let values = &mut values[..self.n()];
        let Some(permutation) = &self.permutation else {
            Self::solve_blocks(&self.blocks, values, &mut solve);
            return;
        };
        let mut scratch = vec![T::zero(); self.n()];
        permutation.gather_into(values, &mut scratch);
        Self::solve_blocks(&self.blocks, &mut scratch, &mut solve);
        permutation.scatter_from(&scratch, values);
    }

    #[inline(always)]
    fn solve_blocks(
        blocks: &[Block<T>],
        values: &mut [T],
        solve: &mut impl FnMut(&Block<T>, &mut [T]),
    ) {
        let mut start = 0usize;
        for block in blocks {
            let end = start + block.dim.total();
            solve(block, &mut values[start..end]);
            start = end;
        }
    }

    /// Solve `M x = b`, returning the zero-mean least-squares solution for singular `M`.
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, SolveError> {
        let mut work = vec![T::zero(); self.n()];
        self.solve_into(b, &mut work)?;
        work.truncate(self.original_n);
        Ok(work)
    }

    /// Solve `M x = b` into a caller-provided work buffer.
    pub fn solve_into(&self, b: &[T], work: &mut [T]) -> Result<(), SolveError> {
        if b.len() > self.original_n {
            return Err(SolveError::RhsLengthExceedsFactor {
                rhs_len: b.len(),
                factor_dim: self.original_n,
            });
        }
        self.validate_work(work)?;
        self.solve_kernel(b, work);
        Ok(())
    }

    /// Solve in place, leaving one variable per block pinned to zero.
    pub fn solve_in_place(&self, values: &mut [T]) -> Result<(), SolveError> {
        self.validate_work(values)?;
        self.for_each_block(values, Block::solve_anchored);
        Ok(())
    }
}
