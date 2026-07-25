//! The [`Factor`] decomposition and its solve API.

use super::sequence::EliminationSequence;
#[cfg(any(feature = "serde", test))]
use super::FactorError;
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
pub(crate) struct BlockFactor<T> {
    n: usize,
    /// Every block's Laplacian is singular, so one variable must be pinned.
    anchor: u32,
    sequence: EliminationSequence<T>,
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
    pub(crate) ground: Option<u32>,
    pub(crate) factor: BlockFactor<T>,
}

impl<T> Block<T> {
    fn ground(&self) -> Option<usize> {
        self.ground.map(|ground| ground as usize)
    }
}

/// Block-contiguous order to input order: `forward[i]` is the input vertex at
/// permuted position `i`.
///
/// Applied through a scratch buffer rather than in place. An in-place rotation
/// needs the cycle decomposition, which measured slower in both phases — the
/// round trip is pure random access on both sides, where gathering through
/// scratch keeps one side sequential per pass.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub(crate) struct Permutation {
    forward: Vec<u32>,
}

impl Permutation {
    /// `None` for the identity, which is what leaves connected input — the common
    /// case — permutation-free and allocation-free on every solve.
    pub(crate) fn from_forward(forward: &[u32]) -> Option<Self> {
        if forward.iter().enumerate().all(|(i, &v)| i as u32 == v) {
            return None;
        }
        Some(Self {
            forward: forward.to_vec(),
        })
    }

    /// `scratch[i] <- values[forward[i]]`
    fn gather_into<T: Copy>(&self, values: &[T], scratch: &mut [T]) {
        for (slot, &source) in scratch.iter_mut().zip(self.forward.iter()) {
            *slot = values[source as usize];
        }
    }

    /// `values[forward[i]] <- scratch[i]`
    fn scatter_from<T: Copy>(&self, scratch: &[T], values: &mut [T]) {
        for (&value, &target) in scratch.iter().zip(self.forward.iter()) {
            values[target as usize] = value;
        }
    }
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
}

#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
#[serde(bound(deserialize = "T: serde::de::DeserializeOwned"))]
struct FactorData<T> {
    n: usize,
    original_n: usize,
    permutation: Option<Permutation>,
    blocks: Vec<Block<T>>,
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
        };
        factor.validate_structure()?;
        Ok(factor)
    }
}

// Structural validation (no numeric `T` bound). Only the serde boundary needs
// it: the builder produces the invariants by construction, and the solve
// kernels index safe slices, so a corrupt factor could only ever panic on a
// bounds check rather than read past its storage. Without that boundary there is
// nothing to validate, so the whole path compiles away.
#[cfg(any(feature = "serde", test))]
impl<T> Factor<T> {
    /// Check the invariants the solve path relies on: `original_n <= n`, blocks
    /// tiling `[0, n)` with an in-range ground, and a permutation of `0..n`.
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
            // In-range but inconsistent writes `-sum` into a live variable.
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

#[cfg(any(feature = "serde", test))]
impl Permutation {
    fn validate_for_dim(&self, n: usize) -> Result<(), FactorError> {
        // A short map would leave the tail of `values` unwritten by `scatter_from`.
        if self.forward.len() != n {
            return Err(FactorError::PermutationInvalid {
                position: self.forward.len(),
            });
        }
        let mut seen = vec![false; n];
        for &position in &self.forward {
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
///
/// Only the right-hand side gets an error variant: its length comes from data
/// the caller may not control. A short work buffer is caller-side misuse —
/// [`Factor::n`] is the authority on the size — so it panics instead.
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
        }
    }
}

impl std::error::Error for SolveError {}

impl<T> BlockFactor<T> {
    fn n(&self) -> usize {
        self.n
    }

    fn n_steps(&self) -> usize {
        self.sequence.n_steps()
    }
}

#[cfg(any(feature = "serde", test))]
impl<T> BlockFactor<T> {
    fn validate(&self) -> Result<(), FactorError> {
        if self.anchor as usize >= self.n {
            return Err(FactorError::BlockAnchorInvalid {
                anchor: self.anchor as usize,
                n: self.n,
            });
        }
        self.sequence.validate_for_dim(self.n)
    }
}

impl<T> BlockFactor<T>
where
    T: num_traits::Float + Send + Sync + 'static,
{
    pub(crate) fn approx(n: usize, anchor: u32, sequence: EliminationSequence<T>) -> Self {
        Self {
            n,
            anchor,
            sequence,
        }
    }

    fn solve_raw(&self, values: &mut [T]) {
        let sequence = &self.sequence;
        for index in 0..sequence.n_steps() {
            sequence.step(index).apply_forward(values);
        }
        for index in (0..sequence.n_steps()).rev() {
            sequence.step(index).apply_backward(values);
        }
    }

    /// Solves the block system, leaving `values[pin]` equal to zero, where `pin`
    /// is the ground vertex if this block has one and the anchor otherwise. A
    /// floating block's result is determined only up to a constant;
    /// [`Self::solve_recovered`] projects that out.
    fn apply_anchored(&self, values: &mut [T], ground: Option<usize>) {
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
        let pin = ground.unwrap_or(self.anchor as usize);
        self.solve_raw(values);
        let pinned = values[pin];
        for value in values.iter_mut() {
            *value = *value - pinned;
        }
    }

    fn solve_recovered(&self, values: &mut [T], ground: Option<usize>) {
        self.apply_anchored(values, ground);
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
        let Some(count) = num_traits::cast::<usize, T>(values.len()) else {
            return;
        };
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
    ) -> Self {
        debug_assert!(forward.is_empty() || forward.len() == n);
        Self {
            n,
            original_n,
            permutation: Permutation::from_forward(forward),
            blocks,
        }
    }

    pub(crate) fn empty(original_n: usize) -> Self {
        Self {
            n: 0,
            original_n,
            permutation: None,
            blocks: Vec::new(),
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

    /// Total number of elimination steps across all blocks.
    pub fn n_steps(&self) -> usize {
        self.blocks.iter().map(|block| block.factor.n_steps()).sum()
    }

    #[inline]
    fn assert_work_fits(&self, work: &[T]) {
        assert!(
            work.len() >= self.n,
            "work buffer too small: got {}, need at least {}",
            work.len(),
            self.n
        );
    }

    #[inline]
    fn validate_rhs(&self, b: &[T]) -> Result<(), SolveError> {
        if b.len() > self.original_n {
            return Err(SolveError::RhsLengthExceedsFactor {
                rhs_len: b.len(),
                factor_dim: self.original_n,
            });
        }
        Ok(())
    }

    fn solve_kernel(&self, b: &[T], work: &mut [T]) {
        work[..b.len()].copy_from_slice(b);
        work[b.len()..self.n].fill(T::zero());
        self.for_each_block(work, |block, values| {
            block.factor.solve_recovered(values, block.ground());
        });
    }

    fn for_each_block(&self, values: &mut [T], mut solve: impl FnMut(&Block<T>, &mut [T])) {
        #[cfg(any(feature = "serde", test))]
        debug_assert_eq!(self.validate_structure(), Ok(()));
        let values = &mut values[..self.n];
        let Some(permutation) = &self.permutation else {
            Self::solve_blocks(&self.blocks, values, &mut solve);
            return;
        };
        // Only a disconnected input whose components interleave with input
        // numbering reaches here, so the common case never allocates.
        let mut scratch = vec![T::zero(); self.n];
        permutation.gather_into(values, &mut scratch);
        Self::solve_blocks(&self.blocks, &mut scratch, &mut solve);
        permutation.scatter_from(&scratch, values);
    }

    // Extracted only to serve both arms of `for_each_block`; left out of line it
    // measurably blocked the closure from inlining into the common path.
    #[inline(always)]
    fn solve_blocks(
        blocks: &[Block<T>],
        values: &mut [T],
        solve: &mut impl FnMut(&Block<T>, &mut [T]),
    ) {
        for block in blocks {
            let start = block.start as usize;
            solve(block, &mut values[start..start + block.factor.n()]);
        }
    }

    /// Solve `M x = b`, returning a newly allocated solution.
    ///
    /// For singular `M` (a pure Laplacian, or a graph that splits into components)
    /// this is the zero-mean least-squares solution: `M x == b` does not hold.
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, SolveError> {
        let mut work = vec![T::zero(); self.n];
        self.solve_into(b, &mut work)?;
        work.truncate(self.original_n);
        Ok(work)
    }

    /// Solve `M x = b` into a caller-provided work buffer.
    ///
    /// # Errors
    ///
    /// Returns [`SolveError::RhsLengthExceedsFactor`] if `b.len() > self.original_n()`.
    ///
    /// # Panics
    ///
    /// If `work.len() < self.n()`.
    pub fn solve_into(&self, b: &[T], work: &mut [T]) -> Result<(), SolveError> {
        self.validate_rhs(b)?;
        self.assert_work_fits(work);
        self.solve_kernel(b, work);
        Ok(())
    }

    /// Solve in place, skipping the zero-mean canonicalization that [`Self::solve`]
    /// applies to floating blocks.
    ///
    /// Each block is left anchored: one variable per block is pinned to zero — the
    /// ground vertex where augmentation added one, the un-eliminated vertex
    /// otherwise. For SDDM input that is the ground vertex, so the result already
    /// solves `M x = b` and matches [`Self::solve_into`].
    ///
    /// For Laplacian input it differs from [`Self::solve_into`] by a constant per
    /// block.
    ///
    /// # Panics
    ///
    /// If `values.len() < self.n()`.
    pub fn solve_in_place(&self, values: &mut [T]) {
        self.assert_work_fits(values);
        self.for_each_block(values, |block, values| {
            block.factor.apply_anchored(values, block.ground());
        });
    }
}
