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
    pin: Pin,
    sequence: EliminationSequence<T>,
}

/// Every block's Laplacian is singular, so one variable is pinned to zero. Which
/// one it is also decides how the right-hand side gets into the block's range,
/// so the two travel together rather than as an `Option` each solve re-reads.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug)]
pub(crate) enum Pin {
    /// A Gremban ground vertex, which absorbs the null-space component exactly.
    Ground(u32),
    /// The un-eliminated vertex of a block that has no ground vertex, so the
    /// null-space component is projected out instead.
    Floating(u32),
}

impl Pin {
    #[inline]
    fn index(self) -> usize {
        match self {
            Self::Ground(vertex) | Self::Floating(vertex) => vertex as usize,
        }
    }

    /// Whether the anchored solution still carries an arbitrary constant.
    #[inline]
    fn is_floating(self) -> bool {
        matches!(self, Self::Floating(_))
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
    blocks: Vec<BlockFactor<T>>,
}

#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
#[serde(bound(deserialize = "T: serde::de::DeserializeOwned"))]
struct FactorData<T> {
    n: usize,
    original_n: usize,
    permutation: Option<Permutation>,
    blocks: Vec<BlockFactor<T>>,
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
    /// covering `n` with an in-range pin, and a permutation of `0..n`.
    fn validate_structure(&self) -> Result<(), FactorError> {
        if self.original_n > self.n {
            return Err(FactorError::OriginalDimExceedsInternal {
                original_n: self.original_n,
                n: self.n,
            });
        }
        // Each block's range begins where the previous one ended, so they tile
        // `[0, n)` by construction and only the total is left to check. The
        // checked sum is what stops a wrapped total from passing as that total.
        let mut covered = 0usize;
        for block in &self.blocks {
            block.validate()?;
            covered = covered
                .checked_add(block.n())
                .ok_or(FactorError::BlockDimsDoNotCoverFactor { covered, n: self.n })?;
        }
        if covered != self.n {
            return Err(FactorError::BlockDimsDoNotCoverFactor { covered, n: self.n });
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
        // An in-range but wrong ground pin writes `-sum` into a live variable; an
        // out-of-range one indexes past the block. A zero-dimension block, which
        // no pin can be an index of, is rejected here too.
        if self.pin.index() >= self.n {
            return Err(FactorError::BlockPinInvalid {
                pin: self.pin.index(),
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
    pub(crate) fn approx(n: usize, pin: Pin, sequence: EliminationSequence<T>) -> Self {
        Self { n, pin, sequence }
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

    /// Solves the block system, leaving the pinned variable equal to zero. A
    /// floating block's result is determined only up to a constant;
    /// [`Self::solve_recovered`] projects that out.
    fn apply_anchored(&self, values: &mut [T]) {
        // Every block is a singular Laplacian, so it solves only a zero-sum
        // right-hand side; both arms put `values` in that range.
        match self.pin {
            // The exact embedding of `M x = b` as `L_aug [x; 0] = [b; -sum b]`.
            Pin::Ground(ground) => {
                let ground = ground as usize;
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
            Pin::Floating(_) => Self::project_zero_mean(values),
        }
        self.solve_raw(values);
        let pinned = values[self.pin.index()];
        for value in values.iter_mut() {
            *value = *value - pinned;
        }
    }

    fn solve_recovered(&self, values: &mut [T]) {
        self.apply_anchored(values);
        // Pinning the ground vertex already gives the SDDM solution; a floating
        // block is determined only up to a constant, fixed here to zero mean.
        if self.pin.is_floating() {
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
    pub(crate) fn from_blocks(
        n: usize,
        original_n: usize,
        permutation: Option<Permutation>,
        blocks: Vec<BlockFactor<T>>,
    ) -> Self {
        let factor = Self {
            n,
            original_n,
            permutation,
            blocks,
        };
        // Once per factorization, not once per solve: the solve path relies on
        // these invariants but cannot violate them.
        #[cfg(any(feature = "serde", test))]
        debug_assert_eq!(factor.validate_structure(), Ok(()));
        factor
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
        self.blocks.iter().map(BlockFactor::n_steps).sum()
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

    fn solve_kernel(&self, b: &[T], work: &mut [T]) {
        work[..b.len()].copy_from_slice(b);
        work[b.len()..self.n].fill(T::zero());
        self.for_each_block(work, BlockFactor::solve_recovered);
    }

    fn for_each_block(&self, values: &mut [T], mut solve: impl FnMut(&BlockFactor<T>, &mut [T])) {
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
        blocks: &[BlockFactor<T>],
        values: &mut [T],
        solve: &mut impl FnMut(&BlockFactor<T>, &mut [T]),
    ) {
        let mut start = 0usize;
        for block in blocks {
            let end = start + block.n();
            solve(block, &mut values[start..end]);
            start = end;
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
        if b.len() > self.original_n {
            return Err(SolveError::RhsLengthExceedsFactor {
                rhs_len: b.len(),
                factor_dim: self.original_n,
            });
        }
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
        self.for_each_block(values, BlockFactor::apply_anchored);
    }
}
