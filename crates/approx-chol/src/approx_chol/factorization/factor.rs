use super::block::Block;
use super::permutation::Permutation;
#[cfg(any(feature = "serde", test))]
use super::FactorError;
use crate::Fallback;
use core::fmt;
use core::ops::Range;

#[cfg(test)]
mod tests;

/// Bump the low half on any encoding change; the tag half stops an unversioned payload's `n` matching.
#[cfg(feature = "serde")]
pub const FACTOR_FORMAT_VERSION: u32 = 0x4143_0005;

#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(
        bound(deserialize = "T: serde::de::DeserializeOwned + num_traits::Float"),
        try_from = "OwnedFactor<T>"
    )
)]
#[derive(Clone, Debug)]
/// Exact or approximate Cholesky decomposition of an SDDM matrix.
pub struct Factor<T = f64> {
    n: usize,
    slots: usize,
    /// Over input vertices only; ground slots never appear in it.
    permutation: Option<Permutation>,
    blocks: Vec<Block<T>>,
    fallbacks: Vec<Fallback>,
}

/// One wire shape: owned decoding, borrowed encoding, so writing the version copies nothing.
#[cfg(feature = "serde")]
#[derive(serde::Serialize, serde::Deserialize)]
struct FactorData<P, B, F> {
    /// Defaulted, so a payload predating the field fails on its version, not a missing field.
    #[serde(default, deserialize_with = "current_version")]
    format_version: u32,
    permutation: P,
    blocks: B,
    #[serde(default)]
    fallbacks: F,
}

#[cfg(feature = "serde")]
type OwnedFactor<T> = FactorData<Option<Permutation>, Vec<Block<T>>, Vec<Fallback>>;

#[cfg(feature = "serde")]
impl<T: serde::Serialize> serde::Serialize for Factor<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        FactorData {
            format_version: FACTOR_FORMAT_VERSION,
            permutation: self.permutation.as_ref(),
            blocks: self.blocks.as_slice(),
            fallbacks: self.fallbacks.as_slice(),
        }
        .serialize(serializer)
    }
}

/// Checked as read, so another version fails on its version, not on the first moved field.
#[cfg(feature = "serde")]
fn current_version<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<u32, D::Error> {
    let found = <u32 as serde::Deserialize>::deserialize(deserializer)?;
    FactorError::check_version(found).map_err(serde::de::Error::custom)?;
    Ok(found)
}

#[cfg(feature = "serde")]
impl<T: num_traits::Float> TryFrom<OwnedFactor<T>> for Factor<T> {
    type Error = FactorError;

    fn try_from(data: OwnedFactor<T>) -> Result<Self, Self::Error> {
        // Only a payload missing the field reaches here unchecked.
        FactorError::check_version(data.format_version)?;
        let factor = Self::of(data.permutation, data.blocks, data.fallbacks);
        factor.validate_structure()?;
        Ok(factor)
    }
}

#[cfg(any(feature = "serde", test))]
impl<T: num_traits::Float> Factor<T> {
    fn validate_structure(&self) -> Result<(), FactorError> {
        for block in &self.blocks {
            block.validate()?;
        }
        if let Some(permutation) = &self.permutation {
            permutation.validate_for_dim(self.n)?;
        }
        Ok(())
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
/// Errors returned by fallible [`Factor`] solve methods.
pub enum SolveError {
    /// Right-hand side longer than [`Factor::n`].
    RhsLengthExceedsFactor {
        /// Provided RHS length.
        rhs_len: usize,
        /// Maximum accepted RHS length.
        factor_dim: usize,
    },
    /// Work buffer shorter than [`Factor::n`].
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
                "rhs length {rhs_len} exceeds matrix dimension {factor_dim}"
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

    /// Dimension of the factored input; ground slots stay internal.
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// The one place `n` and `slots` are ever written, so neither drifts from the blocks.
    fn of(
        permutation: Option<Permutation>,
        blocks: Vec<Block<T>>,
        fallbacks: Vec<Fallback>,
    ) -> Self {
        Self {
            n: blocks.iter().map(Block::vertices).sum(),
            slots: blocks.iter().map(Block::slots).sum(),
            permutation,
            blocks,
            fallbacks,
        }
    }

    /// Each block's input range and first slot.
    fn spans(&self) -> impl Iterator<Item = (Range<usize>, usize)> + '_ {
        self.blocks.iter().scan((0, 0), |(input, slot), block| {
            let span = (*input..*input + block.vertices(), *slot);
            *input += block.vertices();
            *slot += block.slots();
            Some(span)
        })
    }
}

impl<T> Factor<T>
where
    T: num_traits::Float + Send + Sync + 'static,
{
    pub(in crate::approx_chol) fn from_blocks(
        permutation: Option<Permutation>,
        blocks: Vec<Block<T>>,
        fallbacks: Vec<Fallback>,
    ) -> Self {
        let factor = Self::of(permutation, blocks, fallbacks);
        #[cfg(any(feature = "serde", test))]
        debug_assert_eq!(factor.validate_structure(), Ok(()));
        factor
    }

    pub(crate) fn empty() -> Self {
        Self::from_blocks(None, Vec::new(), Vec::new())
    }

    /// Total elimination steps: every slot but one per block, whichever arm factored it.
    pub fn n_steps(&self) -> usize {
        self.blocks.iter().map(Block::eliminated).sum()
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

    #[inline(always)]
    fn solve_blocks(&self, slots: &mut [T]) {
        let mut start = 0usize;
        for block in &self.blocks {
            let end = start + block.slots();
            block.solve(&mut slots[start..end]);
            start = end;
        }
    }

    /// `x` holds the input-dimension right-hand side on entry and the solution on return.
    fn solve_input(&self, x: &mut [T]) {
        let x = &mut x[..self.n()];
        if self.permutation.is_none() && self.slots == self.n {
            self.solve_blocks(x);
            return;
        }
        // A ground slot needs no initial value: its block writes it before reading it.
        let mut slots = vec![T::zero(); self.slots];
        for (input, slot) in self.spans() {
            let slots = &mut slots[slot..slot + input.len()];
            match &self.permutation {
                None => slots.copy_from_slice(&x[input]),
                Some(permutation) => permutation.gather_into(x, input, slots),
            }
        }
        self.solve_blocks(&mut slots);
        for (input, slot) in self.spans() {
            let slots = &slots[slot..slot + input.len()];
            match &self.permutation {
                None => x[input].copy_from_slice(slots),
                Some(permutation) => permutation.scatter_from(slots, input, x),
            }
        }
    }

    /// Solve `M x = b`, returning the zero-mean least-squares solution for singular `M`.
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, SolveError> {
        let mut work = vec![T::zero(); self.n()];
        self.solve_into(b, &mut work)?;
        Ok(work)
    }

    /// Solve `M x = b` into the first [`n`](Self::n) entries of a caller-provided buffer.
    pub fn solve_into(&self, b: &[T], work: &mut [T]) -> Result<(), SolveError> {
        if b.len() > self.n() {
            return Err(SolveError::RhsLengthExceedsFactor {
                rhs_len: b.len(),
                factor_dim: self.n(),
            });
        }
        self.validate_work(work)?;
        work[..b.len()].copy_from_slice(b);
        work[b.len()..self.n()].fill(T::zero());
        self.solve_input(work);
        Ok(())
    }

    /// Solve in place: the first [`n`](Self::n) entries hold `b` on entry and `x` on return.
    pub fn solve_in_place(&self, values: &mut [T]) -> Result<(), SolveError> {
        self.validate_work(values)?;
        self.solve_input(values);
        Ok(())
    }
}
