use super::block::Block;
use super::permutation::Permutation;
#[cfg(any(feature = "serde", test))]
use super::FactorError;
use core::fmt;

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
    if found != FACTOR_FORMAT_VERSION {
        return Err(serde::de::Error::custom(
            FactorError::UnsupportedFormatVersion {
                found,
                supported: FACTOR_FORMAT_VERSION,
            },
        ));
    }
    Ok(found)
}

#[cfg(feature = "serde")]
impl<T: num_traits::Float> TryFrom<OwnedFactor<T>> for Factor<T> {
    type Error = FactorError;

    fn try_from(data: OwnedFactor<T>) -> Result<Self, Self::Error> {
        // Only a payload that predates the field reaches here with another version.
        if data.format_version != FACTOR_FORMAT_VERSION {
            return Err(FactorError::UnsupportedFormatVersion {
                found: data.format_version,
                supported: FACTOR_FORMAT_VERSION,
            });
        }
        let factor = Self::of(data.permutation, data.blocks, data.fallbacks);
        factor.validate_structure()?;
        Ok(factor)
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Why an [`ExactBelow`](crate::Backend::ExactBelow) block was factored approximately.
pub enum Fallback {
    /// Dense elimination reached a pivot it could not use.
    InvalidPivot(crate::UnusablePivot),
    /// The dense copy would not fit; never fatal under any [`ExactFailure`](crate::ExactFailure).
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

/// Blocks arrive checked against their own cholesky; this covers what no single block sees.
#[cfg(any(feature = "serde", test))]
impl<T> Factor<T> {
    fn validate_structure(&self) -> Result<(), FactorError> {
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
    /// Solution buffer of a length other than [`Factor::n`].
    LengthMismatch {
        /// Provided length.
        len: usize,
        /// [`Factor::n`].
        factor_dim: usize,
    },
    /// Scratch shorter than [`Factor::scratch_len`].
    ScratchTooSmall {
        /// Provided scratch length.
        scratch_len: usize,
        /// [`Factor::scratch_len`].
        needed: usize,
    },
}

impl fmt::Display for SolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch { len, factor_dim } => {
                write!(f, "length {len} differs from factor dimension {factor_dim}")
            }
            Self::ScratchTooSmall {
                scratch_len,
                needed,
            } => write!(f, "scratch too small: got {scratch_len}, need {needed}"),
        }
    }
}

impl std::error::Error for SolveError {}

impl<T> Factor<T> {
    /// Blocks routed to exact Cholesky and factored approximately anyway.
    pub fn fallbacks(&self) -> &[Fallback] {
        &self.fallbacks
    }

    /// Dimension of the factored input.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Zero for floating input needing no permutation, else every block's slots, grounds included.
    pub fn scratch_len(&self) -> usize {
        if self.permutation.is_some() || self.slots != self.n {
            self.slots
        } else {
            0
        }
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

    /// (input start, vertices, first slot) per block; one span when no block holds a ground.
    fn spans(&self) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
        let whole = (self.slots == self.n).then_some((0, self.n, 0));
        let blocks = whole.is_none().then(|| {
            self.blocks.iter().scan((0, 0), |(input, slot), block| {
                let span = (*input, block.vertices(), *slot);
                *input += block.vertices();
                *slot += block.slots();
                Some(span)
            })
        });
        whole.into_iter().chain(blocks.into_iter().flatten())
    }
}

impl<T> Factor<T>
where
    T: num_traits::Float + Send + Sync + 'static,
{
    pub(crate) fn from_blocks(
        permutation: Option<Permutation>,
        blocks: Vec<Block<T>>,
        fallbacks: Vec<Fallback>,
    ) -> Self {
        #[cfg(any(feature = "serde", test))]
        debug_assert!(blocks.iter().all(|block| block.validate().is_ok()));
        let factor = Self::of(permutation, blocks, fallbacks);
        #[cfg(any(feature = "serde", test))]
        debug_assert_eq!(factor.validate_structure(), Ok(()));
        factor
    }

    /// Total elimination steps: every slot but one per block, whichever arm factored it.
    pub fn n_steps(&self) -> usize {
        self.blocks.iter().map(Block::eliminated).sum()
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

    /// Solve `M x = b`, returning the zero-mean least-squares solution for singular `M`.
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, SolveError> {
        let mut x = b.to_vec();
        let mut scratch = vec![T::zero(); self.scratch_len()];
        self.solve_in_place(&mut x, &mut scratch)?;
        Ok(x)
    }

    /// `x` holds `b` on entry and the solution on return; `scratch` contents never matter.
    pub fn solve_in_place(&self, x: &mut [T], scratch: &mut [T]) -> Result<(), SolveError> {
        if x.len() != self.n() {
            return Err(SolveError::LengthMismatch {
                len: x.len(),
                factor_dim: self.n(),
            });
        }
        let needed = self.scratch_len();
        if scratch.len() < needed {
            return Err(SolveError::ScratchTooSmall {
                scratch_len: scratch.len(),
                needed,
            });
        }
        if needed == 0 {
            self.solve_blocks(x);
            return Ok(());
        }
        // A ground slot keeps whatever scratch held: its block writes it before reading it.
        let work = &mut scratch[..needed];
        for (input, len, slot) in self.spans() {
            let slots = &mut work[slot..slot + len];
            match &self.permutation {
                None => slots.copy_from_slice(&x[input..input + len]),
                Some(permutation) => permutation.gather_into(x, input, slots),
            }
        }
        self.solve_blocks(work);
        for (input, len, slot) in self.spans() {
            let slots = &work[slot..slot + len];
            match &self.permutation {
                None => x[input..input + len].copy_from_slice(slots),
                Some(permutation) => permutation.scatter_from(slots, input, x),
            }
        }
        Ok(())
    }
}
