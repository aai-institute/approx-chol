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
    format_version: CurrentFormat,
    permutation: P,
    blocks: B,
    #[serde(default)]
    fallbacks: F,
}

#[cfg(feature = "serde")]
type OwnedFactor<T> = FactorData<Option<Permutation>, Vec<Block<T>>, Vec<Fallback>>;

/// Deserializes from [`FACTOR_FORMAT_VERSION`] alone, so another version fails before any moved field.
#[cfg(feature = "serde")]
#[derive(Clone, Copy, Debug)]
struct CurrentFormat;

#[cfg(feature = "serde")]
impl serde::Serialize for CurrentFormat {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u32(FACTOR_FORMAT_VERSION)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for CurrentFormat {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let found = <u32 as serde::Deserialize>::deserialize(deserializer)?;
        if found != FACTOR_FORMAT_VERSION {
            return Err(serde::de::Error::invalid_value(
                serde::de::Unexpected::Other(&format!("format version {found:#010x}")),
                &Self,
            ));
        }
        Ok(Self)
    }
}

#[cfg(feature = "serde")]
impl serde::de::Expected for CurrentFormat {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "format version {FACTOR_FORMAT_VERSION:#010x}")
    }
}

#[cfg(feature = "serde")]
impl<T: serde::Serialize> serde::Serialize for Factor<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        FactorData {
            format_version: CurrentFormat,
            permutation: self.permutation.as_ref(),
            blocks: self.blocks.as_slice(),
            fallbacks: self.fallbacks.as_slice(),
        }
        .serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<T: num_traits::Float> TryFrom<OwnedFactor<T>> for Factor<T> {
    type Error = FactorError;

    fn try_from(data: OwnedFactor<T>) -> Result<Self, Self::Error> {
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
    /// Right-hand side of a length other than [`Factor::n`].
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
            Self::LengthMismatch { len, factor_dim } => write!(
                f,
                "rhs length {len} differs from matrix dimension {factor_dim}"
            ),
            Self::ScratchTooSmall {
                scratch_len,
                needed,
            } => write!(
                f,
                "scratch too small: got {scratch_len}, need at least {needed}"
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

    /// Total elimination steps: every slot but one per block, whichever arm factored it.
    pub fn n_steps(&self) -> usize {
        self.blocks.iter().map(Block::eliminated).sum()
    }

    /// Scratch length [`solve_in_place`](Self::solve_in_place) needs; zero when blocks solve `x` directly.
    pub fn scratch_len(&self) -> usize {
        if self.permutation.is_some() || self.slots != self.n {
            self.slots
        } else {
            0
        }
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

    /// Solve in place: `x` holds `b` on entry and the solution on return; scratch is overwritten, never read.
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
        let slots = &mut scratch[..needed];
        for (input, slot) in self.spans() {
            let slots = &mut slots[slot..slot + input.len()];
            match &self.permutation {
                None => slots.copy_from_slice(&x[input]),
                Some(permutation) => permutation.gather_into(x, input, slots),
            }
        }
        self.solve_blocks(slots);
        for (input, slot) in self.spans() {
            let slots = &slots[slot..slot + input.len()];
            match &self.permutation {
                None => x[input].copy_from_slice(slots),
                Some(permutation) => permutation.scatter_from(slots, input, x),
            }
        }
        Ok(())
    }
}
