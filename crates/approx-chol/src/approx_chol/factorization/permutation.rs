//! Block-contiguous ordering: the map between input coordinates and the contiguous
//! ranges the blocks tile.

#[cfg(any(feature = "serde", test))]
use super::FactorError;

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
    pub(super) forward: Vec<u32>,
}

impl Permutation {
    /// `None` for the identity, which is what leaves connected input — the common
    /// case — permutation-free and allocation-free on every solve.
    ///
    /// Takes the block order by value: it is already the map this needs, so there
    /// is nothing to copy out of it.
    pub(crate) fn from_order(forward: Vec<u32>) -> Option<Self> {
        if forward.iter().enumerate().all(|(i, &v)| i as u32 == v) {
            return None;
        }
        Some(Self { forward })
    }

    /// `scratch[i] <- values[forward[i]]`
    pub(super) fn gather_into<T: Copy>(&self, values: &[T], scratch: &mut [T]) {
        for (slot, &source) in scratch.iter_mut().zip(self.forward.iter()) {
            *slot = values[source as usize];
        }
    }

    /// `values[forward[i]] <- scratch[i]`
    pub(super) fn scatter_from<T: Copy>(&self, scratch: &[T], values: &mut [T]) {
        for (&value, &target) in scratch.iter().zip(self.forward.iter()) {
            values[target as usize] = value;
        }
    }
}

#[cfg(any(feature = "serde", test))]
impl Permutation {
    pub(super) fn validate_for_dim(&self, n: usize) -> Result<(), FactorError> {
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
