#[cfg(any(feature = "serde", test))]
use super::FactorError;

/// `forward[i]` is the input vertex at block-contiguous position `i`. Applied through
/// scratch: an in-place cycle rotation measured slower in both phases.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub(crate) struct Permutation {
    pub(super) forward: Vec<u32>,
}

impl Permutation {
    /// `None` for the identity, which leaves connected input — the common case —
    /// allocation-free on every solve.
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
