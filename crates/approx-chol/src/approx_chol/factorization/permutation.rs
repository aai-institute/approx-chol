#[cfg(any(feature = "serde", test))]
use super::FactorError;

/// `forward[i]` is the input vertex at block position `i`; scratch beat in-place (measured).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub(crate) struct Permutation {
    pub(super) forward: Vec<u32>,
}

impl Permutation {
    /// `None` for the identity, keeping connected input allocation-free on every solve.
    pub(crate) fn from_order(forward: Vec<u32>) -> Option<Self> {
        if forward.iter().enumerate().all(|(i, &v)| i as u32 == v) {
            return None;
        }
        Some(Self { forward })
    }

    pub(super) fn gather_into<T: Copy>(&self, values: &[T], start: usize, out: &mut [T]) {
        for (slot, &source) in out.iter_mut().zip(&self.forward[start..]) {
            *slot = values[source as usize];
        }
    }

    pub(super) fn scatter_from<T: Copy>(&self, slots: &[T], start: usize, values: &mut [T]) {
        for (&value, &target) in slots.iter().zip(&self.forward[start..]) {
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
