use super::cholesky::Cholesky;
#[cfg(any(feature = "serde", test))]
use super::FactorError;
use crate::types::{count_as_scalar, Real};

#[cfg(test)]
mod tests;

/// One component; its cholesky leaves one slot free and the variant is how that is fixed.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "T: serde::Serialize",
        deserialize = "T: serde::de::DeserializeOwned + num_traits::Float"
    ))
)]
#[derive(Clone, Debug)]
pub(crate) enum Block<T> {
    /// `L_C + diag(surplus_C)`, factored as `L_C` plus a ground in the last slot.
    Grounded(Cholesky<T>),
    /// `L_C`, whose solution is the zero-mean one.
    Floating(Cholesky<T>),
}

impl<T> Block<T> {
    fn cholesky(&self) -> &Cholesky<T> {
        match self {
            Self::Grounded(cholesky) | Self::Floating(cholesky) => cholesky,
        }
    }

    pub(super) fn eliminated(&self) -> usize {
        self.cholesky().eliminated()
    }

    /// Input vertices, which is every slot but a ground.
    pub(super) fn vertices(&self) -> usize {
        match self {
            Self::Grounded(cholesky) => cholesky.eliminated(),
            Self::Floating(cholesky) => cholesky.eliminated() + 1,
        }
    }

    pub(super) fn slots(&self) -> usize {
        self.cholesky().eliminated() + 1
    }
}

#[cfg(any(feature = "serde", test))]
impl<T: num_traits::Float> Block<T> {
    pub(super) fn validate(&self) -> Result<(), FactorError> {
        self.cholesky().validate()
    }
}

impl<T: Real> Block<T> {
    /// `slots` holds the right-hand side, then the solution; a ground's input entry is unread.
    pub(super) fn solve(&self, slots: &mut [T]) {
        let len = count_as_scalar::<T, _>(slots.len());
        match self {
            Self::Grounded(cholesky) => {
                // The exact embedding of `M x = b` as `L_aug [x; 0] = [b; -sum b]`.
                if let Some((ground, rest)) = slots.split_last_mut() {
                    *ground = -compensated_sum(rest);
                }
                cholesky.apply(slots);
                // Whichever slot the factor left free, the ground is what reads zero.
                shift_by_last(slots);
            }
            Self::Floating(cholesky) => {
                // Nothing absorbs the null space, so project it out; an inconsistent rhs gets least squares.
                shift(slots, compensated_sum(slots) / len);
                cholesky.apply(slots);
                // Offset to the last slot first, so a plain fold has no large offset to lose terms against.
                shift_by_last(slots);
                let sum = slots.iter().fold(T::zero(), |sum, &value| sum + value);
                shift(slots, sum / len);
            }
        }
    }
}

fn shift<T: Real>(values: &mut [T], by: T) {
    for value in values.iter_mut() {
        *value = *value - by;
    }
}

fn shift_by_last<T: Real>(values: &mut [T]) {
    if let Some(&last) = values.last() {
        shift(values, last);
    }
}

/// A plain fold drops the small terms of a large block; branchless TwoSum is cheaper than Neumaier.
fn compensated_sum<T: Real>(values: &[T]) -> T {
    let mut sum = T::zero();
    let mut compensation = T::zero();
    for &value in values {
        let next = sum + value;
        let back = next - sum;
        compensation = compensation + ((sum - (next - back)) + (value - back));
        sum = next;
    }
    sum + compensation
}
