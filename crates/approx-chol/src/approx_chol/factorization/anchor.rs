use crate::types::{count_as_scalar, Real};

/// Whether pinning a block's last variable (its Laplacian null space is `span{1}`) is the answer.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Anchor {
    Ground,
    Floating,
}

impl Anchor {
    /// Make `values` zero-sum, which is the only right-hand side a block solves.
    pub(super) fn prepare<T: Real>(self, values: &mut [T]) {
        match self {
            // The exact embedding of `M x = b` as `L_aug [x; 0] = [b; -sum b]`.
            Self::Ground => {
                let Some((pinned, rest)) = values.split_last_mut() else {
                    return;
                };
                *pinned = -compensated_sum(rest);
            }
            // Nothing absorbs the null space, so project it out; an inconsistent rhs gets least squares.
            Self::Floating => {
                let mean = compensated_sum(values) / count_as_scalar::<T, _>(values.len());
                for value in values.iter_mut() {
                    *value = *value - mean;
                }
            }
        }
    }

    pub(super) fn recover<T: Real>(self, values: &mut [T], canonical: bool) {
        let Some(&pinned) = values.last() else {
            return;
        };
        for value in values.iter_mut() {
            *value = *value - pinned;
        }
        if canonical && self == Self::Floating {
            // Pinned to zero above, so no offset is left for compensation to keep.
            let sum = values.iter().fold(T::zero(), |sum, &value| sum + value);
            let mean = sum / count_as_scalar::<T, _>(values.len());
            for value in values.iter_mut() {
                *value = *value - mean;
            }
        }
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
    // Once the sum overflows TwoSum's error is NaN; keep the plain fold's infinity.
    if compensation.is_finite() {
        sum + compensation
    } else {
        sum
    }
}

#[cfg(test)]
mod tests;
