use crate::types::{count_as_scalar, Real};

/// Every block is a connected pure Laplacian, so its null space is `span{1}` and its
/// last variable is pinned; this is whether pinning it is by itself the answer.
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
            // Nothing to absorb the null-space component, so project it out; an
            // inconsistent right-hand side then gives least squares.
            Self::Floating => project_zero_mean(values),
        }
    }

    pub(super) fn recover<T: Real>(self, values: &mut [T]) {
        let Some(&pinned) = values.last() else {
            return;
        };
        for value in values.iter_mut() {
            *value = *value - pinned;
        }
        if self == Self::Floating {
            project_zero_mean(values);
        }
    }
}

fn project_zero_mean<T: Real>(values: &mut [T]) {
    let count = count_as_scalar::<T, _>(values.len());
    let mean = compensated_sum(values) / count;
    for value in values.iter_mut() {
        *value = *value - mean;
    }
}

/// Neumaier: a plain fold loses the small terms of a large block.
fn compensated_sum<T: Real>(values: &[T]) -> T {
    let mut sum = T::zero();
    let mut compensation = T::zero();
    for &value in values {
        let next = sum + value;
        compensation = compensation
            + if sum.abs() >= value.abs() {
                (sum - next) + value
            } else {
                (value - next) + sum
            };
        sum = next;
    }
    sum + compensation
}

#[cfg(test)]
mod tests;
