use crate::types::{count_as_scalar, Real};

/// The exact embedding of `M x = b` as `L_aug [x; 0] = [b; -sum b]`: the ground's entry
/// is the last slot.
pub(super) fn pin_ground<T: Real>(slots: &mut [T]) {
    if let Some((ground, rest)) = slots.split_last_mut() {
        *ground = -compensated_sum(rest);
    }
}

/// Whichever vertex the factor left free, the ground is what reads zero.
pub(super) fn relative_to_ground<T: Real>(slots: &mut [T]) {
    let Some(&ground) = slots.last() else {
        return;
    };
    for value in slots.iter_mut() {
        *value = *value - ground;
    }
}

/// Removes `span{1}`, the null space of a floating block, so an inconsistent right-hand
/// side gives least squares and the solution comes back zero-mean.
pub(super) fn project_zero_mean<T: Real>(values: &mut [T]) {
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
