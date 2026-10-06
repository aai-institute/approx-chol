use crate::types::{count_as_scalar, Real};

/// The exact embedding of `M x = b` as `L_aug [x; 0] = [b; -sum b]`, the ground in the last slot.
pub(super) fn pin_ground<T: Real>(slots: &mut [T]) {
    if let Some((ground, rest)) = slots.split_last_mut() {
        *ground = -compensated_sum(rest);
    }
}

pub(super) fn relative_to_last<T: Real>(values: &mut [T]) {
    if let Some(&last) = values.last() {
        shift(values, last);
    }
}

/// Nothing absorbs the null space, so project it out; an inconsistent rhs gets least squares.
pub(super) fn project_zero_mean<T: Real>(values: &mut [T]) {
    shift(
        values,
        compensated_sum(values) / count_as_scalar::<T, _>(values.len()),
    );
}

/// Offset to the last slot first, so a plain fold has no large offset to lose terms against.
pub(super) fn recenter_zero_mean<T: Real>(values: &mut [T]) {
    relative_to_last(values);
    let sum = values.iter().fold(T::zero(), |sum, &value| sum + value);
    shift(values, sum / count_as_scalar::<T, _>(values.len()));
}

fn shift<T: Real>(values: &mut [T], by: T) {
    for value in values.iter_mut() {
        *value = *value - by;
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

#[cfg(test)]
mod tests;
