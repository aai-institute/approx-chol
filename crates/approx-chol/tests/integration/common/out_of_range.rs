use approx_chol::BuildError;

/// The smallest nonzero magnitude admitted (#163): `MIN_POSITIVE/ε`, `2^-970`.
pub const FLOOR: f64 = f64::MIN_POSITIVE / f64::EPSILON;
/// `f32`'s own floor, `2^-103`.
pub const F32_FLOOR: f32 = f32::MIN_POSITIVE / f32::EPSILON;

pub fn out_of_range(vertex: usize) -> BuildError {
    BuildError::OutOfRange { vertex }
}
