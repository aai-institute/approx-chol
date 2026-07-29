use core::cmp::Ordering;

use num_traits::{Float, NumCast};

/// Internal scalar bound for approximate Cholesky kernels. Nothing but the bound:
/// the tolerances below are the algorithm's policy, not the scalar's capability,
/// and they read the same way `count_as_scalar` and `float_total_cmp` do.
pub(crate) trait Real: Float + Send + Sync + 'static {}

impl<T> Real for T where T: Float + Send + Sync + 'static {}

/// Pick a threshold by the type's precision: `single` for `f32`-width scalars,
/// `double` otherwise.
#[inline]
fn by_precision<T: Float>(single: f64, double: f64) -> T {
    let value = if core::mem::size_of::<T>() <= 4 {
        single
    } else {
        double
    };
    <T as NumCast>::from(value).unwrap_or_else(T::epsilon)
}

/// Near-zero threshold for numeric guards: the scale below which a weight or pivot
/// is not resolvable, so dividing by it is what breaks rather than the value itself
/// being wrong.
#[inline]
pub(crate) fn near_zero<T: Float>() -> T {
    by_precision(1e-6, 1e-14)
}

/// Relative slack on a row sum: a departure from zero smaller than this fraction of
/// the row's magnitude is not real. Both signs read it, so the deficit that is not a
/// dominance error and the surplus not worth grounding are one size — four orders
/// coarser than [`near_zero`], which answers a different question.
#[inline]
pub(crate) fn row_sum_slack<T: Float>() -> T {
    by_precision(1e-6, 1e-10)
}

/// The smallest row surplus worth closing with a ground edge. Three floors, each
/// rejecting a different way a surplus fails to be dominance worth acting on:
///   policy    — too small relative to the row to matter, capped absolutely so a
///               1e12-scale row's real surplus is not swallowed;
///   noise     — inside the error this row's own sum could have accumulated over
///               its `terms` additions, so it may not be there at all;
///   resolvable— below the pivot scale the elimination can invert, so grounding on
///               it manufactures a link the solve cannot use and silently returns
///               the right-hand side unchanged.
///
/// `row_tolerance` is the caller's [`row_sum_slack`] against the row, the same
/// quantity its dominance check rejects a deficit by. All three floors are pinned by
/// tests: `*_scale_*`, `*_noise_floor_*`, `sub_near_zero_*`.
#[inline]
pub(crate) fn augmentation_floor<T: Float>(scale: T, row_tolerance: T, terms: usize) -> T {
    let policy = row_tolerance.min(T::epsilon().sqrt());
    let noise = T::epsilon() * scale * count_as_scalar::<T, _>(terms);
    policy.max(noise).max(near_zero::<T>())
}

/// A count (edge multiplicity, neighbor count) as a scalar, for the weight and
/// sort-key formulas that multiply or divide by it.
///
/// `f32` and `f64` represent every such count, so the cast only fails for an
/// exotic `Float`. It panics there rather than substituting a value: every
/// substitute a call site could pick — skip the neighbor, use one, skip the whole
/// edge split — silently corrupts the factor instead.
#[inline]
pub(crate) fn count_as_scalar<T: Float, N: num_traits::ToPrimitive>(count: N) -> T {
    <T as NumCast>::from(count).expect("count is representable in T")
}

/// Total ordering for floats, NaN last. `partial_cmp` alone returns `None` at NaN,
/// which violates the total order Rust's sorts require (1.81+ panics on it).
#[inline]
pub(crate) fn float_total_cmp<T: Float>(a: &T, b: &T) -> Ordering {
    a.partial_cmp(b)
        .unwrap_or_else(|| a.is_nan().cmp(&b.is_nan()))
}
