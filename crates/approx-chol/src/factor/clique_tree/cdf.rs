use rand::rngs::Xoshiro256PlusPlus;
use rand::Rng;

use crate::Real;

const LINEAR_THRESHOLD: usize = 32;

/// The weight past `position` of a star's prefix sums, all of it for `None`: one subtraction, as each draw's range.
pub(super) fn suffix_past<T: Real>(cumsum: &[T], position: Option<usize>) -> T {
    let total = *cumsum.last().expect("a prepared star");
    position.map_or(total, |p| total - cumsum[p])
}

/// A position in the suffix past `position`, clamped to its last: an `f32` draw can round up to the total; a linear scan up to 32 entries (256 measured +4-5%).
#[inline(always)] // Otherwise LLVM outlines it from an AC fold: a call per draw.
pub(super) fn draw_after<T: Real>(
    cumsum: &[T],
    rng: &mut Xoshiro256PlusPlus,
    position: usize,
) -> usize {
    let (start, end) = (position + 1, cumsum.len());
    // 53 bits times a power of two is exact, so never 1.0.
    let u = ((rng.next_u64() >> 11) as f64) * (1.0 / (1u64 << 53) as f64);
    let u = T::from(u).expect("the draw is representable in T");
    let r = u * suffix_past(cumsum, Some(position)) + cumsum[position];
    let k = if end - start <= LINEAR_THRESHOLD {
        let mut k = start;
        while k < end && cumsum[k] < r {
            k += 1;
        }
        k
    } else {
        cumsum[start..end].partition_point(|&c| c < r) + start
    };
    k.min(end - 1)
}
