use rand::rngs::Xoshiro256PlusPlus;
use rand::Rng;

const LINEAR_THRESHOLD: usize = 32;

/// The weight past `position` of a star's prefix sums, all of it for `None`: one subtraction, as each draw's range.
pub(super) fn suffix_past(cumsum: &[f64], position: Option<usize>) -> f64 {
    let total = *cumsum.last().expect("a prepared star");
    position.map_or(total, |p| total - cumsum[p])
}

/// A position in the suffix past `position`, clamped to its last: a draw can round up to the total; a linear scan up to 32 entries (256 measured +4-5%).
pub(super) fn draw_after(cumsum: &[f64], rng: &mut Xoshiro256PlusPlus, position: usize) -> usize {
    let (start, end) = (position + 1, cumsum.len());
    // 53 bits times a power of two is exact, so never 1.0.
    let u = ((rng.next_u64() >> 11) as f64) * (1.0 / (1u64 << 53) as f64);
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
