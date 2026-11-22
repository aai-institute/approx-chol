use rand::rngs::Xoshiro256PlusPlus;
use rand::Rng;

const LINEAR_THRESHOLD: usize = 32;

/// The weight past `position` of a star's prefix sums, all of it for `None`: one subtraction, as each draw's range.
pub(super) fn suffix_past(cumsum: &[f64], position: Option<usize>) -> f64 {
    let total = *cumsum.last().expect("a prepared star");
    position.map_or(total, |p| total - cumsum[p])
}

/// A position in the suffix past `position`, clamped to its last: a draw can round up to the total.
pub(super) fn draw_after(cumsum: &[f64], rng: &mut Xoshiro256PlusPlus, position: usize) -> usize {
    let (start, end) = (position + 1, cumsum.len());
    // 53 bits times a power of two is exact, so never 1.0.
    let u = ((rng.next_u64() >> 11) as f64) * (1.0 / (1u64 << 53) as f64);
    let r = u * suffix_past(cumsum, Some(position)) + cumsum[position];
    (start + first_at_or_above(&cumsum[start..end], r)).min(end - 1)
}

/// The first of never-decreasing prefix sums at or above `r`, `len` if none: up to 32 the count below `r`, branchless, as a scan's exit mispredicts once per draw (a scan to 256 measured +4-5%; the count at 16-128, ±3% of 32).
fn first_at_or_above(window: &[f64], r: f64) -> usize {
    if window.len() <= LINEAR_THRESHOLD {
        window.iter().filter(|&&c| c < r).count()
    } else {
        window.partition_point(|&c| c < r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_search_stops_where_a_scan_does_on_ties_and_past_the_total() {
        let weights: [&[f64]; 4] = [
            &[1.0; 40],
            &[0.5, 0.0, 0.0, 2.0, 2.0, 0.0, 7.0, 7.0, 7.0],
            &[3.0, 0.0, 0.0, 0.0],
            &[1e16, 1.0, 1.0, 1.0, 1e-8, 1.0],
        ];
        for weights in weights {
            let cumsum: Vec<f64> = (weights.iter())
                .scan(0.0, |acc, w| {
                    *acc += w;
                    Some(*acc)
                })
                .collect();
            let total = cumsum[cumsum.len() - 1];
            let between = cumsum.windows(2).map(|pair| 0.5 * (pair[0] + pair[1]));
            let draws = cumsum.iter().copied().chain(between);
            for start in 0..cumsum.len() {
                let window = &cumsum[start..];
                for r in draws.clone().chain([-1.0, total * 2.0, f64::INFINITY]) {
                    let scanned = window.iter().take_while(|&&c| c < r).count();
                    assert_eq!(first_at_or_above(window, r), scanned, "r = {r}");
                }
            }
        }
    }
}
