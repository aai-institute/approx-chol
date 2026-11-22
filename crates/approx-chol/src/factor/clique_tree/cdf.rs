use rand::rngs::Xoshiro256PlusPlus;
use rand::Rng;

const LINEAR_THRESHOLD: usize = 32;

/// The weight past `position` of a star's prefix sums, all of it for `None`: one subtraction, as each draw's range.
pub(super) fn suffix_past(cumsum: &[f64], position: Option<usize>) -> f64 {
    let total = *cumsum.last().expect("a prepared star");
    position.map_or(total, |p| total - cumsum[p])
}

/// A position in the suffix past `position`, clamped to its last: a draw can round up to the total; up to 32 entries a branchless count, the first position at or above the draw as the prefix sums never decrease (a scan to 256 measured +4-5%; the count at 16-128, ±3% of 32).
pub(super) fn draw_after(cumsum: &[f64], rng: &mut Xoshiro256PlusPlus, position: usize) -> usize {
    let (start, end) = (position + 1, cumsum.len());
    // 53 bits times a power of two is exact, so never 1.0.
    let u = ((rng.next_u64() >> 11) as f64) * (1.0 / (1u64 << 53) as f64);
    let r = u * suffix_past(cumsum, Some(position)) + cumsum[position];
    let k = if end - start <= LINEAR_THRESHOLD {
        // A scan that stops at the crossing mispredicts once per draw; the count reads the whole window instead.
        start + (cumsum[start..end].iter()).filter(|&&c| c < r).count()
    } else {
        cumsum[start..end].partition_point(|&c| c < r) + start
    };
    k.min(end - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    /// The scan that stops at the first prefix sum at or above the draw, clamped as `draw_after` clamps.
    fn scanned(cumsum: &[f64], rng: &mut Xoshiro256PlusPlus, position: usize) -> usize {
        let u = ((rng.next_u64() >> 11) as f64) * (1.0 / (1u64 << 53) as f64);
        let r = u * suffix_past(cumsum, Some(position)) + cumsum[position];
        let mut k = position + 1;
        while k < cumsum.len() && cumsum[k] < r {
            k += 1;
        }
        k.min(cumsum.len() - 1)
    }

    #[test]
    fn the_count_lands_where_the_scan_stops_through_ties_and_empty_suffixes() {
        let weights: [&[f64]; 4] = [
            &[1.0; 33],
            &[0.5, 0.0, 0.0, 2.0, 2.0, 0.0, 1e-300, 7.0, 7.0, 7.0],
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
            for position in 0..cumsum.len() - 1 {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(position as u64);
                let mut reference = rng.clone();
                for _ in 0..200 {
                    assert_eq!(
                        draw_after(&cumsum, &mut rng, position),
                        scanned(&cumsum, &mut reference, position)
                    );
                }
            }
        }
    }
}
