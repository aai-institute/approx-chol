use approx_chol_next::{factorize, Config, Real, ScalingReport, Sgdd};
use rstest::rstest;

use super::{grounded, reduce};

/// `[[a+s, -a], [-a, a+s]]`: dominant by `s` on both rows, so it floats exactly when `s` is noise.
fn surplus_pair<T: Real>(a: T, s: T) -> Sgdd<T> {
    reduce(&[0, 2, 4], &[0, 1, 0, 1], &[a + s, -a, -a, a + s]).expect("a pair is admitted")
}

fn solves_to<T: Real + std::fmt::LowerExp>(sgdd: &Sgdd<T>, rhs: &[T], expected: &[T]) {
    let solution = factorize(sgdd, &Config::default())
        .solve(rhs)
        .expect("solve");
    let tolerance = T::from(1e-3).unwrap();
    for (&got, &want) in solution.iter().zip(expected) {
        assert!(
            (got - want).abs() <= tolerance * want.abs(),
            "{got:e} vs {want:e}"
        );
    }
}

/// Pins the routing, not the solution: at a `2e-6` row scale summation error is about `1e-21`.
#[rstest]
#[case::far_above_the_floor(6e-12)]
#[case::above_the_floor(6e-15)]
// A floor 1e6 coarser than rounding once answered this as a singular Laplacian.
#[case::just_above_the_floor(6e-18)]
fn surplus_above_summation_error_grounds_and_solves(#[case] surplus: f64) {
    let sgdd = surplus_pair(1e-6, surplus);
    assert!(grounded(&sgdd), "surplus {surplus:e}");
    solves_to(&sgdd, &[1.0, 1.0], &[1.0 / surplus; 2]);
}

#[rstest]
#[case::below_the_floor(6e-22)]
#[case::not_representable_at_this_scale(1e-30)]
fn surplus_within_summation_error_floats(#[case] surplus: f64) {
    assert!(
        !grounded(&surplus_pair(1e-6, surplus)),
        "surplus {surplus:e}"
    );
}

/// The same window in `f32` at a `2e-3` row scale (#85): summation error is about `5e-10`.
#[rstest]
#[case::above_the_floor(1.2e-9, true)]
#[case::below_the_floor(2e-10, false)]
fn f32_surplus_is_judged_against_summation_error_alone(
    #[case] surplus: f32,
    #[case] expected: bool,
) {
    let sgdd = surplus_pair(1e-3f32, surplus);
    assert_eq!(grounded(&sgdd), expected, "surplus {surplus:e}");
}

fn drifted<T: Real>(drift: T) -> Sgdd<T> {
    surplus_pair(T::one(), drift)
}

/// One ulp either way is drift a single addition accounts for, so the pair floats.
#[rstest]
#[case::f32_one_ulp_over(f32::EPSILON)]
#[case::f32_one_ulp_under(-f32::EPSILON)]
#[case::f64_one_ulp_over(f64::EPSILON)]
#[case::f64_one_ulp_under(-f64::EPSILON)]
fn summation_roundoff_does_not_augment<T: Real + std::fmt::LowerExp>(#[case] drift: T) {
    assert!(!grounded(&drifted(drift)), "drift {drift:e}");
}

/// Mid-window for each precision, which no two-term sum invents; through 0.3.1 both floated (#85).
#[test]
fn surplus_beyond_summation_roundoff_augments() {
    assert!(grounded(&drifted(1e-6_f32)));
    assert!(grounded(&drifted(5e-11_f64)));
}

/// A deficit past rounding is reported, not zeroed (#91), even well inside the scaling tolerance.
#[test]
fn deficit_beyond_summation_roundoff_is_reported() {
    let reported = |sgdd: &Sgdd<f64>| sgdd.blocks()[0].scaling_report().is_some();
    assert!(reported(&drifted(-5e-11_f64)), "1e-9 tolerance is no cover");
    assert!(reported(&drifted(-32.0 * f64::EPSILON)));
    assert!(!reported(&drifted(-12.0 * f64::EPSILON)), "inside 16 eps");
    let f32_reported = |drift: f32| drifted(drift).blocks()[0].scaling_report().is_some();
    assert!(f32_reported(-32.0 * f32::EPSILON));
    assert!(!f32_reported(-12.0 * f32::EPSILON), "inside 16 eps");
}

/// Unit leaves, centre short of their sum by `short_ulps` ε; the leaves balance exactly.
fn star(leaves: u32, short_ulps: f64) -> Sgdd<f64> {
    let mut row_ptrs = vec![0u32, leaves + 1];
    let mut columns: Vec<u32> = (0..=leaves).collect();
    let mut values = vec![f64::from(leaves) * (1.0 - short_ulps * f64::EPSILON)];
    values.extend(std::iter::repeat_n(-1.0, leaves as usize));
    for leaf in 1..=leaves {
        columns.extend([0, leaf]);
        values.extend([-1.0, 1.0]);
        row_ptrs.push(columns.len() as u32);
    }
    reduce(&row_ptrs, &columns, &values).expect("a star is admitted")
}

/// A row of `k` entries rounds by `(k + 3)ε`: a degree-100 centre short by 50ε is rounding, by 200ε a deficit.
#[test]
fn a_rows_rounding_band_grows_with_its_degree() {
    assert!(star(100, 50.0).blocks()[0].scaling_report().is_none());
    assert!(star(100, 200.0).blocks()[0].scaling_report().is_some());
}

/// No scaling settled, so no settle band widens the row's: a degree-13 centre short by 24ε passes `max(16, 16)ε`.
#[test]
fn an_unscaled_deficit_past_its_row_rounding_is_reported() {
    let report = star(13, 24.0).blocks()[0]
        .scaling_report()
        .expect("reported");
    assert_eq!(report.iterations, 0, "unscaled");
    assert!(report.violation > 16.0 * f64::EPSILON, "{report:?}");
}

/// Main rejected these as not dominant; the rewrite admits them clamped and reports the violation.
#[rstest]
#[case::f64_pair_2e_9_short(drifted(-2e-9_f64).blocks()[0].scaling_report(), 2e-9)]
#[case::f32_pair_1e_5_short(drifted(-1e-5_f32).blocks()[0].scaling_report(), 1e-5)]
#[case::no_scaling_makes_it_dominant(
    reduce(&[0, 2, 4], &[0, 1, 0, 1], &[1.0, -3.0, -3.0, 1.0])
        .expect("clamped")
        .blocks()[0]
        .scaling_report(),
    1.0
)]
fn a_deficit_past_the_dominance_band_is_admitted_clamped_and_reported(
    #[case] report: Option<ScalingReport>,
    #[case] deficit: f64,
) {
    let report = report.expect("a clamp is reported");
    assert!(report.violation >= deficit, "{report:?}");
}

/// A 4-vertex star whose centre is `offset` ulps above its dyadic off-diagonal mass; the leaves balance exactly.
fn star_grounds_at_ulp_offset(offset: u64) -> bool {
    let centre = f64::from_bits(4e8f64.to_bits() + offset);
    let values = [centre, -1e8, -2e8, -1e8, -1e8, 1e8, -2e8, 2e8, -1e8, 1e8];
    let sgdd = reduce(&[0, 4, 6, 8, 10], &[0, 1, 2, 3, 0, 1, 0, 2, 0, 3], &values).expect("star");
    grounded(&sgdd)
}

/// Brackets the floor at a large row scale, where an absolute threshold misjudges both ends.
#[test]
fn surplus_below_the_row_noise_floor_does_not_augment() {
    assert!(!star_grounds_at_ulp_offset(4), "4 ulps is inside the floor");
    assert!(star_grounds_at_ulp_offset(24), "24 ulps is real dominance");
}

const OFF_BY_FIVE_ULPS: f64 = 1.0 + 5.0 * f64::EPSILON;

/// Charging one mirror to both rows would ground the other row on the tolerated difference.
#[rstest]
#[case::upper_holds_the_smaller([1.0, -1.0, -OFF_BY_FIVE_ULPS, OFF_BY_FIVE_ULPS])]
#[case::lower_holds_the_smaller([OFF_BY_FIVE_ULPS, -OFF_BY_FIVE_ULPS, -1.0, 1.0])]
fn tolerated_mirror_difference_is_not_one_row_s_surplus(#[case] values: [f64; 4]) {
    let sgdd = reduce(&[0, 2, 4], &[0, 1, 0, 1], &values).expect("tolerated mirrors");
    assert!(!grounded(&sgdd), "every stored row sums to zero");
}

/// Ten sub-ulp duplicates are absorbed one way and accumulate the other: no surplus on a row summing to zero.
#[test]
fn coalescing_additions_are_inside_the_error_allowance() {
    let half = f64::EPSILON / 2.0;
    let (mut row_ptrs, mut columns, mut values) = (vec![0u32], Vec::new(), Vec::new());
    for row in 0..2u32 {
        columns.extend(std::iter::repeat_n(row, 10));
        values.extend(std::iter::repeat_n(half, 10));
        columns.extend([row, 1 - row]);
        values.extend([1.0, -1.0]);
        columns.extend(std::iter::repeat_n(1 - row, 10));
        values.extend(std::iter::repeat_n(-half, 10));
        row_ptrs.push(columns.len() as u32);
    }
    let sgdd = reduce(&row_ptrs, &columns, &values).expect("coalesced duplicates");
    assert!(
        !grounded(&sgdd),
        "duplicates coalescing to a balanced Laplacian"
    );
}

/// Both ends of the row-scale range, where an absolute floor fails at one end or the other.
#[rstest]
#[case::at_1e12_scale(
    &[0, 2, 4],
    &[0, 1, 0, 1],
    &[1e12 + 100.0, -1e12, -1e12, 1e12 + 100.0],
    [1.0, 1.0],
    [0.01, 0.01]
)]
#[case::at_5e_11_scale(&[0, 1, 2], &[0, 1], &[5e-11, 5e-11], [1.0, 2.0], [1.0 / 5e-11, 2.0 / 5e-11])]
// 1e9 times the error the row's own additions could carry.
#[case::surplus_far_below_any_absolute_floor(
    &[0, 2, 4],
    &[0, 1, 0, 1],
    &[1e-6 + 6e-15, -1e-6, -1e-6, 1e-6 + 6e-15],
    [1.0, 1.0],
    [1.0 / 6e-15, 1.0 / 6e-15]
)]
fn genuine_surplus_at_either_scale_is_augmented_and_solves(
    #[case] row_ptrs: &[u32],
    #[case] columns: &[u32],
    #[case] values: &[f64],
    #[case] rhs: [f64; 2],
    #[case] expected: [f64; 2],
) {
    let sgdd = reduce(row_ptrs, columns, values).expect("genuine surplus");
    assert!(sgdd.blocks().iter().all(|block| !block.floats()));
    solves_to(&sgdd, &rhs, &expected);
}
