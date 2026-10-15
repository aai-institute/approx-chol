use approx_chol_next::{factorize, Config, Real, Sgdd};

use super::{grounded, reduce};

/// `[[a+s, -a], [-a, a+s]]`: dominant by `s` on both rows, so it floats exactly when `s` is noise.
fn surplus_pair<T: Real>(a: T, s: T) -> Sgdd<T> {
    reduce(&[0, 2, 4], &[0, 1, 0, 1], &[a + s, -a, -a, a + s]).expect("a pair is admitted")
}

fn solves_to<T: Real + std::fmt::LowerExp>(label: &str, sgdd: &Sgdd<T>, rhs: &[T], expected: &[T]) {
    let solution = factorize(sgdd, &Config::default())
        .solve(rhs)
        .expect("solve");
    let tolerance = T::from(1e-3).unwrap();
    for (&got, &want) in solution.iter().zip(expected) {
        assert!(
            (got - want).abs() <= tolerance * want.abs(),
            "{label}: {got:e} vs {want:e}"
        );
    }
}

/// Pins the routing, not the solution: at a `2e-6` row scale summation error is about `1e-21`.
#[test]
fn surplus_is_judged_against_summation_error_alone() {
    let cases: [(&str, f64, bool); 5] = [
        ("far above the floor", 6e-12, true),
        ("above the floor", 6e-15, true),
        // A floor 1e6 coarser than rounding once answered this as a singular Laplacian.
        ("just above the floor", 6e-18, true),
        ("below the floor", 6e-22, false),
        ("not representable at this scale", 1e-30, false),
    ];
    for (label, surplus, expected) in cases {
        let sgdd = surplus_pair(1e-6, surplus);
        assert_eq!(grounded(&sgdd), expected, "{label}: surplus {surplus:e}");
        if expected {
            solves_to(label, &sgdd, &[1.0, 1.0], &[1.0 / surplus; 2]);
        }
    }
}

/// The same window in `f32` at a `2e-3` row scale (#85): summation error is about `5e-10`.
#[test]
fn f32_surplus_is_judged_against_summation_error_alone() {
    for (label, surplus, expected) in [
        ("above the floor", 1.2e-9f32, true),
        ("below", 2e-10, false),
    ] {
        let sgdd = surplus_pair(1e-3f32, surplus);
        assert_eq!(grounded(&sgdd), expected, "{label}: surplus {surplus:e}");
    }
}

fn drifted<T: Real>(drift: T) -> Sgdd<T> {
    surplus_pair(T::one(), drift)
}

/// One ulp either way is drift a single addition accounts for, so the pair floats.
#[test]
fn summation_roundoff_does_not_augment() {
    for drift in [f32::EPSILON, -f32::EPSILON] {
        assert!(!grounded(&drifted(drift)), "f32 drift {drift:e}");
    }
    for drift in [f64::EPSILON, -f64::EPSILON] {
        assert!(!grounded(&drifted(drift)), "f64 drift {drift:e}");
    }
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
    assert!(reported(&drifted(-20.0 * f64::EPSILON)));
    assert!(!reported(&drifted(-12.0 * f64::EPSILON)), "inside 16 eps");
    let f32_reported = |drift: f32| drifted(drift).blocks()[0].scaling_report().is_some();
    assert!(f32_reported(-20.0 * f32::EPSILON));
    assert!(!f32_reported(-12.0 * f32::EPSILON), "inside 16 eps");
}

/// A row of `k` entries rounds by `(k + 3)ε`: a degree-100 centre short by 50ε is rounding, by 200ε a deficit.
#[test]
fn a_rows_rounding_band_grows_with_its_degree() {
    let star = |short: f64| {
        let leaves = 100u32;
        let mut row_ptrs = vec![0u32, leaves + 1];
        let mut columns: Vec<u32> = (0..=leaves).collect();
        let mut values = vec![f64::from(leaves) * (1.0 - short)];
        values.extend(std::iter::repeat_n(-1.0, leaves as usize));
        for leaf in 1..=leaves {
            columns.extend([0, leaf]);
            values.extend([-1.0, 1.0]);
            row_ptrs.push(columns.len() as u32);
        }
        reduce(&row_ptrs, &columns, &values).expect("a star is admitted")
    };
    assert!(star(50.0 * f64::EPSILON).blocks()[0]
        .scaling_report()
        .is_none());
    assert!(star(200.0 * f64::EPSILON).blocks()[0]
        .scaling_report()
        .is_some());
}

/// Main rejected these as not dominant; the rewrite admits them clamped and reports the violation.
#[test]
fn a_deficit_past_the_dominance_band_is_admitted_clamped_and_reported() {
    let violations = [
        (
            "f64 pair 2e-9 short",
            drifted(-2e-9_f64).blocks()[0].scaling_report(),
            2e-9,
        ),
        (
            "f32 pair 1e-5 short",
            drifted(-1e-5_f32).blocks()[0].scaling_report(),
            1e-5,
        ),
        (
            "no scaling makes it dominant",
            reduce(&[0, 2, 4], &[0, 1, 0, 1], &[1.0, -3.0, -3.0, 1.0])
                .expect("clamped")
                .blocks()[0]
                .scaling_report(),
            1.0,
        ),
    ];
    for (label, report, deficit) in violations {
        let report = report.unwrap_or_else(|| panic!("{label}: a clamp is reported"));
        assert!(report.violation >= deficit, "{label}: {report:?}");
    }
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

/// Charging one mirror to both rows would ground the other row on the tolerated difference.
#[test]
fn tolerated_mirror_difference_is_not_one_row_s_surplus() {
    let off = 1.0 + 5.0 * f64::EPSILON;
    let cases = [
        ("upper holds the smaller", [1.0, -1.0, -off, off]),
        ("lower holds the smaller", [off, -off, -1.0, 1.0]),
    ];
    for (label, values) in cases {
        let sgdd = reduce(&[0, 2, 4], &[0, 1, 0, 1], &values).expect(label);
        assert!(!grounded(&sgdd), "{label}: every stored row sums to zero");
    }
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
#[test]
fn genuine_surplus_at_either_scale_is_augmented_and_solves() {
    let cases = [
        (
            "1e12 scale",
            reduce(
                &[0, 2, 4],
                &[0, 1, 0, 1],
                &[1e12 + 100.0, -1e12, -1e12, 1e12 + 100.0],
            ),
            [1.0, 1.0],
            [0.01, 0.01],
        ),
        (
            "5e-11 scale",
            reduce(&[0, 1, 2], &[0, 1], &[5e-11, 5e-11]),
            [1.0, 2.0],
            [1.0 / 5e-11, 2.0 / 5e-11],
        ),
        // 1e9 times the error the row's own additions could carry.
        (
            "surplus far below any absolute floor",
            reduce(
                &[0, 2, 4],
                &[0, 1, 0, 1],
                &[1e-6 + 6e-15, -1e-6, -1e-6, 1e-6 + 6e-15],
            ),
            [1.0, 1.0],
            [1.0 / 6e-15, 1.0 / 6e-15],
        ),
    ];
    for (label, sgdd, rhs, expected) in cases {
        let sgdd = sgdd.expect(label);
        assert!(sgdd.blocks().iter().all(|block| !block.floats()), "{label}");
        solves_to(label, &sgdd, &rhs, &expected);
    }
}
