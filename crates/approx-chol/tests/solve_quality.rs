#[path = "common/grid.rs"]
mod grid;
#[path = "common/residual.rs"]
mod residual;
use grid::grid_laplacian;
use residual::relative_residual_over;

use approx_chol::low_level::Builder;
use approx_chol::{Backend, Config, CsrRef, Error, Sddm, SolveError};
use num_traits::Float;
use rstest::rstest;

/// Routes `[[1+drift, -1], [-1, 1+drift]]`: `Ok(true)` when the drift is real dominance
/// and earns a ground vertex, `Ok(false)` when it is within the row's own summation
/// error, `Err` when it is a real deficit. Row scale is 2 over 2 stored terms, so the
/// floor is `4 * eps` either way.
fn route_at_drift<T: Float + Send + Sync + 'static>(drift: T) -> Result<bool, Error> {
    let one = T::one();
    let row_ptrs = [0u32, 2, 4];
    let col_indices = [0u32, 1, 0, 1];
    let values = [one + drift, -one, -one, one + drift];
    let csr = CsrRef::new(&row_ptrs, &col_indices, &values, 2).expect("valid csr");
    Sddm::<T>::try_from(csr).map(|sddm| matches!(sddm, Sddm::Grounded(_)))
}

/// One ULP either way is drift a single addition accounts for, so the row is left floating.
#[test]
fn summation_roundoff_does_not_augment() {
    for drift in [f32::EPSILON, -f32::EPSILON] {
        assert!(!route_at_drift(drift).expect("f32 roundoff is in class"));
    }
    for drift in [f64::EPSILON, -f64::EPSILON] {
        assert!(!route_at_drift(drift).expect("f64 roundoff is in class"));
    }
}

/// Past that the surplus is real dominance (#85). Both drifts land mid-window for their
/// precision — 8 ULPs of this row in `f32`, 2.25e5 in `f64` — which no two-term sum
/// invents, yet through 0.3.1 both were answered as a singular Laplacian.
#[test]
fn surplus_beyond_summation_roundoff_augments() {
    assert!(route_at_drift(1e-6_f32).expect("f32 dominance"));
    assert!(route_at_drift(5e-11_f64).expect("f64 dominance"));
}

/// The same magnitude with the sign flipped is real *non*-dominance, and is reported
/// rather than zeroed (#91). Through 0.3.1 a coarser deficit tolerance forgave these,
/// which left a matrix drifting both ways factored partially grounded.
#[test]
fn deficit_beyond_summation_roundoff_is_rejected() {
    for err in [
        route_at_drift(-1e-6_f32).expect_err("f32 deficit must be reported"),
        route_at_drift(-5e-11_f64).expect_err("f64 deficit must be reported"),
    ] {
        assert!(
            matches!(err, Error::NotDiagonallyDominant { row: 0 }),
            "{err:?}"
        );
    }
}

/// A 4-vertex star whose centre diagonal is `offset` ULPs above its exactly-dyadic
/// off-diagonal mass, so the drift is the offset and nothing else. Leaves balance
/// exactly, leaving only the centre row in question.
fn star_augments_at_ulp_offset(offset: u64) -> bool {
    let centre = f64::from_bits(4e8f64.to_bits() + offset);
    let row_ptrs = [0u32, 4, 6, 8, 10];
    let col_indices = [0u32, 1, 2, 3, 0, 1, 0, 2, 0, 3];
    let values = [centre, -1e8, -2e8, -1e8, -1e8, 1e8, -2e8, 2e8, -1e8, 1e8];
    let csr = CsrRef::new(&row_ptrs, &col_indices, &values, 4).expect("valid csr");
    matches!(Sddm::try_from(csr), Ok(Sddm::Grounded(_)))
}

/// Brackets the floor at a large row scale, where an absolute threshold would misjudge
/// both ends. Scale is `8e8` and the row has 4 stored terms, so the floor is
/// `eps * 8e8 * 4 = 7.1e-7`, or 11.9 ULPs of the centre diagonal.
#[test]
fn surplus_below_the_row_noise_floor_does_not_augment() {
    assert!(!star_augments_at_ulp_offset(4), "4 ULP is inside the floor");
    assert!(star_augments_at_ulp_offset(24), "24 ULP is real dominance");
}

// A diagonal SDDM matrix augments to a star, which is a tree, so AC is exact here —
// that isolates Gremban recovery from sampling error. The RHS sums non-zero, the case
// the old global zero-mean projection got wrong.

fn diagonal_sddm() -> (Vec<u32>, Vec<u32>, Vec<f64>, u32) {
    (vec![0, 1, 2, 3], vec![0, 1, 2], vec![2.0, 3.0, 5.0], 3)
}

/// A star is a tree at every `k`, so no clique edge is sampled and the tight
/// tolerance holds — the only closed-form check on the AC2 arithmetic.
#[rstest]
#[case::approximate(Backend::Approximate)]
#[case::exact(Backend::default())]
fn sddm_solve_matches_dense_inverse_nonzero_sum_rhs(
    #[case] backend: Backend,
    #[values(None, Some(2), Some(3), Some(7))] split_merge: Option<u32>,
    #[values([1.0, 2.0, 3.0], [1.0, -2.0, 4.0])] b: [f64; 3],
) {
    let (rp, ci, vals, n) = diagonal_sddm();
    let csr = CsrRef::new(&rp, &ci, &vals, n).expect("valid diagonal SDDM");
    assert!(
        matches!(Sddm::try_from(csr), Ok(Sddm::Grounded(_))),
        "diagonal SDDM should be grounded"
    );
    let factor = Builder::new(Config {
        backend,
        split_merge,
        ..Config::default()
    })
    .build(csr)
    .expect("factorization should succeed");

    let x = factor.solve(&b).expect("solve should succeed");
    assert_eq!(x.len(), n as usize);
    for i in 0..n as usize {
        let want = b[i] / vals[i];
        assert!(
            (x[i] - want).abs() < 1e-9,
            "x[{i}] = {:.6}, expected {want:.6} (M^-1 b)",
            x[i]
        );
    }
}

#[test]
fn solve_in_place_rejects_a_length_other_than_n() {
    let (rp, ci, vals, n) = diagonal_sddm();
    let csr = CsrRef::new(&rp, &ci, &vals, n).expect("valid diagonal SDDM");
    let factor = Builder::new(Config::default())
        .build(csr)
        .expect("factorization should succeed");
    let mut scratch = vec![0.0; factor.scratch_len()];
    for len in [factor.n() - 1, factor.n() + 1] {
        let err = factor
            .solve_in_place(&mut vec![0.0; len], &mut scratch)
            .expect_err("only n() entries are a solution");
        assert!(matches!(err, SolveError::LengthMismatch { .. }), "{err:?}");
    }
}

#[test]
fn solve_in_place_rejects_short_scratch() {
    let (rp, ci, vals, n) = diagonal_sddm();
    let csr = CsrRef::new(&rp, &ci, &vals, n).expect("valid diagonal SDDM");
    let factor = Builder::new(Config::default())
        .build(csr)
        .expect("factorization should succeed");
    assert!(factor.scratch_len() > 0, "a grounded factor needs scratch");
    let mut x = vec![1.0; factor.n()];
    let err = factor
        .solve_in_place(&mut x, &mut vec![0.0; factor.scratch_len() - 1])
        .expect_err("short scratch must fail");
    assert!(matches!(err, SolveError::ScratchTooSmall { .. }), "{err:?}");
}

/// Scratch is a buffer, not an input: whatever it held, the solution is the same.
#[rstest]
#[case::approximate(Backend::Approximate)]
#[case::exact(Backend::default())]
fn dirty_scratch_does_not_change_the_solution(#[case] backend: Backend) {
    let row_ptrs = [0u32, 2, 4, 5];
    let columns = [0u32, 1, 0, 1, 2];
    let values = [2.0, -1.0, -1.0, 2.0, 1.0];
    let factor = Builder::<f64>::new(Config {
        backend,
        ..Config::default()
    })
    .build(CsrRef::new(&row_ptrs, &columns, &values, 3).expect("valid CSR"))
    .expect("factorization should succeed");
    let solve = |dirt: f64| {
        let mut x = vec![1.0, -2.0, 0.5];
        let mut scratch = vec![dirt; factor.scratch_len()];
        factor.solve_in_place(&mut x, &mut scratch).expect("solve");
        x
    };
    assert_eq!(solve(3.0), solve(-7.0));
}

/// Scaling a Laplacian by `t` scales its solution by `1/t`, so a factor that drops
/// the scale is wrong by that whole factor rather than slightly less accurate.
fn assert_scaled_path_solves<T>(exponent: i32, backend: Backend)
where
    T: Float + Send + Sync + 'static + std::fmt::LowerExp,
{
    let ten = T::from(10.0).expect("10 is representable");
    let scale = ten.powi(exponent);
    let mut lap = grid_laplacian(1, 4);
    let values: Vec<T> = lap
        .values
        .drain(..)
        .map(|value| T::from(value).expect("fixture weight is representable") * scale)
        .collect();
    let csr = CsrRef::new(&lap.row_ptrs, &lap.col_indices, &values, lap.n)
        .expect("scaled path is valid CSR");
    let one = T::one();
    let b = [one, T::zero(), T::zero(), -one];

    let factor = Builder::<T>::new(Config {
        backend,
        ..Config::default()
    })
    .build(csr)
    .expect("factorization should succeed");
    let x = factor.solve(&b).expect("solve");

    let relative = relative_residual_over(csr, &x, &b, 0..b.len());
    assert!(
        relative < T::from(1e-6).expect("tolerance is representable"),
        "relative residual {relative:e}"
    );
}

/// The two scalars bottom out at different exponents, so each gets the range its
/// solution is still representable in.
#[rstest]
#[case::approximate(Backend::Approximate)]
#[case::exact(Backend::default())]
fn a_scaled_f64_laplacian_solves_wherever_its_solution_is_representable(
    #[case] backend: Backend,
    #[values(0, -5, -15, -100, -250)] exponent: i32,
) {
    assert_scaled_path_solves::<f64>(exponent, backend);
}

#[rstest]
#[case::approximate(Backend::Approximate)]
#[case::exact(Backend::default())]
fn a_scaled_f32_laplacian_solves_wherever_its_solution_is_representable(
    #[case] backend: Backend,
    #[values(0, -3, -10, -25)] exponent: i32,
) {
    assert_scaled_path_solves::<f32>(exponent, backend);
}
