#[path = "common/grounded.rs"]
mod grounded;
#[path = "common/path.rs"]
mod path;
#[path = "common/residual.rs"]
mod residual;
use grounded::is_grounded;
use residual::relative_residual_over;

use approx_chol::{factorize, factorize_with, Backend, Config, CsrRef, Error, Sddm, SolveError};
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
    Sddm::try_from(csr).map(|sddm| is_grounded(&factorize(sddm)))
}

/// Augmentation is decided in ingestion, before routing, so the default suffices. One
/// ULP either way is drift a single addition accounts for, so the row is left floating.
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
    let factor = factorize_with(Sddm::try_from(csr).expect("an SDDM"), Config::default())
        .expect("factorization should succeed");
    is_grounded(&factor)
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
    let factor = factorize_with(
        Sddm::try_from(csr).expect("an SDDM"),
        Config {
            backend,
            split_merge,
            ..Config::default()
        },
    )
    .expect("factorization should succeed");

    assert!(is_grounded(&factor), "diagonal SDDM should be augmented");

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

/// The ground slot is internal, so `n + 1` is too long even though scratch holds `n + 1` slots.
#[test]
fn every_solve_checks_its_buffer_lengths() {
    let (rp, ci, vals, n) = diagonal_sddm();
    let csr = CsrRef::new(&rp, &ci, &vals, n).expect("valid diagonal SDDM");
    let factor = factorize_with(Sddm::try_from(csr).expect("an SDDM"), Config::default())
        .expect("factorization should succeed");
    let needed = factor.scratch_len();
    assert!(
        needed > factor.n(),
        "a grounded factor solves through scratch"
    );

    let mut scratch = vec![0.0; needed];
    for len in [factor.n() - 1, factor.n() + 1] {
        let mut x = vec![0.0; len];
        for err in [
            factor.solve(&x).expect_err("solve must reject the length"),
            factor
                .solve_in_place(&mut x, &mut scratch)
                .expect_err("solve_in_place must reject the length"),
        ] {
            assert_eq!(
                err,
                SolveError::LengthMismatch {
                    len,
                    factor_dim: factor.n()
                }
            );
        }
    }

    let mut x = vec![1.0; factor.n()];
    assert_eq!(
        factor.solve_in_place(&mut x, &mut scratch[..needed - 1]),
        Err(SolveError::ScratchTooSmall {
            scratch_len: needed - 1,
            needed
        })
    );
}

/// NaN reaches the result if any slot, a ground included, is read before it is written.
#[rstest]
#[case::grounded(&[0, 2, 4], &[0, 1, 0, 1], &[2.0, -1.0, -1.0, 2.0])]
#[case::interleaved_components(
    &[0, 2, 4, 6, 8],
    &[0, 2, 1, 3, 0, 2, 1, 3],
    &[1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0]
)]
fn dirty_scratch_does_not_change_the_solution(
    #[case] row_ptrs: &[u32],
    #[case] columns: &[u32],
    #[case] values: &[f64],
    #[values(Backend::Approximate, Backend::default())] backend: Backend,
) {
    let n = row_ptrs.len() - 1;
    let csr = CsrRef::new(row_ptrs, columns, values, n as u32).expect("valid CSR");
    let factor = factorize_with(
        Sddm::try_from(csr).expect("an SDDM"),
        Config {
            seed: 7,
            backend,
            ..Config::default()
        },
    )
    .expect("factorization should succeed");
    assert!(factor.scratch_len() > 0, "the solve goes through scratch");

    let rhs: Vec<f64> = (0..n).map(|i| i as f64 - 1.5).collect();
    let mut in_place = rhs.clone();
    let mut scratch = vec![f64::NAN; factor.scratch_len() + 1];
    factor
        .solve_in_place(&mut in_place, &mut scratch)
        .expect("solve_in_place should succeed");
    assert_eq!(in_place, factor.solve(&rhs).expect("solve"));
}

/// Scaling a Laplacian by `t` scales its solution by `1/t`, so a factor that drops
/// the scale is wrong by that whole factor rather than slightly less accurate.
fn assert_scaled_path_solves<T>(exponent: i32, backend: Backend)
where
    T: Float + Send + Sync + 'static + std::fmt::LowerExp,
{
    let ten = T::from(10.0).expect("10 is representable");
    let scale = ten.powi(exponent);
    let values: Vec<T> = path::VALUES
        .iter()
        .map(|&value| T::from(value).expect("fixture weight is representable") * scale)
        .collect();
    let csr = CsrRef::new(&path::ROW_PTRS, &path::COL_INDICES, &values, path::N)
        .expect("scaled path is valid CSR");
    let one = T::one();
    let b = [one, T::zero(), T::zero(), -one];

    let factor = factorize_with(
        Sddm::try_from(csr).expect("an SDDM"),
        Config {
            backend,
            ..Config::default()
        },
    )
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
