#[path = "common/grid.rs"]
mod grid;
#[path = "common/panic_err.rs"]
mod panic_err;
#[path = "common/panic_ok.rs"]
mod panic_ok;
#[path = "common/residual.rs"]
mod residual;
use grid::grid_laplacian;
use panic_err::ErrOrPanic;
use panic_ok::OrPanic;
use residual::relative_residual_over;

use approx_chol::low_level::Builder;
use approx_chol::{Backend, Config, CsrRef, SolveError};
use num_traits::Float;
use rstest::rstest;

/// The floor is precision-dependent, so `eps` comes from the caller.
fn assert_no_augmentation_at_drift<T: Float + Send + Sync + 'static>(eps: T) {
    let one = T::one();
    let row_ptrs = [0u32, 2, 4];
    let col_indices = [0u32, 1, 0, 1];
    let values = [one + eps, -one, -one, one + eps];
    let csr = CsrRef::new(&row_ptrs, &col_indices, &values, 2).or_panic("valid csr");
    let factor = Builder::<T>::new(Config::default())
        .build(csr)
        .or_panic("factorization should succeed");
    assert_eq!(
        factor.n(),
        2,
        "roundoff drift must not trigger augmentation"
    );
}

/// Augmentation is decided in ingestion, before routing, so the default suffices.
#[test]
fn near_zero_surplus_does_not_augment() {
    assert_no_augmentation_at_drift(5e-7_f32);
    assert_no_augmentation_at_drift(5e-11_f64);
}

/// A deficit inside the row's slack is not a dominance error either.
#[test]
fn near_zero_deficit_does_not_augment() {
    assert_no_augmentation_at_drift(-5e-7_f32);
    assert_no_augmentation_at_drift(-5e-11_f64);
}

/// `+1.92e-8` against a row scale of `8e8` is `2.4e-17` relative — below `eps`, so it
/// clears the `sqrt(EPSILON)` arm and only the noise floor can catch it.
#[test]
fn surplus_below_the_row_noise_floor_does_not_augment() {
    // Centre diagonal is the nearest double to 1e8 + 3e8 + 1e-7; each leaf balances
    // exactly, so only the centre row is in question.
    let row_ptrs = [0u32, 4, 6, 8, 10];
    let col_indices = [0u32, 1, 2, 3, 0, 1, 0, 2, 0, 3];
    let values = [
        400_000_000.000_000_1_f64,
        -1e8,
        -3e8,
        -1e-7,
        -1e8,
        1e8,
        -3e8,
        3e8,
        -1e-7,
        1e-7,
    ];
    let csr = CsrRef::new(&row_ptrs, &col_indices, &values, 4).or_panic("valid csr");
    let factor = Builder::<f64>::new(Config::default())
        .build(csr)
        .or_panic("factorization should succeed");

    assert_eq!(
        factor.n(),
        factor.original_n(),
        "surplus below the row's noise floor must not add a ground vertex"
    );
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
    let csr = CsrRef::new(&rp, &ci, &vals, n).or_panic("valid diagonal SDDM");
    let factor = Builder::new(Config {
        backend,
        split_merge,
        ..Config::default()
    })
    .build(csr)
    .or_panic("factorization should succeed");

    assert!(factor.n() > n as usize, "diagonal SDDM should be augmented");

    let x = factor.solve(&b).or_panic("solve should succeed");
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
fn solve_into_rejects_rhs_longer_than_original_for_augmented_factor() {
    // For an augmented SDDM factor n() == original_n() + 1, and the aux slot is
    // internal scratch. A RHS of length original_n + 1 must be rejected, not
    // silently accepted with its last entry overwritten by the grounding setup.
    let (rp, ci, vals, n) = diagonal_sddm();
    let csr = CsrRef::new(&rp, &ci, &vals, n).or_panic("valid diagonal SDDM");
    let factor = Builder::new(Config::default())
        .build(csr)
        .or_panic("factorization should succeed");
    assert_eq!(factor.n(), factor.original_n() + 1, "SDDM augments by one");

    let rhs = vec![0.0; factor.original_n() + 1]; // == factor.n(): the aux slot
    let mut work = vec![0.0; factor.n()];
    let err = factor
        .solve_into(&rhs, &mut work)
        .err_or_panic("rhs longer than original dimension must fail");
    assert!(
        matches!(err, SolveError::RhsLengthExceedsFactor { .. }),
        "{err:?}"
    );
}

/// Both entry points size their buffer through the same check.
#[test]
fn every_solve_entry_point_reports_a_short_work_buffer() {
    let lap = grid_laplacian(4, 4);
    let n_orig = lap.n as usize;
    let factor = Builder::new(Config::default())
        .build(lap.as_csr().or_panic("grid_laplacian must build valid CSR"))
        .or_panic("factorization should succeed");

    let mut rhs = vec![0.0; n_orig];
    rhs[0] = 1.0;
    rhs[n_orig - 1] = -1.0;
    let mut work = vec![0.0; factor.n().saturating_sub(1)];

    for err in [
        factor
            .solve_into(&rhs, &mut work)
            .err_or_panic("solve_into must reject a short work buffer"),
        factor
            .solve_in_place(&mut work)
            .err_or_panic("solve_in_place must reject a short work buffer"),
    ] {
        assert!(
            matches!(err, SolveError::WorkBufferTooSmall { .. }),
            "{err:?}"
        );
    }
}

// A grounded block's anchored solve *is* the SDDM solution, so solve_in_place and
// solve_into must agree.
#[rstest]
#[case::approximate(Backend::Approximate)]
#[case::exact(Backend::default())]
fn grounded_raw_solve_matches_recovered_solve(#[case] backend: Backend) {
    let row_ptrs = [0u32, 2, 4];
    let columns = [0u32, 1, 0, 1];
    let values = [2.0, -1.0, -1.0, 2.0];
    let factor = Builder::<f64>::new(Config {
        backend,
        ..Config::default()
    })
    .build(CsrRef::new(&row_ptrs, &columns, &values, 2).or_panic("valid CSR"))
    .or_panic("factorization should succeed");

    let n = factor.n();
    assert_eq!(n, 3, "strictly dominant input must gain a ground vertex");

    let rhs = [1.0, -2.0];
    let mut recovered = vec![0.0; n];
    factor
        .solve_into(&rhs, &mut recovered)
        .or_panic("solve_into should succeed");

    let mut raw = vec![0.0; n];
    raw[..rhs.len()].copy_from_slice(&rhs);
    factor
        .solve_in_place(&mut raw)
        .or_panic("solve_in_place should succeed");

    assert_eq!(raw[..2], recovered[..2]);
    assert_eq!(raw[n - 1], 0.0, "ground must be pinned");
}

// A floating block has no ground vertex to absorb the null-space component, so
// `solve_in_place` pins one variable and differs from `solve_into` by that
// constant. The grounded case above catches neither.
#[rstest]
#[case::approximate(Backend::Approximate)]
#[case::exact(Backend::default())]
fn floating_raw_solve_differs_from_recovered_by_one_constant(#[case] backend: Backend) {
    let grid = grid_laplacian(5, 5);
    let csr = grid.as_csr().or_panic("valid CSR");
    let n = grid.n as usize;
    let mut rhs: Vec<f64> = (0..n).map(|i| i as f64 - 12.0).collect();
    let sum: f64 = rhs.iter().sum();
    rhs[0] -= sum;

    let factor = Builder::<f64>::new(Config {
        seed: 7,
        backend,
        ..Config::default()
    })
    .build(csr)
    .or_panic("factorization should succeed");
    assert_eq!(factor.n(), n, "pure Laplacian must not be augmented");

    let mut raw = rhs.clone();
    factor
        .solve_in_place(&mut raw)
        .or_panic("solve_in_place should succeed");
    let mut recovered = vec![0.0; factor.n()];
    factor
        .solve_into(&rhs, &mut recovered)
        .or_panic("solve_into");

    // Both backends pin the block's last variable, so this index is not
    // backend-dependent the way the pinned *value* once was.
    assert_eq!(raw[n - 1], 0.0, "the block's last variable is pinned");

    // Same factor, so this is exact up to rounding.
    let shift = raw[0] - recovered[0];
    for (index, (&value, &canonical)) in raw.iter().zip(recovered.iter()).enumerate() {
        assert!(
            (value - canonical - shift).abs() < 1e-9,
            "raw must differ from recovered by one constant; index {index} differs by {}",
            value - canonical
        );
    }
    assert!(shift.abs() > 1e-9, "the constant must be non-zero");

    let mean = recovered.iter().sum::<f64>() / n as f64;
    assert!(
        mean.abs() < 1e-9,
        "recovered solve must be the zero-mean representative"
    );
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
        .or_panic("scaled path is valid CSR");
    let one = T::one();
    let b = [one, T::zero(), T::zero(), -one];

    let factor = Builder::<T>::new(Config {
        backend,
        ..Config::default()
    })
    .build(csr)
    .or_panic("factorization should succeed");
    let x = factor.solve(&b).or_panic("solve");

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
