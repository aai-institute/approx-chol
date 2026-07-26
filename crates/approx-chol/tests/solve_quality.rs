#[path = "common/grid.rs"]
mod grid;
#[path = "common/panic_err.rs"]
mod panic_err;
#[path = "common/panic_ok.rs"]
mod panic_ok;
use grid::grid_laplacian;
use panic_err::ErrOrPanic;
use panic_ok::OrPanic;

use approx_chol::low_level::Builder;
use approx_chol::{Config, CsrRef, SolveError};
use num_traits::Float;

/// Row-sum drift at roundoff scale must not read as diagonal dominance. The
/// floor is precision-dependent, so `eps` comes from the caller.
fn assert_no_augmentation_at_surplus<T: Float + Send + Sync + 'static>(eps: T) {
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

#[test]
fn near_zero_surplus_does_not_augment() {
    assert_no_augmentation_at_surplus(5e-7_f32);
    assert_no_augmentation_at_surplus(5e-11_f64);
}

/// A surplus can be real and still be noise: this star's centre carries
/// `+1.92e-8` against a row scale of `8e8`, so `2.4e-17` relative — below `eps`,
/// hence not representable as dominance. It clears the absolute `sqrt(EPSILON)`
/// arm of the floor and must be caught by the row's noise floor instead, or the
/// factor gains a ground edge whose weight the input cannot actually express.
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

// ---------------------------------------------------------------------------
// SDDM solve accuracy (issue #35): solve(b) must equal M^-1 b
//
// A *diagonal* SDDM matrix augments to a star graph (aux vertex + leaves),
// which is a tree, so approximate Cholesky is EXACT here — no sampling error.
// That isolates the Gremban *recovery* (grounding) from AC approximation, so we
// can assert against the closed-form inverse x_i = b_i / d_i with tight tol.
// The RHS deliberately has a non-zero sum, which is exactly the case the old
// global zero-mean projection got wrong.
// ---------------------------------------------------------------------------

/// Diagonal SDDM matrix `diag(2, 3, 5)` as CSR (row sums 2, 3, 5 -> augmented).
fn diagonal_sddm() -> (Vec<u32>, Vec<u32>, Vec<f64>, u32) {
    (vec![0, 1, 2, 3], vec![0, 1, 2], vec![2.0, 3.0, 5.0], 3)
}

#[test]
fn sddm_solve_matches_dense_inverse_nonzero_sum_rhs() {
    let (rp, ci, vals, n) = diagonal_sddm();
    let csr = CsrRef::new(&rp, &ci, &vals, n).or_panic("valid diagonal SDDM");
    let factor = Builder::new(Config::default())
        .build(csr)
        .or_panic("factorization should succeed");

    assert!(factor.n() > n as usize, "diagonal SDDM should be augmented");

    // Both a strictly-positive and a mixed-sign RHS, each with a non-zero sum —
    // the case the old global zero-mean recovery corrupted. For diagonal M the
    // closed-form solution is x_i = b_i / d_i.
    for b in [[1.0_f64, 2.0, 3.0], [1.0, -2.0, 4.0]] {
        let x = factor.solve(&b).or_panic("solve should succeed");
        assert_eq!(x.len(), n as usize);
        for i in 0..n as usize {
            let want = b[i] / vals[i];
            assert!(
                (x[i] - want).abs() < 1e-9,
                "b={b:?}: x[{i}] = {:.6}, expected {want:.6} (M^-1 b); \
                 solve returned a non-solution of the SDDM system",
                x[i]
            );
        }
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
    assert!(matches!(err, SolveError::RhsLengthExceedsFactor { .. }));
}

#[test]
#[should_panic(expected = "work buffer too small")]
fn short_work_buffer_panics() {
    let lap = grid_laplacian(4, 4);
    let factor = Builder::new(Config::default())
        .build(lap.as_csr().or_panic("grid_laplacian must build valid CSR"))
        .or_panic("factorization should succeed");

    factor.solve_in_place(&mut vec![0.0; factor.n() - 1]);
}

// A grounded block's anchored solve *is* the SDDM solution, so solve_in_place and
// solve_into must agree.
#[test]
fn grounded_raw_solve_matches_recovered_solve() {
    let row_ptrs = [0u32, 2, 4];
    let columns = [0u32, 1, 0, 1];
    let values = [2.0, -1.0, -1.0, 2.0];
    let factor = Builder::<f64>::new(Config::default())
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
    factor.solve_in_place(&mut raw);

    assert_eq!(raw[..2], recovered[..2]);
    assert_eq!(raw[n - 1], 0.0, "ground must be pinned");
}

// A floating block has no ground vertex to absorb the null-space component, so
// `solve_in_place` pins one variable and differs from `solve_into` by that
// constant. The grounded case above catches neither.
#[test]
fn floating_raw_solve_differs_from_recovered_by_one_constant() {
    let grid = grid_laplacian(5, 5);
    let csr = grid.as_csr().or_panic("valid CSR");
    let n = grid.n as usize;
    let mut rhs: Vec<f64> = (0..n).map(|i| i as f64 - 12.0).collect();
    let sum: f64 = rhs.iter().sum();
    rhs[0] -= sum;

    let factor = Builder::<f64>::new(Config {
        seed: 7,
        ..Config::default()
    })
    .build(csr)
    .or_panic("factorization should succeed");
    assert_eq!(factor.n(), n, "pure Laplacian must not be augmented");

    let mut raw = rhs.clone();
    factor.solve_in_place(&mut raw);
    let mut recovered = vec![0.0; factor.n()];
    factor
        .solve_into(&rhs, &mut recovered)
        .or_panic("solve_into");

    assert!(
        raw.iter().any(|value| value.abs() < 1e-12),
        "one variable per block is pinned to zero"
    );

    // Same factor, so this is exact up to rounding.
    let shift = raw[0] - recovered[0];
    for (index, (&value, &canonical)) in raw.iter().zip(recovered.iter()).enumerate() {
        assert!(
            (value - canonical - shift).abs() < 1e-9,
            "raw must differ from recovered by one constant; \
             index {index} differs by {}",
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
