#[path = "common/factor.rs"]
mod factor;

use approx_chol::{Backend, Config, CsrRef, Sddm, SolveError};
use factor::factor;
use rstest::rstest;

/// Three grounded singletons; a right-hand side summing non-zero is what a floating gauge would lose.
fn diagonal_sddm() -> (Vec<u32>, Vec<u32>, Vec<f64>, u32) {
    (vec![0, 1, 2, 3], vec![0, 1, 2], vec![2.0, 3.0, 5.0], 3)
}

/// Each singleton is one edge to its ground, which stays a tree however AC2 splits it, so
/// the tight tolerance holds — the only closed-form check on the AC2 arithmetic.
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
    let factor = factor(
        Config {
            backend,
            split_merge,
            ..Config::default()
        },
        csr,
    )
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
    let factor = factor(Config::default(), csr).expect("factorization should succeed");
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
    let factor = factor(Config::default(), csr).expect("factorization should succeed");
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
    let factor = factor(
        Config {
            backend,
            ..Config::default()
        },
        CsrRef::new(&row_ptrs, &columns, &values, 3).expect("valid CSR"),
    )
    .expect("factorization should succeed");
    let solve = |dirt: f64| {
        let mut x = vec![1.0, -2.0, 0.5];
        let mut scratch = vec![dirt; factor.scratch_len()];
        factor.solve_in_place(&mut x, &mut scratch).expect("solve");
        x
    };
    assert_eq!(solve(3.0), solve(-7.0));
}
