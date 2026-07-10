#[path = "common/panic_ok.rs"]
mod panic_ok;
use panic_ok::OrPanic;

use approx_chol::low_level::Builder;
use approx_chol::{Config, CsrRef, Error};

fn solve_with_default_ac(
    row_ptrs: &[u32],
    col_indices: &[u32],
    values: &[f64],
    n: u32,
    rhs: &[f64],
) -> Vec<f64> {
    let csr = CsrRef::new(row_ptrs, col_indices, values, n).or_panic("valid CSR");
    let factor = Builder::<f64>::new(Config {
        seed: 7,
        ..Config::default()
    })
    .build(csr)
    .or_panic("factorization");
    let mut work = vec![0.0; factor.n()];
    work[..rhs.len()].copy_from_slice(rhs);
    factor
        .solve_in_place(&mut work)
        .or_panic("solve_in_place should succeed");
    work
}

#[test]
fn duplicate_diagonal_entries_do_not_change_solve_behavior() {
    // Two mathematically equivalent 2x2 SDDM matrices:
    // - `dup_*`: diagonal split into duplicate entries in each row
    // - `coal_*`: diagonal already coalesced
    //
    // A = [ 5  -1 ]
    //     [ -1  4 ]
    let dup_rp = vec![0u32, 3, 6];
    let dup_ci = vec![0u32, 0, 1, 0, 1, 1];
    let dup_vals = vec![2.0f64, 3.0, -1.0, -1.0, 1.5, 2.5];

    let coal_rp = vec![0u32, 2, 4];
    let coal_ci = vec![0u32, 1, 0, 1];
    let coal_vals = vec![5.0f64, -1.0, -1.0, 4.0];

    let rhs = vec![1.0f64, -1.0];

    let x_dup = solve_with_default_ac(&dup_rp, &dup_ci, &dup_vals, 2, &rhs);
    let x_coal = solve_with_default_ac(&coal_rp, &coal_ci, &coal_vals, 2, &rhs);

    assert_eq!(
        x_dup.len(),
        x_coal.len(),
        "equivalent inputs must produce factors with equal dimension"
    );
    for (i, (&a, &b)) in x_dup.iter().zip(x_coal.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-10,
            "equivalent matrices should solve identically; mismatch at {i}: {a} vs {b}"
        );
    }
}

#[test]
fn positive_off_diagonal_is_rejected_not_silently_dropped() {
    // A = [ 5  1 ]   the +1 off-diagonals are outside the SDDM/Laplacian class.
    //     [ 1  4 ]   Ingestion used to fall through both the diagonal and the
    // `val < 0` edge branch, silently dropping them and factorizing diag(5, 4)
    // — a confidently wrong factor. It now rejects the input instead.
    let rp = [0u32, 2, 4];
    let ci = [0u32, 1, 0, 1];
    let vals = [5.0f64, 1.0, 1.0, 4.0];

    // The AC (None) and AC2 (Some) paths share this ingestion; both must reject.
    for split_merge in [None, Some(2)] {
        let csr = CsrRef::new(&rp, &ci, &vals, 2).or_panic("valid CSR");
        let err = Builder::<f64>::new(Config {
            split_merge,
            ..Config::default()
        })
        .build(csr)
        .expect_err("positive off-diagonal must be rejected");
        assert!(
            matches!(err, Error::PositiveOffDiagonal { edge } if edge == (0, 1)),
            "expected PositiveOffDiagonal at (0, 1), got {err:?}"
        );
    }
}
