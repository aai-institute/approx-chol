#[path = "common/panic_ok.rs"]
mod panic_ok;
use panic_ok::OrPanic;

use approx_chol::{CsrRef, Error};

// ---------------------------------------------------------------------------
// Helpers — 4-node path Laplacian (valid CSR)
// ---------------------------------------------------------------------------

fn path_laplacian_4() -> (Vec<u32>, Vec<u32>, Vec<f64>, u32) {
    // 0 — 1 — 2 — 3
    let row_ptrs = vec![0u32, 2, 5, 8, 10];
    let col_indices = vec![0u32, 1, 0, 1, 2, 1, 2, 3, 2, 3];
    let values = vec![1.0f64, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0];
    let n = 4u32;
    (row_ptrs, col_indices, values, n)
}

// ---------------------------------------------------------------------------
// CsrRef::new() — valid input
// ---------------------------------------------------------------------------

#[test]
fn new_succeeds_on_valid_path_laplacian() {
    let (rp, ci, vals, n) = path_laplacian_4();
    let result = CsrRef::new(&rp, &ci, &vals, n);
    assert!(result.is_ok(), "expected Ok but got {result:?}");
}

// ---------------------------------------------------------------------------
// CsrRef::new() — error cases
// ---------------------------------------------------------------------------

#[test]
fn new_rejects_wrong_row_ptrs_length() {
    let (mut rp, ci, vals, n) = path_laplacian_4();
    rp.pop(); // length is now n instead of n+1
    let result = CsrRef::new(&rp, &ci, &vals, n);
    assert!(
        matches!(result, Err(Error::InvalidCsr(_))),
        "expected InvalidCsr, got {result:?}"
    );
}

#[test]
fn new_rejects_mismatched_col_indices_values_lengths() {
    let (rp, ci, mut vals, n) = path_laplacian_4();
    vals.pop(); // col_indices.len() != values.len()
    let result = CsrRef::new(&rp, &ci, &vals, n);
    assert!(
        matches!(result, Err(Error::InvalidCsr(_))),
        "expected InvalidCsr, got {result:?}"
    );
}

#[test]
fn new_rejects_row_ptrs_n_not_equal_nnz() {
    let (mut rp, ci, vals, n) = path_laplacian_4();
    // row_ptrs[n] should equal col_indices.len() (10); set it to 9
    *rp.last_mut().or_panic("fixture must be non-empty") = 9;
    let result = CsrRef::new(&rp, &ci, &vals, n);
    assert!(
        matches!(result, Err(Error::InvalidCsr(_))),
        "expected InvalidCsr, got {result:?}"
    );
}

#[test]
fn new_rejects_non_monotonic_row_ptrs() {
    let (mut rp, ci, vals, n) = path_laplacian_4();
    // Make row_ptrs[2] < row_ptrs[1] — non-monotonic
    rp[2] = 0;
    let result = CsrRef::new(&rp, &ci, &vals, n);
    assert!(
        matches!(result, Err(Error::InvalidCsr(_))),
        "expected InvalidCsr, got {result:?}"
    );
}

#[test]
fn new_rejects_non_zero_based_row_ptrs() {
    // This matrix is intentionally malformed: row_ptrs starts at 1, so the
    // first payload entry is not addressable by any row.
    //
    // If accepted, that leading entry is silently ignored and factorization
    // runs on a different matrix than the caller provided.
    let rp = vec![1u32, 3, 5];
    let ci = vec![0u32, 0, 1, 0, 1];
    let vals = vec![1234.0f64, 2.0, -1.0, -1.0, 2.0];
    let n = 2u32;

    match CsrRef::new(&rp, &ci, &vals, n) {
        Err(Error::InvalidCsr(_)) => {}
        Err(other) => panic!("expected InvalidCsr, got {other:?}"),
        Ok(csr) => {
            let (_, row0_vals) = csr.try_row(0).or_panic("row 0 should be valid");
            let silently_dropped = !row0_vals.iter().any(|&v| (v - 1234.0).abs() < 1e-12);
            assert!(
                silently_dropped,
                "expected malformed row_ptrs[0] to make leading payload unreachable"
            );
            panic!(
                "accepted malformed CSR (row_ptrs[0] != 0): leading payload was silently ignored"
            );
        }
    }
}

#[test]
fn new_rejects_out_of_bounds_column_index() {
    let (rp, mut ci, vals, n) = path_laplacian_4();
    // Replace last column index with n (out of bounds; valid range is 0..n-1)
    *ci.last_mut().or_panic("fixture must be non-empty") = n;
    let result = CsrRef::new(&rp, &ci, &vals, n);
    assert!(
        matches!(result, Err(Error::InvalidCsr(_))),
        "expected InvalidCsr, got {result:?}"
    );
}
