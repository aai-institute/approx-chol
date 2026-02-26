use approx_chol::{Error, CsrRef};

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
    *rp.last_mut().unwrap() = 9;
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
fn new_rejects_out_of_bounds_column_index() {
    let (rp, mut ci, vals, n) = path_laplacian_4();
    // Replace last column index with n (out of bounds; valid range is 0..n-1)
    *ci.last_mut().unwrap() = n;
    let result = CsrRef::new(&rp, &ci, &vals, n);
    assert!(
        matches!(result, Err(Error::InvalidCsr(_))),
        "expected InvalidCsr, got {result:?}"
    );
}

// ---------------------------------------------------------------------------
// CsrRef::new_unchecked() — constructs without validation
// ---------------------------------------------------------------------------

#[test]
fn new_unchecked_constructs_without_validation() {
    let (rp, ci, vals, n) = path_laplacian_4();
    // This must not panic even if we skip validation
    let csr = CsrRef::new_unchecked(&rp, &ci, &vals, n);
    assert_eq!(csr.n(), n as usize);
}
