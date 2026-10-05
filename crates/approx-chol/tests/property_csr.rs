#[path = "common/laplacian_prop.rs"]
mod laplacian_prop;

use approx_chol::{CsrError, CsrRef, Laplacian, Sddm};
use laplacian_prop::{laplacian_csr_strategy, one_grounded_component_strategy};
use proptest::prelude::*;

/// The CSR path proves edges by canonical order, the typed constructors by checking them;
/// whatever the first accepts, the second must accept unchanged.
fn assert_revalidates(row_ptrs: &[u32], col_indices: &[u32], values: &[f64], n: u32) {
    let csr = CsrRef::new(row_ptrs, col_indices, values, n).expect("valid CSR");
    let converted = Sddm::try_from(csr).expect("valid SDDM");
    let laplacian = converted.laplacian();
    let rebuilt = Laplacian::new(
        laplacian.row_ptrs().to_vec(),
        laplacian.neighbors().to_vec(),
        laplacian.weights().to_vec(),
    )
    .expect("a converted Laplacian passes the typed checks");
    let surplus = match &converted {
        Sddm::Laplacian(_) => vec![0.0; laplacian.n()],
        Sddm::Grounded(grounded) => grounded.surplus().to_vec(),
    };
    let retyped =
        Sddm::with_surplus(rebuilt, surplus).expect("converted surplus passes the typed checks");
    assert_eq!(
        matches!(retyped, Sddm::Grounded(_)),
        matches!(converted, Sddm::Grounded(_))
    );
    assert_eq!(retyped.laplacian().weights(), laplacian.weights());
}

proptest! {
    #[test]
    fn a_converted_laplacian_revalidates((row_ptrs, col_indices, values, n) in laplacian_csr_strategy()) {
        assert_revalidates(&row_ptrs, &col_indices, &values, n);
    }

    #[test]
    fn a_converted_grounded_sddm_revalidates(((row_ptrs, col_indices, values, n), _) in one_grounded_component_strategy()) {
        assert_revalidates(&row_ptrs, &col_indices, &values, n);
    }

    #[test]
    fn reports_row_ptr_length_mismatch(
        (mut row_ptrs, col_indices, values, n) in laplacian_csr_strategy()
    ) {
        row_ptrs.pop();
        let err = CsrRef::new(&row_ptrs, &col_indices, &values, n).expect_err("must fail");
        prop_assert_eq!(
            err,
            CsrError::RowPtrsLenMismatch {
                expected: (n as usize) + 1,
                got: row_ptrs.len(),
            }
        );
    }

    #[test]
    fn reports_col_values_length_mismatch(
        (row_ptrs, col_indices, mut values, n) in laplacian_csr_strategy()
    ) {
        values.pop();
        let err = CsrRef::new(&row_ptrs, &col_indices, &values, n).expect_err("must fail");
        prop_assert_eq!(
            err,
            CsrError::ColIndicesValuesLenMismatch {
                col_indices_len: col_indices.len(),
                values_len: values.len(),
            }
        );
    }

    // A non-zero start leaves the first payload entry addressable by no row, so
    // accepting it would factorize a different matrix than the caller passed.
    #[test]
    fn reports_non_zero_row_ptr_start(
        (mut row_ptrs, col_indices, values, n) in laplacian_csr_strategy()
    ) {
        row_ptrs[0] = 1;
        let err = CsrRef::new(&row_ptrs, &col_indices, &values, n).expect_err("must fail");
        prop_assert_eq!(
            err,
            CsrError::RowPtrsMustStartAtZero { got: 1 }
        );
    }

    #[test]
    fn reports_row_ptr_end_mismatch(
        (mut row_ptrs, col_indices, values, n) in laplacian_csr_strategy()
    ) {
        let last = row_ptrs.len() - 1;
        row_ptrs[last] = row_ptrs[last].saturating_sub(1);
        let err = CsrRef::new(&row_ptrs, &col_indices, &values, n).expect_err("must fail");
        prop_assert_eq!(
            err,
            CsrError::RowPtrsEndMismatchNnz {
                row_ptr_end: row_ptrs[last] as usize,
                nnz: col_indices.len(),
            }
        );
    }

    #[test]
    fn reports_non_monotone_row_ptrs(
        (mut row_ptrs, col_indices, values, n) in laplacian_csr_strategy()
    ) {
        prop_assume!(n >= 2);
        row_ptrs[1] = row_ptrs[2].saturating_add(1);
        let err = CsrRef::new(&row_ptrs, &col_indices, &values, n).expect_err("must fail");
        prop_assert_eq!(
            err,
            CsrError::RowPtrsNotNonDecreasing {
                row: 1,
                prev: row_ptrs[1] as usize,
                next: row_ptrs[2] as usize,
            }
        );
    }

    #[test]
    fn reports_out_of_bounds_column(
        (row_ptrs, mut col_indices, values, n) in laplacian_csr_strategy()
    ) {
        col_indices[0] = n;
        let err = CsrRef::new(&row_ptrs, &col_indices, &values, n).expect_err("must fail");
        prop_assert_eq!(
            err,
            CsrError::ColumnIndexOutOfBounds {
                position: 0,
                col: n as usize,
                n: n as usize,
            }
        );
    }
}
