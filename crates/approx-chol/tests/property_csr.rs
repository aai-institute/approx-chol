#[path = "common/laplacian_prop.rs"]
mod laplacian_prop;
#[path = "common/panic_err.rs"]
mod panic_err;
#[path = "common/panic_ok.rs"]
mod panic_ok;
use panic_err::ErrOrPanic;
use panic_ok::OrPanic;

use approx_chol::{CsrError, CsrRef, Error};
use laplacian_prop::laplacian_csr_strategy;
use proptest::prelude::*;
use std::panic::{catch_unwind, AssertUnwindSafe};

proptest! {
    #[test]
    fn valid_try_row_access_is_panic_free(
        (row_ptrs, col_indices, values, n) in laplacian_csr_strategy()
    ) {
        let csr = CsrRef::new(&row_ptrs, &col_indices, &values, n)
            .or_panic("generated CSR must be valid");

        let access = catch_unwind(AssertUnwindSafe(|| {
            for i in 0..(n as usize) {
                let _ = csr.try_row(i);
            }
        }));
        prop_assert!(access.is_ok(), "try_row() panicked on validated CSR");
    }

    #[test]
    fn try_row_out_of_bounds_is_structured_error(
        (row_ptrs, col_indices, values, n) in laplacian_csr_strategy(),
        extra in 0usize..=8
    ) {
        let csr = CsrRef::new(&row_ptrs, &col_indices, &values, n)
            .or_panic("generated CSR must be valid");
        let row = (n as usize).saturating_add(extra);
        let err = csr.try_row(row).err_or_panic("out-of-bounds row must fail");
        prop_assert_eq!(
            err,
            Error::InvalidCsr(CsrError::RowIndexOutOfBounds {
                row,
                n: n as usize,
            })
        );
    }

    #[test]
    fn reports_row_ptr_length_mismatch(
        (mut row_ptrs, col_indices, values, n) in laplacian_csr_strategy()
    ) {
        row_ptrs.pop();
        let err = CsrRef::new(&row_ptrs, &col_indices, &values, n).err_or_panic("must fail");
        prop_assert_eq!(
            err,
            Error::InvalidCsr(CsrError::RowPtrsLenMismatch {
                expected: (n as usize) + 1,
                got: row_ptrs.len(),
            })
        );
    }

    #[test]
    fn reports_col_values_length_mismatch(
        (row_ptrs, col_indices, mut values, n) in laplacian_csr_strategy()
    ) {
        values.pop();
        let err = CsrRef::new(&row_ptrs, &col_indices, &values, n).err_or_panic("must fail");
        prop_assert_eq!(
            err,
            Error::InvalidCsr(CsrError::ColIndicesValuesLenMismatch {
                col_indices_len: col_indices.len(),
                values_len: values.len(),
            })
        );
    }

    #[test]
    fn reports_non_zero_row_ptr_start(
        (mut row_ptrs, col_indices, values, n) in laplacian_csr_strategy()
    ) {
        row_ptrs[0] = 1;
        let err = CsrRef::new(&row_ptrs, &col_indices, &values, n).err_or_panic("must fail");
        prop_assert_eq!(
            err,
            Error::InvalidCsr(CsrError::RowPtrsMustStartAtZero { got: 1 })
        );
    }

    #[test]
    fn reports_row_ptr_end_mismatch(
        (mut row_ptrs, col_indices, values, n) in laplacian_csr_strategy()
    ) {
        let last = row_ptrs.len() - 1;
        row_ptrs[last] = row_ptrs[last].saturating_sub(1);
        let err = CsrRef::new(&row_ptrs, &col_indices, &values, n).err_or_panic("must fail");
        prop_assert_eq!(
            err,
            Error::InvalidCsr(CsrError::RowPtrsEndMismatchNnz {
                row_ptr_end: row_ptrs[last] as usize,
                nnz: col_indices.len(),
            })
        );
    }

    #[test]
    fn reports_non_monotone_row_ptrs(
        (mut row_ptrs, col_indices, values, n) in laplacian_csr_strategy()
    ) {
        prop_assume!(n >= 2);
        row_ptrs[1] = row_ptrs[2].saturating_add(1);
        let err = CsrRef::new(&row_ptrs, &col_indices, &values, n).err_or_panic("must fail");
        prop_assert_eq!(
            err,
            Error::InvalidCsr(CsrError::RowPtrsNotNonDecreasing {
                row: 1,
                prev: row_ptrs[1] as usize,
                next: row_ptrs[2] as usize,
            })
        );
    }

    #[test]
    fn reports_out_of_bounds_column(
        (row_ptrs, mut col_indices, values, n) in laplacian_csr_strategy()
    ) {
        col_indices[0] = n;
        let err = CsrRef::new(&row_ptrs, &col_indices, &values, n).err_or_panic("must fail");
        prop_assert_eq!(
            err,
            Error::InvalidCsr(CsrError::ColumnIndexOutOfBounds {
                position: 0,
                col: n as usize,
                n: n as usize,
            })
        );
    }
}
