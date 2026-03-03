mod common;
use common::{ErrOrPanic, OrPanic};

use approx_chol::{CsrError, CsrRef, Error};
use proptest::prelude::*;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn build_laplacian_csr(n: usize, edge_weights: &[u8]) -> (Vec<u32>, Vec<u32>, Vec<f64>, u32) {
    let mut dense = vec![0.0_f64; n * n];
    let mut edge_pos = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let w = edge_weights[edge_pos] as f64;
            edge_pos += 1;
            if w <= 0.0 {
                continue;
            }
            dense[i * n + j] -= w;
            dense[j * n + i] -= w;
            dense[i * n + i] += w;
            dense[j * n + j] += w;
        }
    }

    let mut row_ptrs = Vec::with_capacity(n + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    row_ptrs.push(0u32);
    for i in 0..n {
        for j in 0..n {
            let value = dense[i * n + j];
            if i == j || value != 0.0 {
                col_indices.push(j as u32);
                values.push(value);
            }
        }
        row_ptrs.push(col_indices.len() as u32);
    }

    (row_ptrs, col_indices, values, n as u32)
}

fn laplacian_csr_strategy() -> impl Strategy<Value = (Vec<u32>, Vec<u32>, Vec<f64>, u32)> {
    (1usize..=8).prop_flat_map(|n| {
        let pair_count = n * (n - 1) / 2;
        prop::collection::vec(0u8..=4, pair_count)
            .prop_map(move |edge_weights| build_laplacian_csr(n, &edge_weights))
    })
}

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
