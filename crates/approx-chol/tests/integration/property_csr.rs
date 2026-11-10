use approx_chol::input::{CsrRef, InputError, TriangleCsrRef};
use approx_chol::{factorize, ApproxInverse, Error};
use num_traits::PrimInt;
use proptest::prelude::*;

use crate::common::entry::cast_all;

type LaplacianCsr = (Vec<u32>, Vec<u32>, Vec<f64>, usize);

fn laplacian_csr(n: usize, edge_weights: &[u8]) -> LaplacianCsr {
    let mut dense = vec![0.0_f64; n * n];
    let mut weights = edge_weights.iter();
    for i in 0..n {
        for j in (i + 1)..n {
            let w = f64::from(*weights.next().expect("one weight per pair"));
            dense[i * n + j] -= w;
            dense[j * n + i] -= w;
            dense[i * n + i] += w;
            dense[j * n + j] += w;
        }
    }
    let (mut row_ptrs, mut col_indices, mut values) = (vec![0u32], Vec::new(), Vec::new());
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
    (row_ptrs, col_indices, values, n)
}

fn laplacian_csr_strategy() -> impl Strategy<Value = LaplacianCsr> {
    (1usize..=8).prop_flat_map(|n| {
        prop::collection::vec(0u8..=4, n * (n - 1) / 2)
            .prop_map(move |edge_weights| laplacian_csr(n, &edge_weights))
    })
}

fn upper((row_ptrs, col_indices, values, n): &LaplacianCsr) -> LaplacianCsr {
    let (mut ptrs, mut columns, mut kept) = (vec![0u32], Vec::new(), Vec::new());
    for row in 0..*n {
        for k in row_ptrs[row] as usize..row_ptrs[row + 1] as usize {
            if col_indices[k] as usize >= row {
                columns.push(col_indices[k]);
                kept.push(values[k]);
            }
        }
        ptrs.push(columns.len() as u32);
    }
    (ptrs, columns, kept, *n)
}

/// Both constructors share the shape checks, so each must name the same defect.
fn shape_errors<I: PrimInt + std::fmt::Debug>(
    n: usize,
    row_ptrs: &[I],
    col_indices: &[I],
    values: &[f64],
) -> [InputError; 2] {
    let full = CsrRef::new(n, row_ptrs, col_indices, values).expect_err("full storage must fail");
    let upper = TriangleCsrRef::upper(n, row_ptrs, col_indices, values)
        .expect_err("upper storage must fail");
    [full, upper]
}

/// Columns are checked at reduction, where both storages canonicalize.
fn column_errors<I: PrimInt + Sync + std::fmt::Debug>(
    n: usize,
    row_ptrs: &[I],
    col_indices: &[I],
    values: &[f64],
) -> [Error; 2] {
    let csr = CsrRef::new(n, row_ptrs, col_indices, values).unwrap_err();
    let upper = TriangleCsrRef::upper(n, row_ptrs, col_indices, values).unwrap_err();
    [Error::Input(csr), Error::Input(upper)]
}

proptest! {
    #[test]
    fn generated_laplacians_reduce_alike_from_full_and_upper_storage(csr in laplacian_csr_strategy()) {
        let (row_ptrs, col_indices, values, n) = &csr;
        let (upper_ptrs, upper_columns, upper_values, _) = upper(&csr);
        let full = CsrRef::new(*n, row_ptrs, col_indices, values).expect("valid shape");
        let triangle = TriangleCsrRef::upper(*n, &upper_ptrs, &upper_columns, &upper_values).expect("valid shape");
        let full: ApproxInverse = factorize(full).expect("a Laplacian is admitted");
        let triangle: ApproxInverse = factorize(triangle).expect("a Laplacian is admitted");
        prop_assert_eq!(format!("{full:?}"), format!("{triangle:?}"));
        prop_assert!(full.components().all(|component| component.floats()));
    }

    #[test]
    fn row_pointers_one_short_are_a_row_pointer_length_mismatch(
        (mut row_ptrs, col_indices, values, n) in laplacian_csr_strategy()
    ) {
        row_ptrs.pop();
        let expected = InputError::LengthMismatch { parameter: "row_ptrs", expected: n + 1, found: n };
        prop_assert_eq!(shape_errors(n, &row_ptrs, &col_indices, &values), [expected; 2]);
    }

    #[test]
    fn values_one_short_mismatch_the_columns(
        (row_ptrs, col_indices, mut values, n) in laplacian_csr_strategy()
    ) {
        values.pop();
        let expected = InputError::LengthMismatch { parameter: "values", expected: col_indices.len(), found: values.len() };
        prop_assert_eq!(shape_errors(n, &row_ptrs, &col_indices, &values), [expected; 2]);
    }

    #[test]
    fn columns_one_short_mismatch_the_values(
        (row_ptrs, mut col_indices, values, n) in laplacian_csr_strategy()
    ) {
        col_indices.pop();
        let expected = InputError::LengthMismatch { parameter: "values", expected: col_indices.len(), found: values.len() };
        prop_assert_eq!(shape_errors(n, &row_ptrs, &col_indices, &values), [expected; 2]);
    }

    // A nonzero start leaves the first entry in no row, so accepting it would reduce another matrix.
    #[test]
    fn row_pointers_not_starting_at_zero_report_their_start(
        (mut row_ptrs, col_indices, values, n) in laplacian_csr_strategy()
    ) {
        row_ptrs[0] = 1;
        let expected = InputError::RowPtrs { row: 0 };
        prop_assert_eq!(shape_errors(n, &row_ptrs, &col_indices, &values), [expected; 2]);
    }

    #[test]
    fn row_pointer_end_short_of_the_entries_mismatches_nnz(
        (mut row_ptrs, col_indices, values, n) in laplacian_csr_strategy()
    ) {
        row_ptrs[n] -= 1;
        let expected = InputError::LengthMismatch { parameter: "col_indices", expected: row_ptrs[n] as usize, found: col_indices.len() };
        prop_assert_eq!(shape_errors(n, &row_ptrs, &col_indices, &values), [expected; 2]);
    }

    #[test]
    fn falling_row_pointers_are_rejected_at_the_row_they_close(
        (mut row_ptrs, col_indices, values, n) in laplacian_csr_strategy()
    ) {
        prop_assume!(n >= 2);
        row_ptrs[1] = row_ptrs[2] + 1;
        let expected = InputError::RowPtrs { row: 1 };
        prop_assert_eq!(shape_errors(n, &row_ptrs, &col_indices, &values), [expected; 2]);
    }

    #[test]
    fn a_negative_last_row_pointer_is_out_of_range_at_its_position(
        (row_ptrs, col_indices, values, n) in laplacian_csr_strategy()
    ) {
        let (mut row_ptrs, col_indices) = (cast_all::<u32, i64>(&row_ptrs), cast_all::<u32, i64>(&col_indices));
        row_ptrs[n] = -1;
        let expected = InputError::IndexOutOfRange { parameter: "row_ptrs", entry: n };
        prop_assert_eq!(shape_errors(n, &row_ptrs, &col_indices, &values), [expected; 2]);
    }

    #[test]
    fn a_column_past_n_is_out_of_bounds_at_its_position(
        (row_ptrs, mut col_indices, values, n) in laplacian_csr_strategy()
    ) {
        col_indices[0] = n as u32;
        let expected = Error::Input(InputError::IndexOutOfRange { parameter: "col_indices", entry: 0 });
        prop_assert_eq!(column_errors(n, &row_ptrs, &col_indices, &values), [expected; 2]);
    }

    #[test]
    fn a_negative_column_is_out_of_range_at_its_position(
        (row_ptrs, col_indices, values, n) in laplacian_csr_strategy()
    ) {
        let (row_ptrs, mut col_indices) = (cast_all::<u32, i64>(&row_ptrs), cast_all::<u32, i64>(&col_indices));
        let last = col_indices.len() - 1;
        col_indices[last] = -1;
        let expected = Error::Input(InputError::IndexOutOfRange { parameter: "col_indices", entry: last });
        prop_assert_eq!(column_errors(n, &row_ptrs, &col_indices, &values), [expected; 2]);
    }
}

#[rstest::rstest]
#[case::negative_middle(CsrRef::<f64, i64>::new(2, &[0, -1, 2], &[0, 1], &[1.0, 1.0]).err(), 1)]
#[case::negative_first(CsrRef::<f64, i64>::new(2, &[-1, 1, 2], &[0, 1], &[1.0, 1.0]).err(), 0)]
#[case::past_usize_last(CsrRef::<f64, u128>::new(1, &[0, u128::MAX], &[0], &[1.0]).err(), 1)]
fn a_row_pointer_past_usize_is_out_of_range_at_its_position(
    #[case] error: Option<InputError>,
    #[case] entry: usize,
) {
    let expected = InputError::IndexOutOfRange {
        parameter: "row_ptrs",
        entry,
    };
    assert_eq!(error, Some(expected));
}

#[test]
fn a_decreasing_pair_of_valid_pointers_is_out_of_order_at_the_first() {
    let error = CsrRef::<f64>::new(3, &[0, 2, 1, 2], &[0, 1], &[1.0, 1.0]).err();
    assert_eq!(error, Some(InputError::RowPtrs { row: 1 }));
}
