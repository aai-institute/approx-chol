use approx_chol_next::input::{Csr, CsrError, UpperCsr};
use approx_chol_next::{FromCsrError, ScalingPolicy, Sgdd};
use num_traits::PrimInt;
use proptest::prelude::*;

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

fn widen<I: PrimInt>(indices: &[u32]) -> Vec<I> {
    (indices.iter())
        .map(|&index| I::from(index).expect("fits"))
        .collect()
}

/// Both constructors share the shape checks, so each must name the same defect.
fn shape_errors<I: PrimInt + std::fmt::Debug>(
    n: usize,
    row_ptrs: &[I],
    col_indices: &[I],
    values: &[f64],
) -> [CsrError; 2] {
    let full = Csr::new(n, row_ptrs, col_indices, values).expect_err("full storage must fail");
    let upper =
        UpperCsr::new(n, row_ptrs, col_indices, values).expect_err("upper storage must fail");
    [full, upper]
}

/// Columns are checked at reduction, where both storages canonicalize.
fn column_errors<I: PrimInt + Sync + std::fmt::Debug>(
    n: usize,
    row_ptrs: &[I],
    col_indices: &[I],
    values: &[f64],
) -> [FromCsrError; 2] {
    let policy = ScalingPolicy::default();
    let csr = Csr::new(n, row_ptrs, col_indices, values).expect("valid shape");
    let upper = UpperCsr::new(n, row_ptrs, col_indices, values).expect("valid shape");
    [
        Sgdd::<f64>::from_csr(csr, policy).expect_err("full storage must fail"),
        Sgdd::<f64>::from_upper_csr(upper, policy).expect_err("upper storage must fail"),
    ]
}

proptest! {
    #[test]
    fn generated_laplacians_reduce_alike_from_full_and_upper_storage(csr in laplacian_csr_strategy()) {
        let (row_ptrs, col_indices, values, n) = &csr;
        let (upper_ptrs, upper_columns, upper_values, _) = upper(&csr);
        let full = Csr::new(*n, row_ptrs, col_indices, values).expect("valid shape");
        let triangle = UpperCsr::new(*n, &upper_ptrs, &upper_columns, &upper_values).expect("valid shape");
        let full: Sgdd = Sgdd::from_csr(full, ScalingPolicy::default()).expect("a Laplacian is admitted");
        let triangle: Sgdd = Sgdd::from_upper_csr(triangle, ScalingPolicy::default()).expect("a Laplacian is admitted");
        prop_assert_eq!(format!("{full:?}"), format!("{triangle:?}"));
        prop_assert!(full.blocks().iter().all(|block| block.floats()));
    }

    #[test]
    fn row_pointers_one_short_are_a_length_mismatch(
        (mut row_ptrs, col_indices, values, n) in laplacian_csr_strategy()
    ) {
        row_ptrs.pop();
        let expected = CsrError::LengthMismatch { expected: n + 1, found: n };
        prop_assert_eq!(shape_errors(n, &row_ptrs, &col_indices, &values), [expected; 2]);
    }

    #[test]
    fn values_one_short_are_a_length_mismatch(
        (row_ptrs, col_indices, mut values, n) in laplacian_csr_strategy()
    ) {
        values.pop();
        let expected = CsrError::LengthMismatch { expected: col_indices.len(), found: values.len() };
        prop_assert_eq!(shape_errors(n, &row_ptrs, &col_indices, &values), [expected; 2]);
    }

    #[test]
    fn columns_one_short_are_a_length_mismatch(
        (row_ptrs, mut col_indices, values, n) in laplacian_csr_strategy()
    ) {
        col_indices.pop();
        let expected = CsrError::LengthMismatch { expected: values.len(), found: col_indices.len() };
        prop_assert_eq!(shape_errors(n, &row_ptrs, &col_indices, &values), [expected; 2]);
    }

    // A nonzero start leaves the first entry in no row, so accepting it would reduce another matrix.
    #[test]
    fn row_pointers_not_starting_at_zero_are_rejected_at_row_zero(
        (mut row_ptrs, col_indices, values, n) in laplacian_csr_strategy()
    ) {
        row_ptrs[0] = 1;
        let expected = CsrError::RowPointers { row: 0 };
        prop_assert_eq!(shape_errors(n, &row_ptrs, &col_indices, &values), [expected; 2]);
    }

    #[test]
    fn row_pointer_end_short_of_the_entries_is_a_length_mismatch(
        (mut row_ptrs, col_indices, values, n) in laplacian_csr_strategy()
    ) {
        row_ptrs[n] -= 1;
        let expected = CsrError::LengthMismatch { expected: row_ptrs[n] as usize, found: col_indices.len() };
        prop_assert_eq!(shape_errors(n, &row_ptrs, &col_indices, &values), [expected; 2]);
    }

    #[test]
    fn falling_row_pointers_are_rejected_at_the_row_they_close(
        (mut row_ptrs, col_indices, values, n) in laplacian_csr_strategy()
    ) {
        prop_assume!(n >= 2);
        row_ptrs[1] = row_ptrs[2] + 1;
        let expected = CsrError::RowPointers { row: 1 };
        prop_assert_eq!(shape_errors(n, &row_ptrs, &col_indices, &values), [expected; 2]);
    }

    #[test]
    fn a_negative_row_pointer_is_rejected_at_the_row_it_closes(
        (row_ptrs, col_indices, values, n) in laplacian_csr_strategy()
    ) {
        let (mut row_ptrs, col_indices) = (widen::<i64>(&row_ptrs), widen::<i64>(&col_indices));
        row_ptrs[n] = -1;
        let expected = CsrError::RowPointers { row: n - 1 };
        prop_assert_eq!(shape_errors(n, &row_ptrs, &col_indices, &values), [expected; 2]);
    }

    #[test]
    fn a_column_past_n_is_rejected_at_its_row(
        (row_ptrs, mut col_indices, values, n) in laplacian_csr_strategy()
    ) {
        col_indices[0] = n as u32;
        let expected = FromCsrError::Csr(CsrError::ColumnOutOfRange { row: 0 });
        prop_assert_eq!(column_errors(n, &row_ptrs, &col_indices, &values), [expected; 2]);
    }

    #[test]
    fn a_negative_column_is_rejected_at_its_row(
        (row_ptrs, col_indices, values, n) in laplacian_csr_strategy()
    ) {
        let (row_ptrs, mut col_indices) = (widen::<i64>(&row_ptrs), widen::<i64>(&col_indices));
        let last = col_indices.len() - 1;
        col_indices[last] = -1;
        let expected = FromCsrError::Csr(CsrError::ColumnOutOfRange { row: n - 1 });
        prop_assert_eq!(column_errors(n, &row_ptrs, &col_indices, &values), [expected; 2]);
    }
}
