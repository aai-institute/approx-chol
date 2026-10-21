use approx_chol::input::{
    BipartiteGram, CsrError, CsrRef, FactorWidth1, FactorWidth2, Laplacian, UpperCsrRef, Weights,
};
use approx_chol::{factorize, Error, Factor, ReduceError};
use num_traits::PrimInt;
use rstest::rstest;

use super::{assert_names_a_defect, reduce, reduce_upper};
use crate::grid::grid;

const F32_FLOOR: f32 = f32::MIN_POSITIVE / f32::EPSILON;

fn csr(error: CsrError) -> Error {
    Error::InvalidCsr(error)
}

fn out_of_range(vertex: usize) -> Error {
    Error::Reduce(ReduceError::OutOfRange { vertex })
}

/// The smallest nonzero magnitude admitted (#163).
fn floor() -> f64 {
    f64::MIN_POSITIVE / f64::EPSILON
}

/// The same matrix with every row's entries reversed, which forces the canonicalizing copy.
fn reversed_rows<T: Copy>(row_ptrs: &[u32], columns: &[u32], values: &[T]) -> (Vec<u32>, Vec<T>) {
    let (mut reversed_columns, mut reversed_values) = (Vec::new(), Vec::new());
    for span in row_ptrs.windows(2) {
        let span = span[0] as usize..span[1] as usize;
        reversed_columns.extend(columns[span.clone()].iter().rev());
        reversed_values.extend(values[span].iter().rev());
    }
    (reversed_columns, reversed_values)
}

#[rstest]
#[case::missing_transpose(
    &[0, 2, 3],
    &[0, 1, 1],
    &[1.0, -1.0, 1.0],
    &[csr(CsrError::NotSymmetric { row: 0 }), csr(CsrError::NotSymmetric { row: 1 })],
)]
// The lower entry is stored and its upper is absent: the mirror cursor skips past it.
#[case::missing_upper_mirror(
    &[0, 1, 3],
    &[0, 0, 1],
    &[1.0, -1.0, 1.0],
    &[csr(CsrError::NotSymmetric { row: 0 }), csr(CsrError::NotSymmetric { row: 1 })],
)]
#[case::unequal_transpose(
    &[0, 2, 4],
    &[0, 1, 0, 1],
    &[1.0, -1.0, -2.0, 2.0],
    &[csr(CsrError::NotSymmetric { row: 0 }), csr(CsrError::NotSymmetric { row: 1 })],
)]
#[case::mirrors_nine_ulps_apart(
    &[0, 2, 4],
    &[0, 1, 0, 1],
    &[2.0, -1.0, -(1.0 + 9.0 * f64::EPSILON), 2.0],
    &[csr(CsrError::NotSymmetric { row: 0 }), csr(CsrError::NotSymmetric { row: 1 })],
)]
#[case::stored_nan(&[0, 1], &[0], &[f64::NAN], &[csr(CsrError::OutOfRange { row: 0 })])]
#[case::stored_infinity_off_the_diagonal(
    &[0, 2, 4],
    &[0, 1, 0, 1],
    &[1.0, f64::NEG_INFINITY, f64::NEG_INFINITY, 1.0],
    &[csr(CsrError::OutOfRange { row: 0 }), csr(CsrError::OutOfRange { row: 1 })],
)]
// Descending columns in row 0, so the row must be the caller's, not the canonical copy's.
#[case::stored_nan_in_a_later_row_of_non_canonical_input(
    &[0, 2, 5, 7],
    &[1, 0, 2, 1, 0, 2, 1],
    &[-1.0, 1.0, -1.0, 2.0, f64::NAN, 1.0, -1.0],
    &[csr(CsrError::OutOfRange { row: 1 })],
)]
#[case::overflow_via_coalesced_duplicates(
    &[0, 3, 6],
    &[0, 1, 1, 0, 0, 1],
    &[f64::MAX, -f64::MAX, -f64::MAX, -f64::MAX, -f64::MAX, f64::MAX],
    &[csr(CsrError::OutOfRange { row: 0 }), csr(CsrError::OutOfRange { row: 1 })],
)]
#[case::overflow_via_duplicate_diagonal(
    &[0, 2],
    &[0, 0],
    &[f64::MAX, f64::MAX],
    &[csr(CsrError::OutOfRange { row: 0 })],
)]
// Every entry in range and the matrix GDD, but the centre's `Σ|a_ij|` passes `MAX/4`.
#[case::absolute_row_sum_past_a_quarter_of_the_maximum(
    &[0, 3, 5, 7],
    &[0, 1, 2, 0, 1, 0, 2],
    &[
        f64::MAX / 4.0,
        -0.6 * (f64::MAX / 4.0),
        -0.6 * (f64::MAX / 4.0),
        -0.6 * (f64::MAX / 4.0),
        f64::MAX / 4.0,
        -0.6 * (f64::MAX / 4.0),
        f64::MAX / 4.0,
    ],
    &[out_of_range(0)],
)]
#[case::diagonal_past_a_quarter_of_the_maximum(&[0, 1], &[0], &[f64::MAX], &[out_of_range(0)])]
// Normal diagonals; the edge's pivot alone would have no finite inverse (#163).
#[case::subnormal_edge(
    &[0, 2, 4],
    &[0, 1, 0, 1],
    &[1.0, -1e-310, -1e-310, 1.0],
    &[csr(CsrError::OutOfRange { row: 0 }), csr(CsrError::OutOfRange { row: 1 })],
)]
#[case::subnormal_diagonal(&[0, 1], &[0], &[1e-310], &[csr(CsrError::OutOfRange { row: 0 })])]
#[case::normal_edge_below_min_positive_over_epsilon(
    &[0, 2, 4],
    &[0, 1, 0, 1],
    &[1.0, -floor() / 2.0, -floor() / 2.0, 1.0],
    &[csr(CsrError::OutOfRange { row: 0 }), csr(CsrError::OutOfRange { row: 1 })],
)]
#[case::subnormal_edge_in_a_later_row(
    &[0, 1, 3, 5],
    &[0, 1, 2, 1, 2],
    &[1.0, 1.0, -1e-310, -1e-310, 1.0],
    &[csr(CsrError::OutOfRange { row: 1 }), csr(CsrError::OutOfRange { row: 2 })],
)]
#[case::negative_diagonal(
    &[0, 1],
    &[0],
    &[-1.0],
    &[Error::Reduce(ReduceError::NonpositiveDiagonal { vertex: 0 })],
)]
#[case::zero_diagonal_beside_an_edge(
    &[0, 2, 4],
    &[0, 1, 0, 1],
    &[0.0, -1.0, -1.0, 1.0],
    &[Error::Reduce(ReduceError::NonpositiveDiagonal { vertex: 0 })],
)]
// Components {0, 2} and {1, 3}: the vertex is in the caller's numbering, not the block's.
#[case::nonpositive_diagonal_in_an_interleaved_later_block(
    &[0, 2, 4, 6, 8],
    &[0, 2, 1, 3, 0, 2, 1, 3],
    &[1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0],
    &[Error::Reduce(ReduceError::NonpositiveDiagonal { vertex: 3 })],
)]
fn out_of_class_input_is_rejected_at_its_reported_position(
    #[case] row_ptrs: &[u32],
    #[case] columns: &[u32],
    #[case] values: &[f64],
    #[case] defects: &[Error],
) {
    let observed = reduce(row_ptrs, columns, values).expect_err("out of class");
    assert_names_a_defect(observed, defects);
}

#[rstest]
#[case::entry_below_the_diagonal(
    &[0, 1, 3],
    &[0, 0, 1],
    &[1.0, -1.0, 1.0],
    &[csr(CsrError::BelowDiagonal { row: 1 })],
)]
#[case::entry_below_the_diagonal_in_non_canonical_input(
    &[0, 2, 4],
    &[1, 0, 1, 0],
    &[-1.0, 1.0, 1.0, -1.0],
    &[csr(CsrError::BelowDiagonal { row: 1 })],
)]
#[case::stored_nan(
    &[0, 2, 3],
    &[0, 1, 1],
    &[1.0, f64::NAN, 1.0],
    &[csr(CsrError::OutOfRange { row: 0 })],
)]
fn out_of_class_upper_storage_is_rejected_at_its_reported_position(
    #[case] row_ptrs: &[u32],
    #[case] columns: &[u32],
    #[case] values: &[f64],
    #[case] defects: &[Error],
) {
    let observed = reduce_upper(row_ptrs, columns, values).expect_err("out of class");
    assert_names_a_defect(observed, defects);
}

/// Shapes a stricter check would reject, admitted as given and through the canonicalizing copy.
#[rstest]
#[case::one_ulp_transpose_difference(
    &[0, 2, 4],
    &[0, 1, 0, 1],
    &[2.0, -1.0, -f64::from_bits(1.0f64.to_bits() + 1), 2.0],
)]
#[case::eight_ulp_transpose_difference(
    &[0, 2, 4],
    &[0, 1, 0, 1],
    &[2.0, -1.0, -(1.0 + 8.0 * f64::EPSILON), 2.0],
)]
#[case::transposes_equal_only_after_coalescing(
    &[0, 3, 6],
    &[0, 1, 1, 0, 0, 1],
    &[2.0, -0.25, -0.75, -0.5, -0.5, 2.0],
)]
#[case::stored_zero_without_a_mirror(&[0, 2, 3], &[0, 1, 1], &[1.0, 0.0, 1.0])]
#[case::block_diagonal_sddm(
    &[0, 2, 4, 6, 8],
    &[0, 1, 0, 1, 2, 3, 2, 3],
    &[5.0, -1.0, -1.0, 4.0, 5.0, -1.0, -1.0, 4.0],
)]
// Generalized dominance: a signature folds it into a Z-matrix.
#[case::positive_off_diagonal(&[0, 2, 4], &[0, 1, 0, 1], &[5.0, 1.0, 1.0, 4.0])]
#[case::diagonal_at_a_quarter_of_the_maximum(
    &[0, 2, 4],
    &[0, 1, 0, 1],
    &[f64::MAX / 4.0, -(f64::MAX / 4.0) / 2.0, -(f64::MAX / 4.0) / 2.0, f64::MAX / 4.0],
)]
#[case::edge_at_min_positive_over_epsilon(
    &[0, 2, 4],
    &[0, 1, 0, 1],
    &[1.0, -floor(), -floor(), 1.0],
)]
fn in_class_input_is_admitted_on_both_paths(
    #[case] row_ptrs: &[u32],
    #[case] columns: &[u32],
    #[case] values: &[f64],
) {
    let (reversed_columns, reversed_values) = reversed_rows(row_ptrs, columns, values);
    let given = reduce(row_ptrs, columns, values).expect("as given");
    let copied = reduce(row_ptrs, &reversed_columns, &reversed_values).expect("reversed");
    assert_eq!(format!("{given:?}"), format!("{copied:?}"));
}

#[rstest]
#[case::csr(factorize(CsrRef::<f64>::new(&[0], &[], &[], 0).unwrap()).expect("CSR"))]
#[case::upper_csr(factorize(UpperCsrRef::<f64>::new(&[0], &[], &[], 0).unwrap()).expect("upper CSR"))]
#[case::laplacian(
    factorize(Laplacian::<f64>::new(0, &[], &[], Weights::unit()).unwrap()).expect("Laplacian")
)]
#[case::gram(
    factorize(FactorWidth2::<f64>::new(0, &[], &[], &[], &[]).unwrap()).expect("Gram")
)]
#[case::bipartite_gram(factorize(
    BipartiteGram::new(
        FactorWidth1::<f64>::new(0, &[], None).unwrap(),
        FactorWidth1::<f64>::new(0, &[], None).unwrap(),
        Weights::unit(),
    )
    .unwrap(),
)
.expect("bipartite Gram"))]
fn an_empty_system_is_admitted_by_every_entry_point(#[case] factor: Factor) {
    assert_eq!((factor.n(), factor.blocks().len()), (0, 0));
}

#[rstest]
#[case::csr(CsrRef::<f64>::new(&[0], &[], &[], u32::MAX).err())]
#[case::upper_csr(UpperCsrRef::<f64>::new(&[0], &[], &[], u32::MAX).err())]
fn a_dimension_past_the_largest_row_count_is_a_row_pointer_length_mismatch(
    #[case] error: Option<Error>,
) {
    let expected = CsrError::RowPtrsLenMismatch {
        expected: u32::MAX as usize + 1,
        got: 1,
    };
    assert_eq!(error, Some(csr(expected)));
}

#[test]
fn a_diagonal_at_a_quarter_of_the_maximum_still_solves() {
    let quarter = f64::MAX / 4.0;
    let factor = reduce(&[0, 1], &[0], &[quarter]).expect("diagonal at the bound");
    let solution = factor.solve(&[quarter]).expect("solve");
    assert!((solution[0] - 1.0).abs() < 1e-12, "{solution:?}");
}

/// The #163 range in `f32`'s own bounds: `MIN_POSITIVE/EPSILON` is `2^-103`, not `f64`'s `2^-970`.
fn f32_pair(diagonal: f32, edge: f32) -> Result<Factor<f32>, Error> {
    reduce(&[0, 2, 4], &[0, 1, 0, 1], &[diagonal, edge, edge, diagonal])
}

#[rstest]
#[case::subnormal_edge(
    1.0,
    -1e-40,
    &[csr(CsrError::OutOfRange { row: 0 }), csr(CsrError::OutOfRange { row: 1 })],
)]
#[case::normal_edge_below_the_floor(
    1.0,
    -F32_FLOOR / 2.0,
    &[csr(CsrError::OutOfRange { row: 0 }), csr(CsrError::OutOfRange { row: 1 })],
)]
#[case::diagonal_past_a_quarter_of_the_maximum(
    f32::MAX,
    -1.0,
    &[out_of_range(0), out_of_range(1)],
)]
fn f32_out_of_range_input_is_rejected(
    #[case] diagonal: f32,
    #[case] edge: f32,
    #[case] defects: &[Error],
) {
    let observed = f32_pair(diagonal, edge).expect_err("out of f32 range");
    assert_names_a_defect(observed, defects);
}

#[rstest]
#[case::edge_at_the_floor(1.0, -F32_FLOOR)]
#[case::diagonal_at_a_quarter_of_the_maximum(f32::MAX / 4.0, -(f32::MAX / 4.0) / 2.0)]
fn f32_in_range_input_is_admitted(#[case] diagonal: f32, #[case] edge: f32) {
    f32_pair(diagonal, edge).expect("inside f32 range");
}

fn grid_laplacian(rows: usize, cols: usize) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    let (row_ptrs, columns, values) = grid(rows, cols).csr(false);
    let narrow = |indices: Vec<usize>| indices.into_iter().map(|i| i as u32).collect();
    (narrow(row_ptrs), narrow(columns), values)
}

fn widen<I: PrimInt>(indices: &[u32]) -> Vec<I> {
    (indices.iter())
        .map(|&index| I::from(index).expect("fits"))
        .collect()
}

fn bits(factor: &Factor, rhs: &[f64]) -> (String, Vec<u64>) {
    let solution = factor.solve(rhs).expect("solve");
    (
        format!("{factor:?}"),
        solution.iter().map(|x| x.to_bits()).collect(),
    )
}

/// Descending columns force the canonicalizing copy; the `u64` copy is rewritten in its own index type.
#[test]
fn canonical_and_reordered_ingestion_agree_bit_for_bit() {
    let (row_ptrs, columns, values) = grid_laplacian(6, 7);
    let n = (row_ptrs.len() - 1) as u32;
    let (reversed_columns, reversed_values) = reversed_rows(&row_ptrs, &columns, &values);
    let rhs: Vec<f64> = (0..n).map(|i| (i % 5) as f64 - 2.0).collect();

    let canonical = bits(
        &factorize(CsrRef::new(&row_ptrs, &columns, &values, n).unwrap()).expect("grid"),
        &rhs,
    );
    let reordered = CsrRef::new(&row_ptrs, &reversed_columns, &reversed_values, n).unwrap();
    assert_eq!(canonical, bits(&factorize(reordered).expect("grid"), &rhs));

    let (wide_row_ptrs, wide_columns) = (widen::<u64>(&row_ptrs), widen::<u64>(&reversed_columns));
    let wide = CsrRef::new(&wide_row_ptrs, &wide_columns, &reversed_values, n).unwrap();
    assert_eq!(canonical, bits(&factorize(wide).expect("grid"), &rhs));

    let upper = |row_ptrs: &[u32], columns: &[u32], values: &[f64]| {
        let (mut ptrs, mut kept_columns, mut kept_values) = (vec![0u32], Vec::new(), Vec::new());
        for (row, span) in row_ptrs.windows(2).enumerate() {
            for k in span[0] as usize..span[1] as usize {
                if columns[k] as usize >= row {
                    kept_columns.push(columns[k]);
                    kept_values.push(values[k]);
                }
            }
            ptrs.push(kept_columns.len() as u32);
        }
        (ptrs, kept_columns, kept_values)
    };
    for (label, columns, values) in [
        ("canonical", &columns, &values),
        ("reversed", &reversed_columns, &reversed_values),
    ] {
        let (ptrs, kept_columns, kept_values) = upper(&row_ptrs, columns, values);
        let triangle = UpperCsrRef::new(&ptrs, &kept_columns, &kept_values, n).unwrap();
        let triangle = factorize(triangle).expect("grid");
        assert_eq!(canonical, bits(&triangle, &rhs), "{label} upper storage");
    }
}
