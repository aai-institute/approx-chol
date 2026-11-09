use approx_chol::input::InputError;
use approx_chol::{factorize, CsrRef, OwnedCsr};
use rstest::rstest;

use crate::common::solve::Solve;

#[test]
fn owned_csr_routes_factor_alike() {
    let (rp, ci, v) = (
        [0usize, 2, 5, 7],
        [0usize, 1, 0, 1, 2, 1, 2],
        [1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0],
    );
    let owned: OwnedCsr = OwnedCsr::try_from_usize(&rp, &ci, &v, 3).expect("shape");
    let by_input = factorize(&owned).expect("factor via From<&OwnedCsr> for Input");
    let by_ref = factorize(owned.as_csr_ref()).expect("factor via as_csr_ref");
    let wide = CsrRef::new(&rp, &ci, &v, 3).expect("usize view");
    let narrowed = wide.to_owned_u32().expect("narrow");
    let by_narrow = factorize(&narrowed).expect("factor narrowed");
    let b = [1.0, 0.0, -1.0];
    let x = |f: &approx_chol::ApproxInverse| f.solve(&b).expect("solve");
    assert_eq!(x(&by_input), x(&by_ref));
    assert_eq!(x(&by_input), x(&by_narrow));
}

#[test]
fn a_dimension_past_u32_is_too_large() {
    let n = u32::MAX as usize + 1;
    let error = OwnedCsr::<f64, u32>::try_from_usize(&[0], &[], &[], n).expect_err("n past u32");
    assert_eq!(error, InputError::DimensionTooLarge { n });
}

#[rstest]
#[case::row_pointer(&[0, 1, 1 << 32], &[0, 1], "row_ptrs", 2)]
#[case::column(&[0, 1, 2], &[0, 1 << 32], "col_indices", 1)]
fn an_index_past_u32_is_out_of_range_at_its_entry(
    #[case] row_ptrs: &[usize],
    #[case] col_indices: &[usize],
    #[case] parameter: &'static str,
    #[case] entry: usize,
) {
    let values = [1.0, 1.0];
    let error = OwnedCsr::<f64, u32>::try_from_usize(row_ptrs, col_indices, &values, 2)
        .expect_err("an index past u32");
    assert_eq!(error, InputError::IndexOutOfRange { parameter, entry });
}

#[test]
fn a_column_past_u32_does_not_narrow() {
    let (row_ptrs, col_indices, values) = ([0usize, 1, 2], [0usize, 1 << 32], [1.0, 1.0]);
    let wide = CsrRef::new(&row_ptrs, &col_indices, &values, 2).expect("a shape-valid view");
    let expected = InputError::IndexOutOfRange {
        parameter: "col_indices",
        entry: 1,
    };
    assert_eq!(wide.to_owned_u32().err(), Some(expected));
}
