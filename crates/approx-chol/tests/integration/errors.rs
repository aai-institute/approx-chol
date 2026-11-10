use approx_chol::input::InputError;
use approx_chol::BuildError;
use rstest::rstest;

#[rstest]
#[case::vertex_count_overflow(
    InputError::VertexCountOverflow { n_left: usize::MAX, n_right: 1 },
    &format!("vertex counts {} + 1 overflow usize", usize::MAX)
)]
#[case::length_mismatch(
    InputError::LengthMismatch { parameter: "w", expected: 3, found: 2 },
    "w has length 2, expected 3"
)]
#[case::row_ptrs(
    InputError::RowPtrs { row: 4 },
    "row_ptrs[4] is out of order: the first must be 0, each at most the next"
)]
#[case::index_out_of_range(
    InputError::IndexOutOfRange { parameter: "col_indices", entry: 5 },
    "col_indices[5] is out of range"
)]
#[case::out_of_range(
    InputError::OutOfRange { parameter: "s", entry: 6 },
    "s[6] is outside the representable range"
)]
#[case::negative(
    InputError::Negative { parameter: "surplus", entry: 7 },
    "surplus[7] is negative"
)]
#[case::self_loop(InputError::SelfLoop { edge: 8 }, "edge 8 joins a vertex to itself")]
#[case::repeated_row(InputError::RepeatedRow { column: 9 }, "column 9 names one row twice")]
#[case::not_symmetric(
    InputError::NotSymmetric { row: 10 },
    "row 10 has an entry with no equal mirror"
)]
#[case::outside_triangle(
    InputError::OutsideTriangle { row: 11 },
    "row 11 of an upper triangle has a nonzero below the diagonal"
)]
#[case::diagonal(
    InputError::Diagonal { row: 12 },
    "row 12 has a negative diagonal, a zero one beside an edge, or one past MAX/4"
)]
#[case::not_square(InputError::NotSquare { rows: 13, cols: 14 }, "matrix is not square (13x14)")]
#[case::row_gap(
    InputError::RowGap { row: 15 },
    "row 15 leaves unused slots after its entries"
)]
fn each_input_error_displays_every_field(#[case] error: InputError, #[case] expected: &str) {
    assert_eq!(error.to_string(), expected);
}

#[rstest]
#[case::too_large(BuildError::TooLarge { vertex: 3 }, "the block of vertex 3 passes 1/ε vertices")]
#[case::out_of_range(
    BuildError::OutOfRange { vertex: 4 },
    "a value at vertex 4 is outside the representable range"
)]
fn each_build_error_displays_its_vertex(#[case] error: BuildError, #[case] expected: &str) {
    assert_eq!(error.to_string(), expected);
}
