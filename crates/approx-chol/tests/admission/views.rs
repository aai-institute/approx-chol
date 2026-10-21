use approx_chol::input::{
    BipartiteFactorWidth2, CsrRef, FactorWidth1, FactorWidth2, Laplacian, ViewError, Weights,
};
use approx_chol::{factorize, Error, ReduceError};
use rstest::rstest;

use super::assert_names_a_defect;
use super::gram::{dense_csr, matrix, solve_residual};

const PATH_A: [u32; 2] = [0, 1];
const PATH_B: [u32; 2] = [1, 2];

fn tiny() -> f64 {
    f64::MIN_POSITIVE / f64::EPSILON / 2.0
}

fn out_of_range(vertices: &[usize]) -> Vec<Error> {
    (vertices.iter())
        .map(|&vertex| Error::Reduce(ReduceError::OutOfRange { vertex }))
        .collect()
}

fn two_edges_but(found: usize) -> ViewError {
    ViewError::LengthMismatch { expected: 2, found }
}

fn unit_path() -> Laplacian<'static, f64> {
    Laplacian::new(3, &PATH_A, &PATH_B).unwrap()
}

/// Left rows `0, 1` on one right row: the path `0 − 2 − 1`.
fn bipartite_path(weights: Weights<&[f64]>) -> Result<BipartiteFactorWidth2<'_, f64>, ViewError> {
    let left = FactorWidth1::new(2, &[0u32, 1], None)?;
    let right = FactorWidth1::new(1, &[0u32, 0], None)?;
    BipartiteFactorWidth2::new(left, right, weights)
}

#[rstest]
#[case::negative(-1.0)]
#[case::nan(f64::NAN)]
#[case::infinite(f64::INFINITY)]
fn a_weighted_view_rejects_a_negative_or_non_finite_weight_at_its_entry(#[case] weight: f64) {
    let w = [1.0, weight];
    let laplacian = Laplacian::weighted(3, &PATH_A, &PATH_B, &w).err();
    let bipartite = bipartite_path(Weights::Of(&w)).err();
    let expected = Some(ViewError::InvalidWeight { entry: 1 });
    assert_eq!([laplacian, bipartite], [expected; 2]);
}

#[test]
fn a_weighted_view_admits_a_zero_weight() {
    let w = [0.0, 1.0];
    Laplacian::weighted(3, &PATH_A, &PATH_B, &w).expect("a zero weight drops its edge");
    bipartite_path(Weights::Of(&w)).expect("a zero weight drops its column");
}

#[rstest]
#[case::nan(f64::NAN)]
#[case::infinite(f64::NEG_INFINITY)]
fn sqrt_weights_reject_a_non_finite_root_at_its_entry(#[case] root: f64) {
    let error = bipartite_path(Weights::SqrtOf(&[1.0, root])).expect_err("invalid root");
    assert_eq!(error, ViewError::InvalidWeight { entry: 1 });
}

#[test]
fn sqrt_weights_admit_a_negative_root() {
    bipartite_path(Weights::SqrtOf(&[-2.0, 0.0])).expect("a negative root squares to a weight");
}

#[test]
fn a_weight_below_min_positive_over_epsilon_is_out_of_range_at_its_edge() {
    let w = [1.0, tiny()];
    let graph = Laplacian::weighted(3, &PATH_A, &PATH_B, &w).unwrap();
    let error = factorize(graph).expect_err("tiny weight");
    assert_names_a_defect(error, &out_of_range(&[1, 2]));
}

/// Its cell `s·s` underflows to zero, which must not drop the edge.
#[test]
fn a_root_weight_whose_square_underflows_is_out_of_range_at_its_edge() {
    let path = bipartite_path(Weights::SqrtOf(&[1.0, 1e-170])).unwrap();
    let error = factorize(path).expect_err("tiny root");
    assert_names_a_defect(error, &out_of_range(&[1, 2]));
}

#[rstest]
#[case::ends_of_unequal_length(
    Laplacian::<f64>::new(3, &[0, 1], &[1]).err(),
    two_edges_but(1)
)]
#[case::a_weight_per_edge(
    Laplacian::weighted(3, &PATH_A, &PATH_B, &[1.0, 2.0, 3.0]).err(),
    two_edges_but(3)
)]
#[case::end_past_n(
    Laplacian::<f64>::new(3, &[0, 1], &[1, 3]).err(),
    ViewError::IndexOutOfRange { entry: 1 }
)]
#[case::self_loop(
    Laplacian::<f64>::new(3, &[0, 2], &[1, 2]).err(),
    ViewError::RepeatedVertex { entry: 1 }
)]
fn laplacian_new_rejects_mismatched_lengths_and_bad_ends_at_their_entry(
    #[case] observed: Option<ViewError>,
    #[case] expected: ViewError,
) {
    assert_eq!(observed, Some(expected));
}

#[test]
fn laplacian_grounded_rejects_a_ground_per_vertex_short() {
    let short = unit_path()
        .grounded(&[1.0, 1.0])
        .expect_err("one ground per vertex");
    assert_eq!(
        short,
        ViewError::LengthMismatch {
            expected: 3,
            found: 2
        }
    );
}

#[rstest]
#[case::negative(-1.0)]
#[case::nan(f64::NAN)]
#[case::infinite(f64::INFINITY)]
fn laplacian_grounded_rejects_a_bad_ground_at_its_vertex(#[case] ground: f64) {
    let error = unit_path()
        .grounded(&[0.0, 0.0, ground])
        .expect_err("invalid ground");
    assert_eq!(error, ViewError::InvalidGround { vertex: 2 });
}

#[rstest]
#[case::below_min_positive_over_epsilon(tiny())]
#[case::past_a_quarter_of_the_maximum(f64::MAX / 2.0)]
fn an_out_of_range_ground_is_rejected_at_its_vertex(#[case] ground: f64) {
    let ground = [0.0, 0.0, ground];
    let grounded = unit_path().grounded(&ground).expect("a valid ground");
    let error = factorize(grounded).expect_err("ground out of range");
    assert_names_a_defect(error, &out_of_range(&[2]));
}

#[test]
fn factor_width_1_new_rejects_mismatched_values_and_rows_past_n() {
    let mismatch = FactorWidth1::new(3, &[0u32, 1, 2], Some(&[1.0, 2.0][..])).expect_err("values");
    assert_eq!(
        mismatch,
        ViewError::LengthMismatch {
            expected: 3,
            found: 2
        }
    );
    let past = FactorWidth1::<f64>::new(3, &[0u32, 3, 2], None).expect_err("row past n");
    assert_eq!(past, ViewError::IndexOutOfRange { entry: 1 });
    let negative = FactorWidth1::<f64, i64>::new(3, &[0, 1, -1], None).expect_err("negative row");
    assert_eq!(negative, ViewError::IndexOutOfRange { entry: 2 });
}

fn gram(a: &[u32], b: &[u32], x: &[f64], y: &[f64]) -> Option<ViewError> {
    FactorWidth2::new(3, (a, x), (b, y)).err()
}

#[rstest]
#[case::b_one_short(gram(&PATH_A, &[1], &[1.0, 1.0], &[-1.0, -1.0]), two_edges_but(1))]
#[case::x_one_short(gram(&PATH_A, &PATH_B, &[1.0], &[-1.0, -1.0]), two_edges_but(1))]
#[case::y_one_short(gram(&PATH_A, &PATH_B, &[1.0, 1.0], &[1.0]), two_edges_but(1))]
#[case::end_past_n(
    gram(&PATH_A, &[1, 3], &[1.0, 1.0], &[-1.0, -1.0]),
    ViewError::IndexOutOfRange { entry: 1 }
)]
#[case::repeated_vertex(
    gram(&[0, 2], &PATH_B, &[1.0, 1.0], &[-1.0, -1.0]),
    ViewError::RepeatedVertex { entry: 1 }
)]
fn factor_width_2_new_rejects_mismatched_lengths_and_bad_ends_at_their_entry(
    #[case] observed: Option<ViewError>,
    #[case] expected: ViewError,
) {
    assert_eq!(observed, Some(expected));
}

/// The constructor checks only the shape: a loading outside the range is caught where it is summed.
#[rstest]
#[case::nan(f64::NAN)]
#[case::infinite(f64::INFINITY)]
#[case::below_min_positive_over_epsilon(tiny())]
#[case::past_a_quarter_of_the_maximum(f64::MAX)]
fn an_out_of_range_loading_is_rejected_at_its_vertex(#[case] value: f64) {
    let (x, y) = ([1.0, value], [-1.0, -1.0]);
    let gram = FactorWidth2::new(3, (&PATH_A, &x), (&PATH_B, &y)).unwrap();
    let error = factorize(gram).expect_err("loading out of range");
    assert_names_a_defect(error, &out_of_range(&[1, 2]));
}

#[test]
fn bipartite_gram_new_rejects_sides_of_unequal_column_count() {
    let left = FactorWidth1::<f64>::new(2, &[0, 1, 1], None).unwrap();
    let right = FactorWidth1::new(2, &[0, 1], None).unwrap();
    let error = BipartiteFactorWidth2::new(left, right, Weights::Unit).expect_err("column counts");
    assert_eq!(
        error,
        ViewError::LengthMismatch {
            expected: 3,
            found: 2
        }
    );
}

#[test]
fn every_weighted_view_rejects_weights_of_another_column_count() {
    let w = [1.0];
    let errors = [
        Laplacian::weighted(3, &PATH_A, &PATH_B, &w).err(),
        bipartite_path(Weights::Of(&w)).err(),
        bipartite_path(Weights::SqrtOf(&w)).err(),
    ];
    assert_eq!(errors, [Some(two_edges_but(1)); 3]);
}

#[rstest]
#[case::nan(f64::NAN)]
#[case::below_min_positive_over_epsilon(tiny())]
#[case::past_a_quarter_of_the_maximum(f64::MAX)]
fn an_out_of_range_bipartite_loading_is_rejected_at_its_vertex(#[case] value: f64) {
    let values = [1.0, value];
    let left = FactorWidth1::new(2, &[0u32, 1], Some(&values[..])).unwrap();
    let right = FactorWidth1::<f64>::new(1, &[0u32, 0], None).unwrap();
    let gram = BipartiteFactorWidth2::new(left, right, Weights::Unit).unwrap();
    let error = factorize(gram).expect_err("loading out of range");
    assert_names_a_defect(error, &out_of_range(&[1, 2]));
}

/// Its square underflows, but the vertex's other column keeps the diagonal in range.
#[rstest]
#[case::square_just_underflows(1e-162, 1.0)]
#[case::loading_at_the_floor(2.0 * tiny(), 1.0)]
#[case::weighted(1e-162, 0.5)]
fn a_bipartite_loading_whose_square_underflows_beside_an_in_range_column_factors_and_solves(
    #[case] z: f64,
    #[case] w: f64,
) {
    let (values, weights) = ([z, 1.0], [w, 1.0]);
    let left = FactorWidth1::new(1, &[0u32, 0], Some(&values[..])).unwrap();
    let right = FactorWidth1::<f64>::new(2, &[0u32, 1], None).unwrap();
    let gram = BipartiteFactorWidth2::new(left, right, Weights::Of(&weights)).unwrap();
    let factor = factorize(gram).expect("every diagonal and cell is in range");
    let root = w.sqrt();
    solve_residual(
        &factor,
        &matrix(3, &[0, 0], &PATH_B, &[root * z, 1.0], &[root, 1.0]),
    );
}

#[rstest]
#[case::square_just_underflows(1e-162)]
#[case::loading_at_the_floor(2.0 * tiny())]
fn a_gram_loading_whose_square_underflows_beside_an_in_range_column_factors_and_solves(
    #[case] z: f64,
) {
    let (x, y) = ([z, 1.0], [-1.0, -1.0]);
    let gram = FactorWidth2::new(3, (&[0u32, 0], &x), (&PATH_B, &y)).unwrap();
    let factor = factorize(gram).expect("every diagonal and cell is in range");
    solve_residual(&factor, &matrix(3, &[0, 0], &PATH_B, &x, &y));
}

/// Both vertices are in range, but their cell's product underflows below the floor.
#[test]
fn a_gram_cell_whose_product_underflows_is_out_of_range() {
    let (x, y) = ([1e-170, 1.0, 1.0], [1e-170, -1.0, -1.0]);
    let gram = FactorWidth2::new(4, (&[0u32, 0, 1], &x), (&[1, 2, 3], &y)).unwrap();
    let error = factorize(gram).expect_err("cell below the floor");
    assert_names_a_defect(error, &out_of_range(&[0, 1]));
}

/// Its cell underflows too, so the vertex is out of range, not dead.
#[test]
fn a_bipartite_vertex_whose_every_term_underflows_is_out_of_range() {
    let values = [1e-170];
    let side = || FactorWidth1::new(1, &[0u32], Some(&values[..])).unwrap();
    let gram = BipartiteFactorWidth2::new(side(), side(), Weights::Unit).unwrap();
    let error = factorize(gram).expect_err("every term underflows");
    assert_names_a_defect(error, &out_of_range(&[0, 1]));
}

#[test]
fn a_gram_vertex_whose_every_term_underflows_is_out_of_range() {
    let (x, y) = ([1e-170, 1.0], [1e-170, -1.0]);
    let gram = FactorWidth2::new(4, (&[0u32, 2], &x), (&[1, 3], &y)).unwrap();
    let error = factorize(gram).expect_err("every term underflows");
    assert_names_a_defect(error, &out_of_range(&[0, 1]));
}

fn with_diagonal(mut matrix: Vec<Vec<f64>>, diagonal: &[f64]) -> Vec<Vec<f64>> {
    for (i, d) in diagonal.iter().enumerate() {
        matrix[i][i] += d;
    }
    matrix
}

#[rstest]
#[case::balanced(&[0, 1], &[1, 2], &[true, false])]
#[case::frustrated(&[0, 1, 0], &[1, 2, 2], &[true, true, true])]
#[case::cancelling(&[0, 0, 1], &[1, 1, 2], &[true, false, false])]
fn a_grounded_signed_graph_solves_its_matrix(
    #[case] a: &[u32],
    #[case] b: &[u32],
    #[case] positive: &[bool],
) {
    let ground = [0.5, 0.0, 0.25];
    let x = vec![1.0; a.len()];
    let y: Vec<f64> = positive
        .iter()
        .map(|&p| if p { -1.0 } else { 1.0 })
        .collect();
    let matrix = with_diagonal(matrix(3, a, b, &x, &y), &ground);
    let (ptrs, columns, values) = dense_csr(&matrix);
    let csr = CsrRef::new(&ptrs, &columns, &values, 3).unwrap();
    let factor = factorize(csr).expect("a grounded signed graph");
    solve_residual(&factor, &matrix);
}

#[rstest]
#[case::left_past_its_part(&[0, 2], &[0, 1], Some(ViewError::IndexOutOfRange { entry: 1 }))]
#[case::right_past_its_part(&[0, 1], &[1, 2], Some(ViewError::IndexOutOfRange { entry: 1 }))]
#[case::equal_rows_are_distinct_vertices(&[1, 0], &[1, 0], None)]
fn a_bipartite_incidence_checks_each_end_against_its_own_part(
    #[case] a: &[u32],
    #[case] b: &[u32],
    #[case] expected: Option<ViewError>,
) {
    let side = |rows| FactorWidth1::<f64>::new(2, rows, None);
    let found = side(a)
        .and_then(|l| BipartiteFactorWidth2::new(l, side(b)?, Weights::Unit))
        .err();
    assert_eq!(found, expected);
}
