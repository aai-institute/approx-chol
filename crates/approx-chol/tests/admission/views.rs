use approx_chol::input::{
    BipartiteGram, FactorWidth1, FactorWidth2, Laplacian, SignedGraph, ViewError, Weights,
};
use approx_chol::{factorize, Error, ReduceError};
use rstest::rstest;

use super::assert_names_a_defect;
use super::gram::{matrix, solve_residual};

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
    Laplacian::new(3, &PATH_A, &PATH_B, Weights::unit()).unwrap()
}

#[rstest]
#[case::negative(-1.0)]
#[case::nan(f64::NAN)]
#[case::infinite(f64::INFINITY)]
fn weights_of_rejects_a_negative_or_non_finite_weight_at_its_column(#[case] weight: f64) {
    let error = Weights::of(&[1.0, weight, 1.0]).expect_err("invalid weight");
    assert_eq!(error, ViewError::InvalidWeight { column: 1 });
}

#[test]
fn weights_of_admits_a_zero_weight() {
    Weights::of(&[0.0, 1.0]).expect("a zero weight drops its column");
}

#[rstest]
#[case::nan(f64::NAN)]
#[case::infinite(f64::NEG_INFINITY)]
fn weights_sqrt_of_rejects_a_non_finite_root_at_its_column(#[case] root: f64) {
    let error = Weights::sqrt_of(&[1.0, 1.0, root]).expect_err("invalid root");
    assert_eq!(error, ViewError::InvalidWeight { column: 2 });
}

#[test]
fn weights_sqrt_of_admits_a_negative_root() {
    Weights::sqrt_of(&[-2.0, 0.0]).expect("a negative root squares to a weight");
}

#[test]
fn a_weight_below_min_positive_over_epsilon_is_out_of_range_at_its_edge() {
    let w = [1.0, tiny()];
    let graph = Laplacian::new(3, &PATH_A, &PATH_B, Weights::of(&w).unwrap()).unwrap();
    let error = factorize(graph).expect_err("tiny weight");
    assert_names_a_defect(error, &out_of_range(&[1, 2]));
}

/// Its cell `s·s` underflows to zero, which must not drop the edge.
#[test]
fn a_root_weight_whose_square_underflows_is_out_of_range_at_its_edge() {
    let s = Weights::sqrt_of(&[1.0, 1e-170]).unwrap();
    let laplacian = Laplacian::new(3, &PATH_A, &PATH_B, s).unwrap();
    let signed = SignedGraph::new(3, &PATH_A, &PATH_B, &[true, false], s).unwrap();
    for factor in [factorize(laplacian), factorize(signed)] {
        assert_names_a_defect(factor.expect_err("tiny root"), &out_of_range(&[1, 2]));
    }
}

#[rstest]
#[case::ends_of_unequal_length(
    Laplacian::<f64>::new(3, &[0, 1], &[1], Weights::unit()).err(),
    two_edges_but(1)
)]
#[case::a_weight_per_edge(
    Laplacian::new(3, &PATH_A, &PATH_B, Weights::of(&[1.0, 2.0, 3.0]).unwrap()).err(),
    two_edges_but(3)
)]
#[case::end_past_n(
    Laplacian::<f64>::new(3, &[0, 1], &[1, 3], Weights::unit()).err(),
    ViewError::IndexOutOfRange { column: 1 }
)]
#[case::self_loop(
    Laplacian::<f64>::new(3, &[0, 2], &[1, 2], Weights::unit()).err(),
    ViewError::RepeatedVertex { column: 1 }
)]
fn laplacian_new_rejects_mismatched_lengths_and_bad_ends_at_their_column(
    #[case] observed: Option<ViewError>,
    #[case] expected: ViewError,
) {
    assert_eq!(observed, Some(expected));
}

#[test]
fn laplacian_with_ground_rejects_a_ground_per_vertex_short() {
    let short = unit_path()
        .with_ground(&[1.0, 1.0])
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
fn laplacian_with_ground_rejects_a_bad_ground_at_its_vertex(#[case] ground: f64) {
    let error = unit_path()
        .with_ground(&[0.0, 0.0, ground])
        .expect_err("invalid ground");
    assert_eq!(error, ViewError::InvalidDiagonal { vertex: 2 });
}

#[rstest]
#[case::below_min_positive_over_epsilon(tiny())]
#[case::past_a_quarter_of_the_maximum(f64::MAX / 2.0)]
fn an_out_of_range_ground_is_rejected_at_its_vertex(#[case] ground: f64) {
    let ground = [0.0, 0.0, ground];
    let grounded = unit_path().with_ground(&ground).expect("a valid ground");
    let error = factorize(grounded).expect_err("ground out of range");
    assert_names_a_defect(error, &out_of_range(&[2]));
}

fn signed(a: &[u32], b: &[u32], positive: &[bool]) -> Option<ViewError> {
    SignedGraph::<f64>::new(3, a, b, positive, Weights::unit()).err()
}

#[rstest]
#[case::a_sign_per_edge(signed(&PATH_A, &PATH_B, &[true]), two_edges_but(1))]
#[case::ends_of_unequal_length(signed(&PATH_A, &[1], &[true, false]), two_edges_but(1))]
#[case::end_past_n(
    signed(&[0, 3], &PATH_B, &[true, false]),
    ViewError::IndexOutOfRange { column: 1 }
)]
#[case::self_loop(
    signed(&[1, 1], &PATH_B, &[true, false]),
    ViewError::RepeatedVertex { column: 0 }
)]
#[case::nan_weight(Weights::of(&[1.0, f64::NAN]).err(), ViewError::InvalidWeight { column: 1 })]
fn signed_graph_new_rejects_mismatched_lengths_and_bad_ends_at_their_column(
    #[case] observed: Option<ViewError>,
    #[case] expected: ViewError,
) {
    assert_eq!(observed, Some(expected));
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
    assert_eq!(past, ViewError::IndexOutOfRange { column: 1 });
    let negative = FactorWidth1::<f64, i64>::new(3, &[0, 1, -1], None).expect_err("negative row");
    assert_eq!(negative, ViewError::IndexOutOfRange { column: 2 });
}

fn gram(a: &[u32], b: &[u32], x: &[f64], y: &[f64]) -> Option<ViewError> {
    FactorWidth2::new(3, a, b, x, y, Weights::unit()).err()
}

#[rstest]
#[case::b_one_short(gram(&PATH_A, &[1], &[1.0, 1.0], &[-1.0, -1.0]), two_edges_but(1))]
#[case::x_one_short(gram(&PATH_A, &PATH_B, &[1.0], &[-1.0, -1.0]), two_edges_but(1))]
#[case::y_one_short(gram(&PATH_A, &PATH_B, &[1.0, 1.0], &[1.0]), two_edges_but(1))]
#[case::end_past_n(
    gram(&PATH_A, &[1, 3], &[1.0, 1.0], &[-1.0, -1.0]),
    ViewError::IndexOutOfRange { column: 1 }
)]
#[case::repeated_vertex(
    gram(&[0, 2], &PATH_B, &[1.0, 1.0], &[-1.0, -1.0]),
    ViewError::RepeatedVertex { column: 1 }
)]
fn factor_width_2_new_rejects_mismatched_lengths_and_bad_ends_at_their_column(
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
    let gram = FactorWidth2::new(3, &PATH_A, &PATH_B, &x, &y, Weights::unit()).unwrap();
    let error = factorize(gram).expect_err("loading out of range");
    assert_names_a_defect(error, &out_of_range(&[1, 2]));
}

#[test]
fn bipartite_gram_new_rejects_sides_of_unequal_column_count_and_mismatched_weights() {
    let left = FactorWidth1::<f64>::new(2, &[0, 1, 1], None).unwrap();
    let right = FactorWidth1::<f64>::new(2, &[0, 1], None).unwrap();
    let error = BipartiteGram::new(left, right, Weights::unit()).expect_err("column counts");
    assert_eq!(
        error,
        ViewError::LengthMismatch {
            expected: 3,
            found: 2
        }
    );

    let right = FactorWidth1::<f64>::new(2, &[0, 1, 0], None).unwrap();
    let w = [1.0, 1.0];
    let error = BipartiteGram::new(left, right, Weights::of(&w).unwrap()).expect_err("weights");
    assert_eq!(
        error,
        ViewError::LengthMismatch {
            expected: 3,
            found: 2
        }
    );
}

#[rstest]
#[case::nan(f64::NAN)]
#[case::below_min_positive_over_epsilon(tiny())]
#[case::past_a_quarter_of_the_maximum(f64::MAX)]
fn an_out_of_range_bipartite_loading_is_rejected_at_its_vertex(#[case] value: f64) {
    let values = [1.0, value];
    let left = FactorWidth1::new(2, &[0u32, 1], Some(&values[..])).unwrap();
    let right = FactorWidth1::<f64>::new(1, &[0u32, 0], None).unwrap();
    let gram = BipartiteGram::new(left, right, Weights::unit()).unwrap();
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
    let gram = BipartiteGram::new(left, right, Weights::of(&weights).unwrap()).unwrap();
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
    let gram = FactorWidth2::new(3, &[0u32, 0], &PATH_B, &x, &y, Weights::unit()).unwrap();
    let factor = factorize(gram).expect("every diagonal and cell is in range");
    solve_residual(&factor, &matrix(3, &[0, 0], &PATH_B, &x, &y));
}

#[test]
fn a_weighted_gram_solves_v_w_vt() {
    let (x, y, w) = ([1.0, 2.0], [-1.0, 0.5], [3.0, 0.5]);
    let gram = FactorWidth2::new(3, &PATH_A, &PATH_B, &x, &y, Weights::of(&w).unwrap()).unwrap();
    let factor = factorize(gram).expect("a weighted Gram");
    let root = w.map(f64::sqrt);
    let (x, y) = (
        [root[0] * x[0], root[1] * x[1]],
        [root[0] * y[0], root[1] * y[1]],
    );
    solve_residual(&factor, &matrix(3, &PATH_A, &PATH_B, &x, &y));
}

#[test]
fn factor_width_2_rejects_weights_of_another_column_count() {
    let (x, y, w) = ([1.0, 1.0], [-1.0, -1.0], [1.0]);
    let weights = Weights::of(&w).unwrap();
    let error = FactorWidth2::new(3, &PATH_A, &PATH_B, &x, &y, weights).expect_err("weights");
    assert_eq!(error, two_edges_but(1));
}

fn plus_diagonal(mut matrix: Vec<Vec<f64>>, diagonal: &[f64]) -> Vec<Vec<f64>> {
    for (i, d) in diagonal.iter().enumerate() {
        matrix[i][i] += d;
    }
    matrix
}

/// `A = [[1, −2], [−2, 5]]` needs scaling; read unscaled, `σ_1` would add 1 to `M`'s diagonal, not `f_1²`.
#[test]
fn a_gram_diagonal_enters_the_scaled_surplus_squared() {
    let gram = FactorWidth2::<f64>::new(2, &[0u32], &[1], &[1.0], &[-2.0], Weights::unit())
        .and_then(|gram| gram.with_diagonal(&[0.0, 1.0]))
        .unwrap();
    let x = factorize(gram).unwrap().solve(&[1.0, 1.0]).unwrap();
    assert!(
        (x[0] - 7.0).abs() < 1e-12 && (x[1] - 3.0).abs() < 1e-12,
        "{x:?}"
    );
}

#[rstest]
#[case::grounds_a_floating_path([0.0, 0.5, 0.0, 0.0])]
#[case::is_a_lone_vertex_pivot([0.0, 0.0, 0.0, 2.0])]
fn a_gram_with_a_diagonal_solves_v_vt_plus_it(#[case] diagonal: [f64; 4]) {
    let (x, y) = ([1.0, 2.0], [-1.0, -2.0]);
    let gram = FactorWidth2::new(4, &PATH_A, &PATH_B, &x, &y, Weights::unit())
        .and_then(|gram| gram.with_diagonal(&diagonal))
        .unwrap();
    let factor = factorize(gram).expect("a grounded Gram");
    solve_residual(
        &factor,
        &plus_diagonal(matrix(4, &PATH_A, &PATH_B, &x, &y), &diagonal),
    );
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
    let graph = SignedGraph::new(3, a, b, positive, Weights::unit())
        .and_then(|graph| graph.with_ground(&ground))
        .unwrap();
    let factor = factorize(graph).expect("a grounded signed graph");
    let x = vec![1.0; a.len()];
    let y: Vec<f64> = positive
        .iter()
        .map(|&p| if p { 1.0 } else { -1.0 })
        .collect();
    solve_residual(&factor, &plus_diagonal(matrix(3, a, b, &x, &y), &ground));
}

/// Left rows `0, 1`, right rows `2, 3`; the right side's ridge leaves the left floating alone no more.
#[rstest]
#[case::incidence(None)]
#[case::slopes(Some([2.0, 0.5, 1.0]))]
fn a_bipartite_gram_with_a_diagonal_solves_v_w_vt_plus_it(#[case] slopes: Option<[f64; 3]>) {
    let (rows, cols) = ([0u32, 0, 1], [0u32, 1, 1]);
    let left = FactorWidth1::new(2, &rows, slopes.as_ref().map(|s| &s[..])).unwrap();
    let right = FactorWidth1::<f64>::new(2, &cols, None).unwrap();
    let diagonal = [0.0, 0.0, 0.0, 0.75];
    let gram = BipartiteGram::new(left, right, Weights::unit())
        .and_then(|gram| gram.with_diagonal(&diagonal))
        .unwrap();
    let factor = factorize(gram).expect("a grounded bipartite Gram");
    let x = slopes.unwrap_or([1.0; 3]);
    let heads: Vec<u32> = cols.iter().map(|c| c + 2).collect();
    solve_residual(
        &factor,
        &plus_diagonal(matrix(4, &rows, &heads, &x, &[1.0; 3]), &diagonal),
    );
}

/// One left vertex on two right ones: the block puts the right side first, so `σ` must follow it.
#[test]
fn a_bipartite_incidence_diagonal_follows_the_larger_side_first() {
    let (rows, cols) = ([0u32, 0], [0u32, 1]);
    let left = FactorWidth1::<f64>::new(1, &rows, None).unwrap();
    let right = FactorWidth1::<f64>::new(2, &cols, None).unwrap();
    let diagonal = [1.0, 1.0, 2.0];
    let gram = BipartiteGram::new(left, right, Weights::unit())
        .and_then(|gram| gram.with_diagonal(&diagonal))
        .unwrap();
    let factor = factorize(gram).expect("a grounded star");
    let heads = [1u32, 2];
    solve_residual(
        &factor,
        &plus_diagonal(matrix(3, &rows, &heads, &[1.0; 2], &[1.0; 2]), &diagonal),
    );
}

#[rstest]
#[case::short(&[1.0, 1.0], ViewError::LengthMismatch { expected: 3, found: 2 })]
#[case::negative(&[0.0, -1.0, 0.0], ViewError::InvalidDiagonal { vertex: 1 })]
#[case::nan(&[0.0, 0.0, f64::NAN], ViewError::InvalidDiagonal { vertex: 2 })]
fn every_factor_view_rejects_a_bad_diagonal(#[case] diagonal: &[f64], #[case] expected: ViewError) {
    let (x, y) = ([1.0, 1.0], [-1.0, -1.0]);
    let gram = FactorWidth2::new(3, &PATH_A, &PATH_B, &x, &y, Weights::unit()).unwrap();
    let signed = SignedGraph::new(3, &PATH_A, &PATH_B, &[true, false], Weights::unit()).unwrap();
    let left = FactorWidth1::<f64>::new(1, &[0u32, 0], None).unwrap();
    let right = FactorWidth1::<f64>::new(2, &[0u32, 1], None).unwrap();
    let bipartite = BipartiteGram::new(left, right, Weights::unit()).unwrap();
    let errors = [
        gram.with_diagonal(diagonal).err(),
        signed.with_ground(diagonal).err(),
        bipartite.with_diagonal(diagonal).err(),
        unit_path().with_ground(diagonal).err(),
    ];
    assert_eq!(errors, [Some(expected); 4]);
}

/// Both vertices are in range, but their cell's product underflows below the floor.
#[test]
fn a_gram_cell_whose_product_underflows_is_out_of_range() {
    let (x, y) = ([1e-170, 1.0, 1.0], [1e-170, -1.0, -1.0]);
    let gram = FactorWidth2::new(4, &[0u32, 0, 1], &[1, 2, 3], &x, &y, Weights::unit()).unwrap();
    let error = factorize(gram).expect_err("cell below the floor");
    assert_names_a_defect(error, &out_of_range(&[0, 1]));
}

/// Its cell underflows too, so the vertex is out of range, not dead.
#[test]
fn a_bipartite_vertex_whose_every_term_underflows_is_out_of_range() {
    let values = [1e-170];
    let side = || FactorWidth1::new(1, &[0u32], Some(&values[..])).unwrap();
    let gram = BipartiteGram::new(side(), side(), Weights::unit()).unwrap();
    let error = factorize(gram).expect_err("every term underflows");
    assert_names_a_defect(error, &out_of_range(&[0, 1]));
}

#[test]
fn a_gram_vertex_whose_every_term_underflows_is_out_of_range() {
    let (x, y) = ([1e-170, 1.0], [1e-170, -1.0]);
    let gram = FactorWidth2::new(4, &[0u32, 2], &[1, 3], &x, &y, Weights::unit()).unwrap();
    let error = factorize(gram).expect_err("every term underflows");
    assert_names_a_defect(error, &out_of_range(&[0, 1]));
}
