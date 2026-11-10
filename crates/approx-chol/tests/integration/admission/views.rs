use approx_chol::input::{
    BipartiteFactorWidth2, CsrRef, FactorWidth1, FactorWidth2, InputError, Laplacian, Sddm,
};
use approx_chol::{
    factorize, factorize_with, ApproxInverse, BuildError, Config, Real, ScalingPolicy,
};
use rstest::rstest;

use crate::common::bits::bits;
use crate::common::dense_residual::{apply, solve_residual};
use crate::common::gram::{dense_csr, matrix};
use crate::common::out_of_range::FLOOR;
use crate::common::solve::Solve;

use super::assert_names_a_defect;

const PATH_A: [u32; 2] = [0, 1];
const PATH_B: [u32; 2] = [1, 2];

fn unit_path() -> Laplacian<'static, f64> {
    Laplacian::new(3, &PATH_A, &PATH_B).unwrap()
}

/// Left rows `0, 1` on one right row, columns scaled by `s`: the path `0 − 2 − 1`.
fn scaled_bipartite_path(s: &[f64]) -> Result<BipartiteFactorWidth2<'_, f64>, InputError> {
    let left = FactorWidth1::new(2, &[0u32, 1])?;
    let right = FactorWidth1::new(1, &[0u32, 0])?;
    BipartiteFactorWidth2::scaled(left, right, s)
}

#[rstest]
#[case::negative(-1.0, InputError::Negative { parameter: "w", entry: 1 })]
#[case::nan(f64::NAN, InputError::OutOfRange { parameter: "w", entry: 1 })]
#[case::infinite(f64::INFINITY, InputError::OutOfRange { parameter: "w", entry: 1 })]
fn a_weighted_laplacian_rejects_a_negative_or_non_finite_weight_at_its_entry(
    #[case] weight: f64,
    #[case] expected: InputError,
) {
    let w = [1.0, weight];
    let error = Laplacian::weighted(3, &PATH_A, &PATH_B, &w).expect_err("invalid weight");
    assert_eq!(error, expected);
}

#[test]
fn a_weighted_laplacian_admits_a_zero_weight() {
    let w = [0.0, 1.0];
    Laplacian::weighted(3, &PATH_A, &PATH_B, &w).expect("a zero weight drops its edge");
}

#[rstest]
#[case::nan(f64::NAN)]
#[case::infinite(f64::NEG_INFINITY)]
#[case::below_floor(FLOOR / 2.0)]
#[case::negative_below_floor(-FLOOR / 2.0)]
fn a_scaled_bipartite_rejects_an_out_of_range_scale_at_its_entry(#[case] scale: f64) {
    let error = scaled_bipartite_path(&[1.0, scale]).expect_err("invalid scale");
    assert_eq!(
        error,
        InputError::OutOfRange {
            parameter: "s",
            entry: 1
        }
    );
}

#[test]
fn a_scaled_bipartite_admits_a_negative_or_zero_scale() {
    scaled_bipartite_path(&[-2.0, 0.0]).expect("a zero scale drops its column");
}

#[test]
fn a_weight_below_min_positive_over_epsilon_is_out_of_range_at_its_edge() {
    let w = [1.0, FLOOR / 2.0];
    let error = Laplacian::weighted(3, &PATH_A, &PATH_B, &w).expect_err("tiny weight");
    assert_eq!(
        error,
        InputError::OutOfRange {
            parameter: "w",
            entry: 1
        }
    );
}

/// Its cell `s·s` underflows to zero, which must not drop the column.
#[test]
fn a_scale_whose_square_underflows_is_out_of_range_at_its_column() {
    let path = scaled_bipartite_path(&[1.0, 1e-170]).unwrap();
    let error = factorize(path).expect_err("tiny scale");
    assert_names_a_defect(
        error,
        &[
            BuildError::OutOfRange { vertex: 1 },
            BuildError::OutOfRange { vertex: 2 },
        ],
    );
}

#[rstest]
#[case::ends_of_unequal_length(
    Laplacian::<f64>::new(3, &[0, 1], &[1]).err(),
    InputError::LengthMismatch { parameter: "v", expected: 2, found: 1 }
)]
#[case::a_weight_per_edge(
    Laplacian::weighted(3, &PATH_A, &PATH_B, &[1.0, 2.0, 3.0]).err(),
    InputError::LengthMismatch { parameter: "w", expected: 2, found: 3 }
)]
#[case::end_past_n(
    Laplacian::<f64>::new(3, &[0, 1], &[1, 3]).err(),
    InputError::IndexOutOfRange { parameter: "v", entry: 1 }
)]
#[case::start_past_n(
    Laplacian::<f64>::new(3, &[0, 3], &[1, 2]).err(),
    InputError::IndexOutOfRange { parameter: "u", entry: 1 }
)]
#[case::self_loop(
    Laplacian::<f64>::new(3, &[0, 2], &[1, 2]).err(),
    InputError::SelfLoop { edge: 1 }
)]
fn laplacian_new_rejects_mismatched_lengths_and_bad_ends_at_their_entry(
    #[case] observed: Option<InputError>,
    #[case] expected: InputError,
) {
    assert_eq!(observed, Some(expected));
}

#[test]
fn sddm_new_rejects_a_surplus_per_vertex_short() {
    let short = Sddm::new(unit_path(), &[1.0, 1.0]).expect_err("one surplus per vertex");
    assert_eq!(
        short,
        InputError::LengthMismatch {
            parameter: "surplus",
            expected: 3,
            found: 2
        }
    );
}

#[rstest]
#[case::negative(-1.0, InputError::Negative { parameter: "surplus", entry: 2 })]
#[case::nan(f64::NAN, InputError::OutOfRange { parameter: "surplus", entry: 2 })]
#[case::infinite(f64::INFINITY, InputError::OutOfRange { parameter: "surplus", entry: 2 })]
fn sddm_new_rejects_a_bad_surplus_at_its_vertex(
    #[case] surplus: f64,
    #[case] expected: InputError,
) {
    let error = Sddm::new(unit_path(), &[1.0, 0.0, surplus]).expect_err("invalid surplus");
    assert_eq!(error, expected);
}

/// A `6 × 6` grid and a triangle, weights cycling `1..=5`, the grid past the exact arm, and an isolated vertex: `n = 40`.
fn grid_triangle_and_lone() -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    let right = (0..36u32).filter(|v| v % 6 < 5).map(|v| (v, v + 1));
    let down = (0..30u32).map(|v| (v, v + 6));
    let triangle = [(36, 37), (37, 38), (36, 38)];
    let (u, v): (Vec<u32>, Vec<u32>) = right.chain(down).chain(triangle).unzip();
    let w = (0..u.len()).map(|k| 1.0 + (k % 5) as f64).collect();
    (u, v, w)
}

#[rstest]
#[case::zero(0.0)]
#[case::negative_zero(-0.0)]
fn an_all_zero_sddm_factors_bit_for_bit_as_its_laplacian(#[case] zero: f64) {
    let (u, v, w) = grid_triangle_and_lone();
    let laplacian = Laplacian::weighted(40, &u, &v, &w).unwrap();
    let surplus = [0.0, zero].repeat(20);
    let sddm = Sddm::new(laplacian, &surplus).expect("an all-zero surplus is admitted");
    let (sddm, laplacian) = (factorize(sddm).unwrap(), factorize(laplacian).unwrap());
    assert_eq!(format!("{sddm:?}"), format!("{laplacian:?}"));
    let b: Vec<f64> = (0..40).map(|i| [1.0, -2.0, 0.5, 0.5][i % 4]).collect();
    let bits = |factor: &ApproxInverse<f64>| bits(&factor.solve(&b).unwrap());
    assert_eq!(bits(&sddm), bits(&laplacian));
}

#[rstest]
#[case::past_a_quarter_of_the_maximum([1.0, 1.0], f64::MAX / 2.0)]
fn an_out_of_range_ground_is_rejected_at_its_vertex(
    #[case] weights: [f64; 2],
    #[case] ground: f64,
) {
    let path = Laplacian::weighted(3, &PATH_A, &PATH_B, &weights).unwrap();
    let ground = [0.0, 0.0, ground];
    let grounded = Sddm::new(path, &ground).expect("a valid surplus");
    let error = factorize(grounded).expect_err("ground out of range");
    assert_names_a_defect(error, &[BuildError::OutOfRange { vertex: 2 }]);
}

#[rstest]
#[case::within_rounding([1.0, 1.0])]
#[case::past_rounding([1.0, FLOOR])]
fn a_stated_ground_below_the_floor_is_rejected_at_construction(#[case] weights: [f64; 2]) {
    let graph = Laplacian::weighted(3, &PATH_A, &PATH_B, &weights).unwrap();
    let surplus = [0.0, 0.0, FLOOR / 2.0];
    let error = Sddm::new(graph, &surplus).expect_err("stated surplus below the floor");
    assert_eq!(
        error,
        InputError::OutOfRange {
            parameter: "surplus",
            entry: 2
        }
    );
}

#[test]
fn factor_width_1_new_rejects_mismatched_values_and_rows_past_n() {
    let mismatch = FactorWidth1::valued(3, &[0u32, 1, 2], &[1.0, 2.0][..]).expect_err("values");
    assert_eq!(
        mismatch,
        InputError::LengthMismatch {
            parameter: "values",
            expected: 3,
            found: 2
        }
    );
    let past = FactorWidth1::<f64>::new(3, &[0u32, 3, 2]).expect_err("row past n");
    assert_eq!(
        past,
        InputError::IndexOutOfRange {
            parameter: "rows",
            entry: 1
        }
    );
    let below = FactorWidth1::<f64, i64>::new(3, &[0, 1, -1]).expect_err("negative row");
    assert_eq!(
        below,
        InputError::IndexOutOfRange {
            parameter: "rows",
            entry: 2
        }
    );
}

fn gram(a: &[u32], b: &[u32], x: &[f64], y: &[f64]) -> Option<InputError> {
    FactorWidth2::new(3, (a, x), (b, y)).err()
}

#[rstest]
#[case::b_one_short(gram(&PATH_A, &[1], &[1.0, 1.0], &[-1.0, -1.0]), InputError::LengthMismatch { parameter: "b", expected: 2, found: 1 })]
#[case::x_one_short(gram(&PATH_A, &PATH_B, &[1.0], &[-1.0, -1.0]), InputError::LengthMismatch { parameter: "x", expected: 2, found: 1 })]
#[case::y_one_short(gram(&PATH_A, &PATH_B, &[1.0, 1.0], &[1.0]), InputError::LengthMismatch { parameter: "y", expected: 2, found: 1 })]
#[case::end_past_n(
    gram(&PATH_A, &[1, 3], &[1.0, 1.0], &[-1.0, -1.0]),
    InputError::IndexOutOfRange { parameter: "b", entry: 1 }
)]
#[case::start_past_n(
    gram(&[0, 3], &PATH_B, &[1.0, 1.0], &[-1.0, -1.0]),
    InputError::IndexOutOfRange { parameter: "a", entry: 1 }
)]
#[case::repeated_row(
    gram(&[0, 2], &PATH_B, &[1.0, 1.0], &[-1.0, -1.0]),
    InputError::RepeatedRow { column: 1 }
)]
fn factor_width_2_new_rejects_mismatched_lengths_and_bad_ends_at_their_entry(
    #[case] observed: Option<InputError>,
    #[case] expected: InputError,
) {
    assert_eq!(observed, Some(expected));
}

#[rstest]
#[case::nan(f64::NAN)]
#[case::infinite(f64::INFINITY)]
#[case::below_min_positive_over_epsilon(FLOOR / 2.0)]
fn an_out_of_range_loading_is_rejected_at_construction(#[case] value: f64) {
    let (x, y) = ([1.0, value], [-1.0, -1.0]);
    let error = FactorWidth2::new(3, (&PATH_A, &x), (&PATH_B, &y)).unwrap_err();
    assert_eq!(
        error,
        InputError::OutOfRange {
            parameter: "x",
            entry: 1
        }
    );
    let error = FactorWidth2::new(3, (&PATH_A, &y), (&PATH_B, &x)).unwrap_err();
    assert_eq!(
        error,
        InputError::OutOfRange {
            parameter: "y",
            entry: 1
        }
    );
}

#[test]
fn a_gram_diagonal_magnitude_is_checked_at_build() {
    let (x, y) = ([1.0, f64::MAX], [-1.0, -1.0]);
    let gram = FactorWidth2::new(3, (&PATH_A, &x), (&PATH_B, &y)).unwrap();
    let error = factorize(gram).expect_err("derived diagonal out of range");
    assert_names_a_defect(
        error,
        &[
            BuildError::OutOfRange { vertex: 1 },
            BuildError::OutOfRange { vertex: 2 },
        ],
    );
}

#[test]
fn bipartite_gram_new_rejects_sides_of_unequal_column_count() {
    let left = FactorWidth1::<f64>::new(2, &[0, 1, 1]).unwrap();
    let right = FactorWidth1::new(2, &[0, 1]).unwrap();
    let error = BipartiteFactorWidth2::new(left, right).expect_err("column counts");
    assert_eq!(
        error,
        InputError::LengthMismatch {
            parameter: "right",
            expected: 3,
            found: 2
        }
    );
}

#[test]
fn bipartite_vertex_counts_must_fit_together() {
    let side = |n| FactorWidth1::<f64>::new(n, &[]).unwrap();
    let expected = InputError::VertexCountOverflow {
        n_left: usize::MAX,
        n_right: 1,
    };
    assert_eq!(
        BipartiteFactorWidth2::new(side(usize::MAX), side(1)).unwrap_err(),
        expected
    );
    assert_eq!(
        BipartiteFactorWidth2::scaled(side(usize::MAX), side(1), &[]).unwrap_err(),
        expected
    );
    BipartiteFactorWidth2::new(side(usize::MAX - 1), side(1)).unwrap();
    BipartiteFactorWidth2::scaled(side(usize::MAX - 1), side(1), &[]).unwrap();
}

#[test]
fn weights_or_scales_of_another_column_count_are_rejected() {
    let w = [1.0];
    let errors = [
        Laplacian::weighted(3, &PATH_A, &PATH_B, &w).err(),
        scaled_bipartite_path(&w).err(),
    ];
    assert_eq!(
        errors,
        [
            Some(InputError::LengthMismatch {
                parameter: "w",
                expected: 2,
                found: 1
            }),
            Some(InputError::LengthMismatch {
                parameter: "s",
                expected: 2,
                found: 1
            })
        ]
    );
}

#[rstest]
#[case::nan(f64::NAN)]
#[case::infinite(f64::INFINITY)]
#[case::below_min_positive_over_epsilon(FLOOR / 2.0)]
fn an_out_of_range_bipartite_loading_is_rejected_at_construction(#[case] value: f64) {
    let values = [1.0, value];
    let error = FactorWidth1::valued(2, &[0u32, 1], &values).unwrap_err();
    assert_eq!(
        error,
        InputError::OutOfRange {
            parameter: "values",
            entry: 1
        }
    );
}

#[test]
fn a_bipartite_gram_diagonal_magnitude_is_checked_at_build() {
    let values = [1.0, f64::MAX];
    let left = FactorWidth1::valued(2, &[0u32, 1], &values).unwrap();
    let right = FactorWidth1::<f64>::new(1, &[0u32, 0]).unwrap();
    let gram = BipartiteFactorWidth2::new(left, right).unwrap();
    let error = factorize(gram).expect_err("derived diagonal out of range");
    assert_names_a_defect(
        error,
        &[
            BuildError::OutOfRange { vertex: 1 },
            BuildError::OutOfRange { vertex: 2 },
        ],
    );
}

/// Its square underflows, but the vertex's other column keeps the diagonal in range.
#[rstest]
#[case::square_just_underflows(1e-162, 1.0)]
#[case::loading_at_the_floor(FLOOR, 1.0)]
#[case::scaled(1e-162, 0.5f64.sqrt())]
fn a_bipartite_loading_whose_square_underflows_beside_an_in_range_column_factors_and_solves(
    #[case] z: f64,
    #[case] s: f64,
) {
    let (values, scales) = ([z, 1.0], [s, 1.0]);
    let left = FactorWidth1::valued(1, &[0u32, 0], &values[..]).unwrap();
    let right = FactorWidth1::<f64>::new(2, &[0u32, 1]).unwrap();
    let gram = BipartiteFactorWidth2::scaled(left, right, &scales).unwrap();
    let factor = factorize(gram).expect("every diagonal and cell is in range");
    solve_residual(
        &factor,
        &matrix(3, &[0, 0], &PATH_B, &[s * z, 1.0], &[s, 1.0]),
    );
}

#[rstest]
#[case::square_just_underflows(1e-162)]
#[case::loading_at_the_floor(FLOOR)]
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
    assert_names_a_defect(
        error,
        &[
            BuildError::OutOfRange { vertex: 0 },
            BuildError::OutOfRange { vertex: 1 },
        ],
    );
}

/// Its cell underflows too, so the vertex is out of range, not dead.
#[test]
fn a_bipartite_vertex_whose_every_term_underflows_is_out_of_range() {
    let values = [1e-170];
    let side = || FactorWidth1::valued(1, &[0u32], &values[..]).unwrap();
    let gram = BipartiteFactorWidth2::new(side(), side()).unwrap();
    let error = factorize(gram).expect_err("every term underflows");
    assert_names_a_defect(
        error,
        &[
            BuildError::OutOfRange { vertex: 0 },
            BuildError::OutOfRange { vertex: 1 },
        ],
    );
}

#[test]
fn a_gram_vertex_whose_every_term_underflows_is_out_of_range() {
    let (x, y) = ([1e-170, 1.0], [1e-170, -1.0]);
    let gram = FactorWidth2::new(4, (&[0u32, 2], &x), (&[1, 3], &y)).unwrap();
    let error = factorize(gram).expect_err("every term underflows");
    assert_names_a_defect(
        error,
        &[
            BuildError::OutOfRange { vertex: 0 },
            BuildError::OutOfRange { vertex: 1 },
        ],
    );
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
    let csr = CsrRef::new(3, &ptrs, &columns, &values).unwrap();
    let factor = factorize(csr).expect("a grounded signed graph");
    solve_residual(&factor, &matrix);
}

#[rstest]
#[case::left_past_its_part(&[0, 2], &[0, 1], Some(InputError::IndexOutOfRange { parameter: "rows", entry: 1 }))]
#[case::right_past_its_part(&[0, 1], &[1, 2], Some(InputError::IndexOutOfRange { parameter: "rows", entry: 1 }))]
#[case::equal_rows_are_distinct_vertices(&[1, 0], &[1, 0], None)]
fn a_bipartite_incidence_checks_each_end_against_its_own_part(
    #[case] a: &[u32],
    #[case] b: &[u32],
    #[case] expected: Option<InputError>,
) {
    let side = |rows| FactorWidth1::<f64>::new(2, rows);
    let found = side(a)
        .and_then(|l| BipartiteFactorWidth2::new(l, side(b)?))
        .err();
    assert_eq!(found, expected);
}

/// A star whose spokes `b` each carry `x = s, y = ∓k s` twice, cancelling, and `x = y = 1` once: every row is in range, the ground's `Σs ≈ 20·MAX`.
fn overflowing_ground_star(order: fn(usize) -> usize) -> (Vec<u32>, Vec<u32>, Vec<f64>, Vec<f64>) {
    let (s, k) = ((f64::MAX / 1000.0).sqrt(), 10.0);
    let columns = (0..300).map(order);
    let b: Vec<u32> = columns.clone().map(|c| (c % 100 + 1) as u32).collect();
    let a = vec![0u32; b.len()];
    let x: Vec<f64> = columns
        .clone()
        .map(|c| if c < 200 { s } else { 1.0 })
        .collect();
    let y: Vec<f64> = columns
        .map(|c| match c {
            0..100 => -k * s,
            100..200 => k * s,
            _ => 1.0,
        })
        .collect();
    (a, b, x, y)
}

/// Blocked, the hub's running surplus sum passes `MAX` before its cancelling columns arrive.
#[rstest]
#[case::blocked(|c| c)]
#[case::blocked_spokes_reversed(|c| c / 100 * 100 + 99 - c % 100)]
fn a_gram_whose_running_surplus_sum_overflows_is_rejected_not_a_panic(
    #[case] order: fn(usize) -> usize,
) {
    let (a, b, x, y) = overflowing_ground_star(order);
    let gram = FactorWidth2::new(101, (&a, &x), (&b, &y)).unwrap();
    let outcome = std::panic::catch_unwind(|| factorize(gram).map(|_| ()));
    let error = outcome
        .expect("no panic")
        .expect_err("a running sum overflows");
    assert!(matches!(error, BuildError::OutOfRange { .. }));
}

/// Interleaved, no running sum overflows; the ground is never a pivot, dense or approximate, so its `Σs` past `MAX` is never formed; a link between two spokes frustrates the star, whose cover grounds `2Σs`.
#[rstest]
fn a_gram_whose_ground_total_passes_the_maximum_solves(
    #[values(0, 101)] exact_max_dim: usize,
    #[values(false, true)] frustrated: bool,
) {
    let (mut a, mut b, mut x, mut y) = overflowing_ground_star(|c| c % 3 * 100 + c / 3);
    if frustrated {
        a.push(1);
        b.push(2);
        x.push(1.0);
        y.push(1.0);
    }
    let gram = FactorWidth2::new(101, (&a, &x), (&b, &y)).unwrap();
    let config = Config {
        exact_max_dim,
        ..Config::default()
    };
    let factor: ApproxInverse =
        factorize_with(gram, config, ScalingPolicy::default()).expect("in range row by row");
    solve_residual(&factor, &matrix(101, &a, &b, &x, &y));
}

/// `K₃₀` of weight `MAX/1000`, ten vertices grounded by `MAX/5`: the ground's degree, 10 against 29, puts it first in minimum degree, and its `Σs = 2 MAX` with no weight `1/ε` apart.
#[test]
fn a_low_degree_ground_whose_total_passes_the_maximum_solves() {
    let (n, w, s) = (30, f64::MAX / 1000.0, f64::MAX / 5.0);
    let matrix: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let ground = if i % 3 == 0 { s } else { 0.0 };
            (0..n)
                .map(|j| {
                    if i == j {
                        (n - 1) as f64 * w + ground
                    } else {
                        -w
                    }
                })
                .collect()
        })
        .collect();
    let (ptrs, columns, values) = dense_csr(&matrix);
    let csr = CsrRef::new(n, &ptrs, &columns, &values).unwrap();
    let config = Config {
        exact_max_dim: 0,
        ..Config::default()
    };
    let factor: ApproxInverse =
        factorize_with(csr, config, ScalingPolicy::default()).expect("in range row by row");
    // Sampled, one apply is a preconditioner, not a solve: finite, and residual 2e-2 at seed 0.
    let exact: Vec<f64> = (0..n).map(|i| (i % 5) as f64 - 2.0).collect();
    let rhs = apply(&matrix, &exact);
    let x = factor.solve(&rhs).expect("solve");
    assert!(x.iter().all(|v| v.is_finite()), "{x:?}");
    let norm = |v: &[f64]| v.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let residual: Vec<f64> = (apply(&matrix, &x).iter().zip(&rhs))
        .map(|(r, b)| r - b)
        .collect();
    let relative = norm(&residual) / norm(&rhs);
    assert!(relative < 0.5, "{relative:e}");
}

#[test]
fn stated_values_use_the_input_scalars_range_floor() {
    check_stated_floor::<f32>();
    check_stated_floor::<f64>();
}

fn check_stated_floor<T: Real>() {
    let floor = T::min_positive_value() / T::epsilon();
    let two = T::one() + T::one();
    for value in [
        T::zero(),
        -T::zero(),
        floor,
        -floor,
        floor / two,
        -floor / two,
    ] {
        let values = [T::one(), value];
        let expected = if value != T::zero() && value.abs() < floor {
            Some(InputError::OutOfRange {
                parameter: "values",
                entry: 1,
            })
        } else {
            None
        };
        assert_eq!(FactorWidth1::valued(2, &[0u32, 1], &values).err(), expected);
        for (parameter, found) in [
            (
                "x",
                FactorWidth2::new(3, (&PATH_A, &values), (&PATH_B, &[T::one(); 2])).err(),
            ),
            (
                "y",
                FactorWidth2::new(3, (&PATH_A, &[T::one(); 2]), (&PATH_B, &values)).err(),
            ),
            (
                "s",
                BipartiteFactorWidth2::scaled(
                    FactorWidth1::new(2, &[0u32, 1]).unwrap(),
                    FactorWidth1::new(1, &[0u32, 0]).unwrap(),
                    &values,
                )
                .err(),
            ),
        ] {
            assert_eq!(
                found,
                expected.map(|_| InputError::OutOfRange {
                    parameter,
                    entry: 1
                })
            );
        }
        if value >= T::zero() {
            assert_eq!(
                Laplacian::weighted(3, &PATH_A, &PATH_B, &values).err(),
                expected.map(|_| InputError::OutOfRange {
                    parameter: "w",
                    entry: 1
                })
            );
            assert_eq!(
                Sddm::new(Laplacian::<T>::new(2, &[0], &[1]).unwrap(), &values).err(),
                expected.map(|_| InputError::OutOfRange {
                    parameter: "surplus",
                    entry: 1
                })
            );
        }
    }
}
