use approx_chol_next::input::{
    BipartiteGram, FactorWidth1, FactorWidth2, Laplacian, SignedGraph, ViewError, Weights,
};
use approx_chol_next::{ReduceError, ScalingPolicy, Sgdd};

use super::assert_names_a_defect;

const PATH_A: [u32; 2] = [0, 1];
const PATH_B: [u32; 2] = [1, 2];

fn tiny() -> f64 {
    f64::MIN_POSITIVE / f64::EPSILON / 2.0
}

fn out_of_range(vertices: &[usize]) -> Vec<ReduceError> {
    (vertices.iter())
        .map(|&vertex| ReduceError::OutOfRange { vertex })
        .collect()
}

#[test]
fn weights_of_rejects_a_negative_or_non_finite_weight_at_its_column() {
    for (label, w) in [
        ("negative", [1.0, -1.0, 1.0]),
        ("NaN", [1.0, f64::NAN, 1.0]),
        ("infinite", [1.0, f64::INFINITY, 1.0]),
    ] {
        let error = Weights::of(&w).expect_err(label);
        assert_eq!(error, ViewError::InvalidWeight { column: 1 }, "{label}");
    }
    Weights::of(&[0.0, 1.0]).expect("a zero weight drops its column");
}

#[test]
fn weights_sqrt_of_rejects_a_non_finite_root_at_its_column() {
    for (label, s) in [
        ("NaN", [1.0, 1.0, f64::NAN]),
        ("infinite", [1.0, 1.0, f64::NEG_INFINITY]),
    ] {
        let error = Weights::sqrt_of(&s).expect_err(label);
        assert_eq!(error, ViewError::InvalidWeight { column: 2 }, "{label}");
    }
    Weights::sqrt_of(&[-2.0, 0.0]).expect("a negative root squares to a weight");
}

#[test]
fn a_weight_below_min_positive_over_epsilon_is_out_of_range_at_its_edge() {
    let w = [1.0, tiny()];
    let graph = Laplacian::new(3, &PATH_A, &PATH_B, Weights::of(&w).unwrap()).unwrap();
    let error = Sgdd::<f64>::try_from(graph).expect_err("tiny weight");
    assert_names_a_defect("tiny weight", error, &out_of_range(&[1, 2]));
}

#[test]
fn laplacian_new_rejects_mismatched_lengths_and_bad_ends_at_their_column() {
    let w = [1.0, 2.0, 3.0];
    let mismatch = |found| ViewError::LengthMismatch { expected: 2, found };
    let cases = [
        (
            "ends of unequal length",
            Laplacian::<f64>::new(3, &[0, 1], &[1], Weights::unit()).err(),
            mismatch(1),
        ),
        (
            "a weight per edge",
            Laplacian::new(3, &PATH_A, &PATH_B, Weights::of(&w).unwrap()).err(),
            mismatch(3),
        ),
        (
            "end past n",
            Laplacian::<f64>::new(3, &[0, 1], &[1, 3], Weights::unit()).err(),
            ViewError::IndexOutOfRange { column: 1 },
        ),
        (
            "self-loop",
            Laplacian::<f64>::new(3, &[0, 2], &[1, 2], Weights::unit()).err(),
            ViewError::RepeatedVertex { column: 1 },
        ),
    ];
    for (label, observed, expected) in cases {
        assert_eq!(observed, Some(expected), "{label}");
    }
}

#[test]
fn laplacian_grounded_rejects_a_bad_ground_at_its_vertex() {
    let graph = Laplacian::<f64>::new(3, &PATH_A, &PATH_B, Weights::unit()).unwrap();
    let short = graph
        .grounded(&[1.0, 1.0])
        .expect_err("one ground per vertex");
    assert_eq!(
        short,
        ViewError::LengthMismatch {
            expected: 3,
            found: 2
        }
    );
    for (label, ground) in [
        ("negative", [0.0, 0.0, -1.0]),
        ("NaN", [0.0, 0.0, f64::NAN]),
        ("infinite", [0.0, 0.0, f64::INFINITY]),
    ] {
        let error = graph.grounded(&ground).expect_err(label);
        assert_eq!(error, ViewError::InvalidGround { vertex: 2 }, "{label}");
    }
}

#[test]
fn an_out_of_range_ground_is_rejected_at_its_vertex() {
    let graph = Laplacian::<f64>::new(3, &PATH_A, &PATH_B, Weights::unit()).unwrap();
    for (label, ground) in [
        ("below MIN_POSITIVE/EPSILON", [0.0, 0.0, tiny()]),
        ("past a quarter of the maximum", [0.0, 0.0, f64::MAX / 2.0]),
    ] {
        let grounded = graph.grounded(&ground).expect(label);
        let error = Sgdd::<f64>::try_from(grounded).expect_err(label);
        assert_names_a_defect(label, error, &out_of_range(&[2]));
    }
}

#[test]
fn signed_graph_new_rejects_mismatched_lengths_and_bad_ends_at_their_column() {
    let signs = [true, false];
    let new = |a: &[u32], b: &[u32], positive: &[bool]| {
        SignedGraph::<f64>::new(3, a, b, positive, Weights::unit()).err()
    };
    let w = [1.0, f64::NAN];
    let mismatch = ViewError::LengthMismatch {
        expected: 2,
        found: 1,
    };
    let cases = [
        ("a sign per edge", new(&PATH_A, &PATH_B, &[true]), mismatch),
        (
            "ends of unequal length",
            new(&PATH_A, &[1], &signs),
            mismatch,
        ),
        (
            "end past n",
            new(&[0, 3], &PATH_B, &signs),
            ViewError::IndexOutOfRange { column: 1 },
        ),
        (
            "self-loop",
            new(&[1, 1], &PATH_B, &signs),
            ViewError::RepeatedVertex { column: 0 },
        ),
        (
            "NaN weight",
            Weights::of(&w).err(),
            ViewError::InvalidWeight { column: 1 },
        ),
    ];
    for (label, observed, expected) in cases {
        assert_eq!(observed, Some(expected), "{label}");
    }
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

#[test]
fn factor_width_2_new_rejects_mismatched_lengths_and_bad_ends_at_their_column() {
    let (x, y, short) = ([1.0, 1.0], [-1.0, -1.0], [1.0]);
    let new = |a: &[u32], b: &[u32], x: &[f64], y: &[f64]| FactorWidth2::new(3, a, b, x, y).err();
    let mismatch = ViewError::LengthMismatch {
        expected: 2,
        found: 1,
    };
    let cases = [
        ("b", new(&PATH_A, &[1], &x, &y), mismatch),
        ("x", new(&PATH_A, &PATH_B, &short, &y), mismatch),
        ("y", new(&PATH_A, &PATH_B, &x, &short), mismatch),
        (
            "end past n",
            new(&PATH_A, &[1, 3], &x, &y),
            ViewError::IndexOutOfRange { column: 1 },
        ),
        (
            "repeated vertex",
            new(&[0, 2], &PATH_B, &x, &y),
            ViewError::RepeatedVertex { column: 1 },
        ),
    ];
    for (label, observed, expected) in cases {
        assert_eq!(observed, Some(expected), "{label}");
    }
}

/// The constructor checks only the shape: a loading outside the range is caught where it is summed.
#[test]
fn an_out_of_range_loading_is_rejected_at_its_vertex() {
    for (label, value) in [
        ("NaN", f64::NAN),
        ("infinite", f64::INFINITY),
        ("below MIN_POSITIVE/EPSILON", tiny()),
        ("past a quarter of the maximum", f64::MAX),
    ] {
        let (x, y) = ([1.0, value], [-1.0, -1.0]);
        let gram = FactorWidth2::new(3, &PATH_A, &PATH_B, &x, &y).unwrap();
        let error = Sgdd::<f64>::from_gram(gram, ScalingPolicy::default()).expect_err(label);
        assert_names_a_defect(label, error, &out_of_range(&[1, 2]));
    }
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

#[test]
fn an_out_of_range_bipartite_loading_is_rejected_at_its_vertex() {
    for (label, value) in [
        ("NaN", f64::NAN),
        ("below MIN_POSITIVE/EPSILON", tiny()),
        ("past a quarter of the maximum", f64::MAX),
    ] {
        let values = [1.0, value];
        let left = FactorWidth1::new(2, &[0u32, 1], Some(&values[..])).unwrap();
        let right = FactorWidth1::<f64>::new(1, &[0u32, 0], None).unwrap();
        let gram = BipartiteGram::new(left, right, Weights::unit()).unwrap();
        let error =
            Sgdd::<f64>::from_bipartite_gram(gram, ScalingPolicy::default()).expect_err(label);
        assert_names_a_defect(label, error, &out_of_range(&[1, 2]));
    }
}
