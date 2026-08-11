use super::*;
use crate::approx_chol::factorization::approximate::{EliminationSequence, StepHeader};
use crate::approx_chol::factorization::exact::LowerTriangular;

fn dim(n: usize) -> BlockDim {
    BlockDim::of(n).expect("fixture dimension is non-zero")
}

fn sequence() -> EliminationSequence<f64> {
    EliminationSequence {
        steps: vec![
            StepHeader {
                vertex: 0,
                end: 1,
                inv_diag: 1.0,
            },
            StepHeader {
                vertex: 1,
                end: 2,
                inv_diag: 1.0,
            },
        ],
        neighbor_indices: vec![1, 2],
        elimination_fractions: vec![1.0, 1.0],
        uneliminated: 2,
    }
}

fn data(cholesky: Cholesky<f64>) -> BlockData<f64> {
    BlockData {
        dim: dim(3),
        anchor: Anchor::Floating,
        cholesky,
    }
}

fn approx() -> BlockData<f64> {
    data(Cholesky::Approximate(sequence()))
}

/// The block solves for `3 - 1` variables, so its packed factor holds `2 * 3 / 2`.
fn exact() -> BlockData<f64> {
    data(Cholesky::Exact(LowerTriangular {
        values: vec![1.0; 3],
    }))
}

fn seq_of(data: &mut BlockData<f64>) -> &mut EliminationSequence<f64> {
    match &mut data.cholesky {
        Cholesky::Approximate(sequence) => sequence,
        Cholesky::Exact(_) => unreachable!("fixture is approximate"),
    }
}

fn lower_of(data: &mut BlockData<f64>) -> &mut LowerTriangular<f64> {
    match &mut data.cholesky {
        Cholesky::Exact(lower) => lower,
        Cholesky::Approximate(_) => unreachable!("fixture is exact"),
    }
}

#[test]
fn valid_fixtures_pass() {
    for (label, data) in [("approximate", approx()), ("exact", exact())] {
        if let Err(error) = Block::try_from(data) {
            panic!("{label} fixture is valid: {error}");
        }
    }
}

/// A dim no payload of any length could pin, so nothing downstream ever gets to sum it.
#[test]
fn a_dim_its_cholesky_cannot_pin_never_becomes_a_block() {
    let mut data = exact();
    data.dim = BlockDim::of(usize::MAX).expect("non-zero");

    assert_eq!(
        Block::try_from(data).expect_err("an unpinned dim must be rejected"),
        FactorError::BlockDimMismatch {
            pinned: 3,
            claimed: usize::MAX,
        }
    );
}

/// Every variant a block's own cholesky can raise.
#[test]
fn every_block_error_variant_is_reachable() {
    #[allow(clippy::type_complexity)]
    let cases: Vec<(
        &str,
        fn() -> BlockData<f64>,
        fn(&mut BlockData<f64>),
        FactorError,
    )> = vec![
        (
            "pivot vertex bounds",
            approx,
            |d| seq_of(d).steps[0].vertex = 99,
            FactorError::VertexOutOfBounds {
                step: 0,
                vertex: 99,
                n: 3,
            },
        ),
        (
            "neighbor bounds",
            approx,
            |d| seq_of(d).neighbor_indices[0] = 99,
            FactorError::NeighborOutOfBounds {
                step: 0,
                neighbor: 99,
                n: 3,
            },
        ),
        (
            "exact pivot too small to divide by",
            exact,
            |d| lower_of(d).values[0] = 1e-320,
            FactorError::ExactPivotInvalid { index: 0 },
        ),
        (
            "exact off-diagonal squares to infinity",
            exact,
            |d| lower_of(d).values[1] = 1e308,
            FactorError::ExactRowNotRepresentable { row: 1 },
        ),
        (
            "step inv_diag is not finite",
            approx,
            |d| seq_of(d).steps[0].inv_diag = f64::INFINITY,
            FactorError::StepValueInvalid { step: 0 },
        ),
        (
            "elimination fraction is not a proportion",
            approx,
            |d| seq_of(d).elimination_fractions[0] = 2.0,
            FactorError::StepValueInvalid { step: 0 },
        ),
        (
            "uneliminated vertex bounds",
            approx,
            |d| seq_of(d).uneliminated = 99,
            FactorError::UneliminatedVertexInvalid { vertex: 99, n: 3 },
        ),
        (
            "uneliminated vertex is a pivot a step already eliminated",
            approx,
            |d| seq_of(d).uneliminated = 1,
            FactorError::UneliminatedVertexInvalid { vertex: 1, n: 3 },
        ),
        (
            "steps leave a second vertex uneliminated",
            approx,
            |d| {
                seq_of(d).steps.truncate(1);
            },
            FactorError::BlockDimMismatch {
                pinned: 2,
                claimed: 3,
            },
        ),
        (
            "one vertex is eliminated twice, so another never is",
            approx,
            |d| seq_of(d).steps[1].vertex = 0,
            FactorError::VertexEliminatedTwice { step: 1, vertex: 0 },
        ),
        (
            "exact factor shorter than its block",
            exact,
            |d| lower_of(d).values.truncate(2),
            FactorError::ExactFactorLengthInvalid { len: 2 },
        ),
        (
            "exact factor pivot is zero",
            exact,
            |d| lower_of(d).values[0] = 0.0,
            FactorError::ExactPivotInvalid { index: 0 },
        ),
        (
            "exact factor pivot is not finite",
            exact,
            |d| lower_of(d).values[2] = f64::NAN,
            FactorError::ExactPivotInvalid { index: 1 },
        ),
    ];

    for (label, build, corrupt, expected) in cases {
        let mut data = build();
        corrupt(&mut data);
        let error =
            Block::try_from(data).expect_err(&format!("{label}: corruption must be rejected"));
        assert_eq!(error, expected, "{label}");
    }
}

/// The factorization builder is trusted with its own dims only because this fires when it
/// is wrong.
#[cfg(debug_assertions)]
#[test]
#[should_panic = "assertion"]
fn a_block_built_around_a_cholesky_that_does_not_pin_its_dim_panics() {
    Block::new(
        dim(3),
        Anchor::Floating,
        Cholesky::Exact(LowerTriangular {
            values: vec![1.0; 2],
        }),
    );
}
