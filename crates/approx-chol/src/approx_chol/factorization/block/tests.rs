use super::*;
use crate::approx_chol::factorization::approximate::{EliminationSequence, StepHeader};
use crate::approx_chol::factorization::exact::LowerTriangular;

fn sequence() -> EliminationSequence<f64> {
    EliminationSequence {
        steps: vec![
            StepHeader {
                vertex: 0,
                end: 2,
                pivot_scale: 0.5,
            },
            // An isolated pivot ends where the last step did: no neighbors to close over.
            StepHeader {
                vertex: 1,
                end: 2,
                pivot_scale: 1.0,
            },
        ],
        neighbor_indices: vec![1, 2],
        coefficients: vec![0.2, 0.8],
        uneliminated: 2,
    }
}

fn approx() -> Block<f64> {
    Block::Floating(Cholesky::Approximate(sequence()))
}

/// Two eliminated rows, so its packed factor holds `2 * 3 / 2`.
fn exact() -> Block<f64> {
    Block::Floating(Cholesky::Exact(LowerTriangular {
        values: vec![1.0; 3],
    }))
}

fn seq_of(block: &mut Block<f64>) -> &mut EliminationSequence<f64> {
    match block {
        Block::Floating(Cholesky::Approximate(sequence)) => sequence,
        _ => unreachable!("fixture is floating and approximate"),
    }
}

fn lower_of(block: &mut Block<f64>) -> &mut LowerTriangular<f64> {
    match block {
        Block::Floating(Cholesky::Exact(lower)) => lower,
        _ => unreachable!("fixture is floating and exact"),
    }
}

#[test]
fn valid_fixtures_pass() {
    for (label, block) in [("approximate", approx()), ("exact", exact())] {
        if let Err(error) = block.validate() {
            panic!("{label} fixture is valid: {error}");
        }
    }
}

/// Every variant a block's own cholesky can raise.
#[test]
fn every_block_error_variant_is_reachable() {
    #[allow(clippy::type_complexity)]
    let cases: Vec<(&str, fn() -> Block<f64>, fn(&mut Block<f64>), FactorError)> = vec![
        (
            "pivot vertex bounds",
            approx,
            |b| seq_of(b).steps[0].vertex = 99,
            FactorError::VertexOutOfBounds {
                step: 0,
                vertex: 99,
                n: 3,
            },
        ),
        (
            "neighbor bounds",
            approx,
            |b| seq_of(b).neighbor_indices[0] = 99,
            FactorError::NeighborOutOfBounds {
                step: 0,
                neighbor: 99,
                n: 3,
            },
        ),
        (
            "exact pivot too small to divide by",
            exact,
            |b| lower_of(b).values[0] = 1e-320,
            FactorError::ExactPivotInvalid { index: 0 },
        ),
        (
            "exact off-diagonal squares to infinity",
            exact,
            |b| lower_of(b).values[1] = 1e308,
            FactorError::ExactRowNotRepresentable { row: 1 },
        ),
        (
            "step pivot_scale is not finite",
            approx,
            |b| seq_of(b).steps[0].pivot_scale = f64::INFINITY,
            FactorError::StepValueInvalid { step: 0 },
        ),
        // A negative remainder from overspending shares reads the same as a negative share.
        (
            "solve coefficient is negative",
            approx,
            |b| seq_of(b).coefficients[0] = -0.2,
            FactorError::StepValueInvalid { step: 0 },
        ),
        // A NaN compares false against zero, so nothing but finiteness catches it.
        (
            "solve coefficient is not a number",
            approx,
            |b| seq_of(b).coefficients[1] = f64::NAN,
            FactorError::StepValueInvalid { step: 0 },
        ),
        (
            "uneliminated vertex bounds",
            approx,
            |b| seq_of(b).uneliminated = 99,
            FactorError::UneliminatedVertexInvalid { vertex: 99, n: 3 },
        ),
        (
            "uneliminated vertex is a pivot a step already eliminated",
            approx,
            |b| seq_of(b).uneliminated = 1,
            FactorError::UneliminatedVertexInvalid { vertex: 1, n: 3 },
        ),
        // The slot count shrinks with the steps, so the free vertex falls outside it.
        (
            "steps leave a second vertex uneliminated",
            approx,
            |b| seq_of(b).steps.truncate(1),
            FactorError::UneliminatedVertexInvalid { vertex: 2, n: 2 },
        ),
        (
            "one vertex is eliminated twice, so another never is",
            approx,
            |b| seq_of(b).steps[1].vertex = 0,
            FactorError::VertexEliminatedTwice { step: 1, vertex: 0 },
        ),
        (
            "exact factor shorter than its block",
            exact,
            |b| lower_of(b).values.truncate(2),
            FactorError::ExactFactorLengthInvalid { len: 2 },
        ),
        (
            "exact factor pivot is zero",
            exact,
            |b| lower_of(b).values[0] = 0.0,
            FactorError::ExactPivotInvalid { index: 0 },
        ),
        (
            "exact factor pivot is not finite",
            exact,
            |b| lower_of(b).values[2] = f64::NAN,
            FactorError::ExactPivotInvalid { index: 1 },
        ),
    ];

    for (label, build, corrupt, expected) in cases {
        let mut block = build();
        corrupt(&mut block);
        let error = block
            .validate()
            .expect_err(&format!("{label}: corruption must be rejected"));
        assert_eq!(error, expected, "{label}");
    }
}

/// Three slots, eliminating the two that are not `free` through a chain ending at it.
fn leaving_free(free: u32) -> Cholesky<f64> {
    let [first, second] = match free {
        0 => [1, 2],
        1 => [0, 2],
        _ => [0, 1],
    };
    Cholesky::Approximate(EliminationSequence {
        steps: vec![
            StepHeader {
                vertex: first,
                end: 1,
                pivot_scale: 0.5,
            },
            StepHeader {
                vertex: second,
                end: 2,
                pivot_scale: 0.25,
            },
        ],
        neighbor_indices: vec![second, free],
        coefficients: vec![1.0, 1.0],
        uneliminated: free,
    })
}

#[test]
fn the_gauge_holds_whichever_slot_elimination_left_free() {
    for free in 0..3 {
        let grounded = Block::Grounded(leaving_free(free));
        let mut slots = [1.0, -3.0, 0.5];
        grounded.solve(&mut slots);
        assert_eq!(slots[2], 0.0, "grounded, free slot {free}: {slots:?}");

        let floating = Block::Floating(leaving_free(free));
        let mut slots = [1.0, -3.0, 0.5];
        floating.solve(&mut slots);
        let sum: f64 = slots.iter().sum();
        assert!(sum.abs() < 1e-14, "floating, free slot {free}: {slots:?}");
    }
}

/// The dropped `1.0` is the smaller operand as the addend in one order and as the running sum in the other.
#[test]
fn compensated_sum_recovers_the_term_a_plain_fold_drops() {
    for values in [[1e16, 1.0, -1e16], [1.0, 1e16, -1e16]] {
        assert_eq!(values.iter().sum::<f64>(), 0.0, "plain fold keeps the term");
        assert_eq!(compensated_sum(&values), 1.0);
    }
}
