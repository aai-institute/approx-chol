use super::*;
use crate::approx_chol::factorization::approximate::StepHeader;

/// Two steps over three slots, leaving slot 2.
fn approx() -> Cholesky<f64> {
    Cholesky::Approximate(EliminationSequence {
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
    })
}

/// Two rows over three slots, so the packed factor holds `2 * 3 / 2`.
fn exact() -> Cholesky<f64> {
    Cholesky::Exact(LowerTriangular {
        values: vec![1.0; 3],
    })
}

fn seq_of(cholesky: &mut Cholesky<f64>) -> &mut EliminationSequence<f64> {
    match cholesky {
        Cholesky::Approximate(sequence) => sequence,
        Cholesky::Exact(_) => unreachable!("fixture is approximate"),
    }
}

fn lower_of(cholesky: &mut Cholesky<f64>) -> &mut LowerTriangular<f64> {
    match cholesky {
        Cholesky::Exact(lower) => lower,
        Cholesky::Approximate(_) => unreachable!("fixture is exact"),
    }
}

#[test]
fn valid_fixtures_pass() {
    for (label, cholesky) in [("approximate", approx()), ("exact", exact())] {
        assert_eq!(cholesky.eliminated(), 2, "{label}");
        if let Err(error) = cholesky.validate() {
            panic!("{label} fixture is valid: {error}");
        }
    }
}

/// Every variant a cholesky can raise.
#[test]
fn every_cholesky_error_variant_is_reachable() {
    #[allow(clippy::type_complexity)]
    let cases: Vec<(
        &str,
        fn() -> Cholesky<f64>,
        fn(&mut Cholesky<f64>),
        FactorError,
    )> = vec![
        (
            "pivot vertex bounds",
            approx,
            |c| seq_of(c).steps[0].vertex = 99,
            FactorError::VertexOutOfBounds {
                step: 0,
                vertex: 99,
                n: 3,
            },
        ),
        (
            "uneliminated vertex bounds",
            approx,
            |c| seq_of(c).uneliminated = 99,
            FactorError::UneliminatedVertexInvalid { vertex: 99, n: 3 },
        ),
        (
            "uneliminated vertex is a pivot a step already eliminated",
            approx,
            |c| seq_of(c).uneliminated = 1,
            FactorError::UneliminatedVertexInvalid { vertex: 1, n: 3 },
        ),
        (
            "neighbor bounds",
            approx,
            |c| seq_of(c).neighbor_indices[0] = 99,
            FactorError::NeighborOutOfBounds {
                step: 0,
                neighbor: 99,
                n: 3,
            },
        ),
        (
            "a dropped step leaves the uneliminated vertex past the block",
            approx,
            |c| seq_of(c).steps.truncate(1),
            FactorError::UneliminatedVertexInvalid { vertex: 2, n: 2 },
        ),
        (
            "exact pivot too small to divide by",
            exact,
            |c| lower_of(c).values[0] = 1e-320,
            FactorError::ExactPivotInvalid { index: 0 },
        ),
        (
            "exact off-diagonal squares to infinity",
            exact,
            |c| lower_of(c).values[1] = 1e308,
            FactorError::ExactRowNotRepresentable { row: 1 },
        ),
        (
            "step pivot_scale is not finite",
            approx,
            |c| seq_of(c).steps[0].pivot_scale = f64::INFINITY,
            FactorError::StepValueInvalid { step: 0 },
        ),
        // A share of the pivot is a share of it, and a remainder left negative by shares
        // that overspend the pivot reads the same way here.
        (
            "solve coefficient is negative",
            approx,
            |c| seq_of(c).coefficients[0] = -0.2,
            FactorError::StepValueInvalid { step: 0 },
        ),
        // A NaN compares false against zero, so nothing but finiteness catches it.
        (
            "solve coefficient is not a number",
            approx,
            |c| seq_of(c).coefficients[1] = f64::NAN,
            FactorError::StepValueInvalid { step: 0 },
        ),
        (
            "one vertex is eliminated twice, so another never is",
            approx,
            |c| seq_of(c).steps[1].vertex = 0,
            FactorError::VertexEliminatedTwice { step: 1, vertex: 0 },
        ),
        (
            "exact factor of no triangle's length",
            exact,
            |c| lower_of(c).values.truncate(2),
            FactorError::ExactFactorLengthInvalid { len: 2 },
        ),
        (
            "exact factor pivot is zero",
            exact,
            |c| lower_of(c).values[0] = 0.0,
            FactorError::ExactPivotInvalid { index: 0 },
        ),
        (
            "exact factor pivot is not finite",
            exact,
            |c| lower_of(c).values[2] = f64::NAN,
            FactorError::ExactPivotInvalid { index: 1 },
        ),
    ];

    for (label, build, corrupt, expected) in cases {
        let mut cholesky = build();
        corrupt(&mut cholesky);
        let error = cholesky
            .validate()
            .expect_err(&format!("{label}: corruption must be rejected"));
        assert_eq!(error, expected, "{label}");
    }
}
