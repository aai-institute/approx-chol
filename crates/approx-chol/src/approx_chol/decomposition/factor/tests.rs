use super::super::sequence::StepHeader;
use super::*;

/// Minimal structurally-valid factor: n=3, one step eliminating vertex 0
/// and splitting its weight to neighbors 1 and 2.
fn valid_factor() -> Factor<f64> {
    Factor {
        n: 3,
        original_n: 3,
        sequence: EliminationSequence {
            steps: vec![StepHeader {
                vertex: 0,
                end: 2,
                inv_diag: 1.0,
            }],
            neighbor_indices: vec![1, 2],
            elimination_fractions: vec![0.5, 1.0],
        },
    }
}

#[test]
fn validate_structure_accepts_valid_factor() {
    assert_eq!(valid_factor().validate_structure(), Ok(()));
}

#[test]
fn validate_structure_rejects_each_corruption() {
    // (label, corruption applied to a valid factor, expected error).
    type Case = (&'static str, fn(&mut Factor<f64>), FactorError);
    // Each case corrupts exactly one field of an otherwise-valid factor and
    // expects the matching error, covering every FactorError variant.
    let cases: &[Case] = &[
        (
            "original_n > n",
            |f| f.original_n = f.n + 1,
            FactorError::OriginalDimExceedsInternal {
                original_n: 4,
                n: 3,
            },
        ),
        (
            "neighbor/fraction length mismatch",
            |f| f.sequence.elimination_fractions.push(0.25),
            FactorError::NeighborFractionLengthMismatch {
                neighbor_len: 2,
                fraction_len: 3,
            },
        ),
        (
            "neighbor range past nnz",
            |f| f.sequence.steps[0].end = 5,
            FactorError::NeighborRangeInvalid {
                step: 0,
                start: 0,
                end: 5,
                nnz: 2,
            },
        ),
        (
            "vertex out of bounds",
            |f| f.sequence.steps[0].vertex = 9,
            FactorError::VertexOutOfBounds {
                step: 0,
                vertex: 9,
                n: 3,
            },
        ),
        (
            "neighbor out of bounds",
            |f| f.sequence.neighbor_indices[1] = 9,
            FactorError::NeighborOutOfBounds {
                step: 0,
                neighbor: 9,
                n: 3,
            },
        ),
        (
            "trailing neighbor storage",
            |f| {
                f.sequence.neighbor_indices.push(1);
                f.sequence.elimination_fractions.push(0.1);
            },
            FactorError::TrailingNeighborStorage { covered: 2, nnz: 3 },
        ),
    ];

    for (name, corrupt, expected) in cases {
        let mut f = valid_factor();
        corrupt(&mut f);
        assert_eq!(
            f.validate_structure(),
            Err(expected.clone()),
            "case: {name}"
        );
    }
}
