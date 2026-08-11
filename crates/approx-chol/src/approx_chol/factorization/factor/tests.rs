use super::*;

#[test]
fn permutation_gather_matches_its_definition_and_scatter_inverts_it() {
    // A 2-cycle would make gather and scatter identical, hiding a reversed mapping.
    let forward = [2u32, 0, 1];
    let permutation = Permutation::from_order(forward.to_vec()).expect("not the identity");

    let original = [10.0_f64, 20.0, 30.0];
    let mut scratch = [0.0_f64; 3];
    permutation.gather_into(&original, &mut scratch);
    for (position, &source) in forward.iter().enumerate() {
        assert_eq!(scratch[position], original[source as usize]);
    }

    let mut values = [0.0_f64; 3];
    permutation.scatter_from(&scratch, &mut values);
    assert_eq!(values, original);
}

#[test]
fn permutation_of_identity_is_none() {
    assert!(Permutation::from_order(vec![0, 1, 2, 3]).is_none());
    assert!(Permutation::from_order(Vec::new()).is_none());
}

mod validation {
    use crate::approx_chol::factorization::approximate::{EliminationSequence, StepHeader};
    use crate::approx_chol::factorization::exact::LowerTriangular;
    use crate::approx_chol::factorization::{
        anchor::Anchor,
        block::{Block, BlockDim},
        cholesky::Cholesky,
    };

    use super::*;

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

    fn of_blocks(blocks: Vec<Block<f64>>) -> Factor<f64> {
        Factor::of(None, blocks, Vec::new())
    }

    fn validate(factor: &Factor<f64>) -> Result<(), FactorError> {
        Factor::validate_structure(&factor.blocks, factor.permutation.as_ref())
    }

    fn with_cholesky(cholesky: Cholesky<f64>) -> Factor<f64> {
        of_blocks(vec![Block::new(dim(3), Anchor::Floating, cholesky)])
    }

    /// Two blocks tiling `5 + 1` variables, both claiming the one ground vertex.
    fn two_ground_blocks() -> Factor<f64> {
        of_blocks(vec![
            Block::new(dim(3), Anchor::Ground, Cholesky::Approximate(sequence())),
            Block::new(dim(3), Anchor::Ground, Cholesky::Approximate(sequence())),
        ])
    }

    fn approx() -> Factor<f64> {
        with_cholesky(Cholesky::Approximate(sequence()))
    }

    /// The block solves for `3 - 1` variables, so its packed factor holds `2 * 3 / 2`.
    fn exact() -> Factor<f64> {
        with_cholesky(Cholesky::Exact(LowerTriangular {
            values: vec![1.0; 3],
        }))
    }

    fn seq_of(factor: &mut Factor<f64>) -> &mut EliminationSequence<f64> {
        match &mut factor.blocks[0].cholesky {
            Cholesky::Approximate(sequence) => sequence,
            Cholesky::Exact(_) => unreachable!("fixture is approximate"),
        }
    }

    fn lower_of(factor: &mut Factor<f64>) -> &mut LowerTriangular<f64> {
        match &mut factor.blocks[0].cholesky {
            Cholesky::Exact(lower) => lower,
            Cholesky::Approximate(_) => unreachable!("fixture is exact"),
        }
    }

    #[test]
    fn valid_fixtures_pass() {
        for (label, factor) in [("approximate", approx()), ("exact", exact())] {
            validate(&factor).unwrap_or_else(|error| panic!("{label} fixture is valid: {error}"));
        }
    }

    /// Two blocks whose dims cannot be added; each is checked against its own cholesky
    /// before it is summed, so the total is never formed.
    #[test]
    fn dims_that_cannot_be_summed_are_rejected_before_they_are() {
        let huge = || {
            Block::new(
                BlockDim::of(usize::MAX).expect("non-zero"),
                Anchor::Floating,
                Cholesky::Exact(LowerTriangular {
                    values: vec![1.0; 3],
                }),
            )
        };

        assert_eq!(
            Factor::validate_structure(&[huge(), huge()], None),
            Err(FactorError::ExactFactorLengthInvalid {
                n: usize::MAX,
                len: 3,
            })
        );
    }

    /// Every variant but `NonzeroCountExceedsU32`, which needs more than `u32::MAX`
    /// nonzeros to reach.
    #[test]
    fn every_factor_error_variant_is_reachable() {
        #[allow(clippy::type_complexity)]
        let cases: Vec<(&str, fn() -> Factor<f64>, fn(&mut Factor<f64>), FactorError)> = vec![
            (
                "pivot vertex bounds",
                approx,
                |f| seq_of(f).steps[0].vertex = 99,
                FactorError::VertexOutOfBounds {
                    step: 0,
                    vertex: 99,
                    n: 3,
                },
            ),
            (
                "neighbor bounds",
                approx,
                |f| seq_of(f).neighbor_indices[0] = 99,
                FactorError::NeighborOutOfBounds {
                    step: 0,
                    neighbor: 99,
                    n: 3,
                },
            ),
            (
                "two blocks claim the one ground vertex",
                two_ground_blocks,
                |_| {},
                FactorError::MultipleGroundBlocks { grounded: 2 },
            ),
            (
                "exact pivot too small to divide by",
                exact,
                |f| lower_of(f).values[0] = 1e-320,
                FactorError::ExactPivotInvalid { index: 0 },
            ),
            (
                "exact off-diagonal squares to infinity",
                exact,
                |f| lower_of(f).values[1] = 1e308,
                FactorError::ExactRowNotRepresentable { row: 1 },
            ),
            (
                "step inv_diag is not finite",
                approx,
                |f| seq_of(f).steps[0].inv_diag = f64::INFINITY,
                FactorError::StepValueInvalid { step: 0 },
            ),
            (
                "elimination fraction is not a proportion",
                approx,
                |f| seq_of(f).elimination_fractions[0] = 2.0,
                FactorError::StepValueInvalid { step: 0 },
            ),
            (
                "uneliminated vertex bounds",
                approx,
                |f| seq_of(f).uneliminated = 99,
                FactorError::UneliminatedVertexInvalid { vertex: 99, n: 3 },
            ),
            (
                "uneliminated vertex is a pivot a step already eliminated",
                approx,
                |f| seq_of(f).uneliminated = 1,
                FactorError::UneliminatedVertexInvalid { vertex: 1, n: 3 },
            ),
            (
                "steps leave a second vertex uneliminated",
                approx,
                |f| {
                    seq_of(f).steps.truncate(1);
                },
                FactorError::StepCountDoesNotTileBlock { steps: 1, n: 3 },
            ),
            (
                "one vertex is eliminated twice, so another never is",
                approx,
                |f| seq_of(f).steps[1].vertex = 0,
                FactorError::VertexEliminatedTwice { step: 1, vertex: 0 },
            ),
            (
                "exact factor shorter than its block",
                exact,
                |f| lower_of(f).values.truncate(2),
                FactorError::ExactFactorLengthInvalid { n: 3, len: 2 },
            ),
            (
                "exact factor pivot is zero",
                exact,
                |f| lower_of(f).values[0] = 0.0,
                FactorError::ExactPivotInvalid { index: 0 },
            ),
            (
                "exact factor pivot is not finite",
                exact,
                |f| lower_of(f).values[2] = f64::NAN,
                FactorError::ExactPivotInvalid { index: 1 },
            ),
            (
                "permutation position out of bounds",
                approx,
                |f| {
                    f.permutation = Some(Permutation {
                        forward: vec![0, 1, 99],
                    });
                },
                FactorError::PermutationInvalid { position: 99 },
            ),
            (
                "permutation repeats a position",
                approx,
                |f| {
                    f.permutation = Some(Permutation {
                        forward: vec![0, 1, 1],
                    });
                },
                FactorError::PermutationInvalid { position: 1 },
            ),
            (
                "permutation shorter than the factor",
                approx,
                |f| {
                    f.permutation = Some(Permutation {
                        forward: vec![1, 0],
                    });
                },
                FactorError::PermutationInvalid { position: 2 },
            ),
        ];

        for (label, build, corrupt, expected) in cases {
            let mut factor = build();
            corrupt(&mut factor);
            let error =
                validate(&factor).expect_err(&format!("{label}: corruption must be rejected"));
            assert_eq!(error, expected, "{label}");
        }
    }
}
