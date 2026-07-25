use super::*;

#[test]
fn permutation_gather_matches_its_definition_and_scatter_inverts_it() {
    // A 2-cycle would make gather and scatter identical, hiding a reversed mapping.
    let forward = [2u32, 0, 1];
    let permutation = Permutation::from_forward(&forward).expect("not the identity");

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
    assert!(Permutation::from_forward(&[0, 1, 2, 3]).is_none());
    assert!(Permutation::from_forward(&[]).is_none());
}

mod validation {
    use super::super::super::sequence::StepHeader;
    use super::*;

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
        }
    }

    fn factor_of(block: BlockFactor<f64>) -> Factor<f64> {
        Factor {
            n: 3,
            original_n: 3,
            permutation: None,
            blocks: vec![Block {
                start: 0,
                ground: None,
                factor: block,
            }],
        }
    }

    fn approx() -> Factor<f64> {
        factor_of(BlockFactor::approx(3, 2, sequence()))
    }

    fn seq_of(factor: &mut Factor<f64>) -> &mut EliminationSequence<f64> {
        &mut factor.blocks[0].factor.sequence
    }

    fn anchor_of(factor: &mut Factor<f64>) -> &mut u32 {
        &mut factor.blocks[0].factor.anchor
    }

    #[test]
    fn valid_fixture_passes() {
        approx()
            .validate_structure()
            .expect("approx fixture is valid");
    }

    #[test]
    fn every_factor_error_variant_is_reachable() {
        #[allow(clippy::type_complexity)]
        let cases: Vec<(&str, fn() -> Factor<f64>, fn(&mut Factor<f64>), FactorError)> = vec![
            (
                "original_n exceeds n",
                approx,
                |f| f.original_n = f.n + 1,
                FactorError::OriginalDimExceedsInternal {
                    original_n: 4,
                    n: 3,
                },
            ),
            (
                "neighbor range past nnz",
                approx,
                |f| seq_of(f).steps[0].end = 99,
                FactorError::NeighborRangeInvalid {
                    step: 0,
                    start: 0,
                    end: 99,
                    nnz: 2,
                },
            ),
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
                "trailing neighbor storage",
                approx,
                |f| seq_of(f).steps[1].end = 1,
                FactorError::TrailingNeighborStorage { covered: 1, nnz: 2 },
            ),
            (
                "block start",
                approx,
                |f| f.blocks[0].start = 1,
                FactorError::BlockRangeInvalid { start: 1, n: 3 },
            ),
            (
                "blocks do not cover n",
                approx,
                |f| f.n = 4,
                FactorError::BlockRangeInvalid { start: 3, n: 4 },
            ),
            (
                "anchor out of block",
                approx,
                |f| *anchor_of(f) = 3,
                FactorError::BlockAnchorInvalid { anchor: 3, n: 3 },
            ),
            (
                "ground out of block",
                approx,
                |f| f.blocks[0].ground = Some(3),
                FactorError::BlockGroundInvalid { ground: 3, n: 3 },
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
            let error = factor
                .validate_structure()
                .expect_err(&format!("{label}: corruption must be rejected"));
            assert_eq!(error, expected, "{label}");
        }
    }
}
