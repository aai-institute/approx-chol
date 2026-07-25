use super::*;

#[test]
fn dense_factor_solves_and_projects_to_zero_mean() {
    let factor = BlockFactor::dense(3, vec![4.0, 1.0, 1.0, 3.0], &[0, 1]).unwrap();
    // Zero-sum, so the range projection is a no-op and the solve reduces to the
    // anchor-deleted 2x2 system.
    let mut rhs = [6.0_f64, 7.0, -13.0];
    factor.solve_recovered(&mut rhs, 2, None);
    assert!(rhs[0].abs() < 1e-12);
    assert!((rhs[1] - 1.0).abs() < 1e-12);
    assert!((rhs[2] + 1.0).abs() < 1e-12);
}

#[test]
fn permutation_gather_matches_its_definition_and_scatter_inverts_it() {
    // A 2-cycle would make gather and scatter identical, hiding a reversed rotation.
    let forward = [2u32, 0, 1];
    let permutation = Permutation::from_forward(&forward).expect("not the identity");

    let original = [10.0_f64, 20.0, 30.0];
    let mut values = original;
    permutation.gather(&mut values);
    for (position, &source) in forward.iter().enumerate() {
        assert_eq!(values[position], original[source as usize]);
    }

    permutation.scatter(&mut values);
    assert_eq!(values, original);
}

#[test]
fn permutation_of_identity_is_none() {
    assert!(Permutation::from_forward(&[0, 1, 2, 3]).is_none());
    assert!(Permutation::from_forward(&[]).is_none());
}

#[test]
fn dense_factor_reports_non_positive_pivot() {
    let error = BlockFactor::dense(3, vec![1.0, 1.0, 1.0, 1.0], &[4, 7]).unwrap_err();
    assert_eq!(
        error,
        Error::DenseFactorizationFailed {
            vertex: 7,
            failure: DenseFailure::NonPositivePivot,
        }
    );
}

#[cfg(feature = "serde")]
mod validation {
    use super::*;

    fn sequence() -> EliminationSequence<f64> {
        EliminationSequence {
            vertices: vec![0, 1],
            offsets: vec![0, 1, 2],
            neighbor_indices: vec![1, 2],
            elimination_fractions: vec![1.0, 1.0],
            inv_diagonal: vec![1.0, 1.0],
        }
    }

    fn factor_of(block: BlockFactor<f64>) -> Factor<f64> {
        Factor {
            n: 3,
            original_n: 3,
            permutation: None,
            blocks: vec![Block {
                start: 0,
                anchor: 2,
                ground: None,
                factor: block,
            }],
            exact_fallbacks: Vec::new(),
        }
    }

    fn approx() -> Factor<f64> {
        factor_of(BlockFactor::approx(3, sequence()))
    }

    fn dense() -> Factor<f64> {
        factor_of(BlockFactor::dense(3, vec![4.0, 1.0, 1.0, 3.0], &[0, 1]).expect("valid pivots"))
    }

    fn seq_of(factor: &mut Factor<f64>) -> &mut EliminationSequence<f64> {
        match &mut factor.blocks[0].factor {
            BlockFactor::Approx { sequence, .. } => sequence,
            BlockFactor::Dense { .. } => unreachable!("fixture is an approx factor"),
        }
    }

    fn lower_of(factor: &mut Factor<f64>) -> &mut Vec<f64> {
        match &mut factor.blocks[0].factor {
            BlockFactor::Dense { lower, .. } => lower,
            BlockFactor::Approx { .. } => unreachable!("fixture is a dense factor"),
        }
    }

    #[test]
    fn valid_fixtures_pass() {
        approx()
            .validate_structure()
            .expect("approx fixture is valid");
        dense()
            .validate_structure()
            .expect("dense fixture is valid");
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
                "offsets length",
                approx,
                |f| {
                    seq_of(f).offsets.pop();
                },
                FactorError::OffsetsLengthMismatch {
                    expected: 3,
                    got: 2,
                },
            ),
            (
                "inv_diagonal length",
                approx,
                |f| {
                    seq_of(f).inv_diagonal.pop();
                },
                FactorError::InvDiagonalLengthMismatch {
                    expected: 2,
                    got: 1,
                },
            ),
            (
                "neighbor/fraction length",
                approx,
                |f| {
                    seq_of(f).elimination_fractions.pop();
                },
                FactorError::NeighborFractionLengthMismatch {
                    neighbor_len: 2,
                    fraction_len: 1,
                },
            ),
            (
                "offsets start",
                approx,
                |f| seq_of(f).offsets[0] = 1,
                FactorError::OffsetsMustStartAtZero { got: 1 },
            ),
            (
                "offset range",
                approx,
                |f| seq_of(f).offsets[1] = 99,
                FactorError::OffsetRangeInvalid {
                    step: 0,
                    start: 0,
                    end: 99,
                    nnz: 2,
                },
            ),
            (
                "pivot vertex bounds",
                approx,
                |f| seq_of(f).vertices[0] = 99,
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
                "final offset",
                approx,
                |f| seq_of(f).offsets[2] = 1,
                FactorError::FinalOffsetMismatch { last: 1, nnz: 2 },
            ),
            (
                "dense storage length",
                dense,
                |f| lower_of(f).push(0.0),
                FactorError::DenseLengthInvalid { n: 3, len: 5 },
            ),
            (
                "dense pivot sign",
                dense,
                |f| lower_of(f)[0] = -1.0,
                FactorError::DensePivotInvalid { index: 0 },
            ),
            (
                "dense pivot finiteness",
                dense,
                |f| lower_of(f)[3] = f64::NAN,
                FactorError::DensePivotInvalid { index: 1 },
            ),
            (
                "block start",
                dense,
                |f| f.blocks[0].start = 1,
                FactorError::BlockRangeInvalid { start: 1, n: 3 },
            ),
            (
                "blocks do not cover n",
                dense,
                |f| f.n = 4,
                FactorError::BlockRangeInvalid { start: 3, n: 4 },
            ),
            (
                "anchor out of block",
                dense,
                |f| f.blocks[0].anchor = 3,
                FactorError::BlockAnchorInvalid { anchor: 3, n: 3 },
            ),
            (
                "ground out of block",
                dense,
                |f| f.blocks[0].ground = Some(3),
                FactorError::BlockGroundInvalid { ground: 3, n: 3 },
            ),
            (
                "permutation position out of bounds",
                dense,
                |f| {
                    f.permutation = Some(Permutation {
                        cycles: vec![0, 99],
                        starts: vec![0, 2],
                    });
                },
                FactorError::PermutationInvalid { position: 99 },
            ),
            (
                "permutation repeats a position",
                dense,
                |f| {
                    f.permutation = Some(Permutation {
                        cycles: vec![1, 1],
                        starts: vec![0, 2],
                    });
                },
                FactorError::PermutationInvalid { position: 1 },
            ),
            (
                "permutation cycle is a fixed point",
                dense,
                |f| {
                    f.permutation = Some(Permutation {
                        cycles: vec![1],
                        starts: vec![0, 1],
                    });
                },
                FactorError::PermutationInvalid { position: 0 },
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
