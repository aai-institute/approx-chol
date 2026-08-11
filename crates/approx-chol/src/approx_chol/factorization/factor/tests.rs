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

/// A block validates its own cholesky against its own dim before it exists, so what is
/// left here is what only the whole factor can see.
mod validation {
    use crate::approx_chol::factorization::exact::LowerTriangular;
    use crate::approx_chol::factorization::{
        anchor::Anchor,
        block::{Block, BlockDim},
        cholesky::Cholesky,
    };

    use super::*;

    /// Three variables, of which the block solves for `3 - 1`, so its packed factor
    /// holds `2 * 3 / 2`.
    fn block(anchor: Anchor) -> Block<f64> {
        Block::new(
            BlockDim::of(3).expect("fixture dimension is non-zero"),
            anchor,
            Cholesky::Exact(LowerTriangular {
                values: vec![1.0; 3],
            }),
        )
    }

    fn of_blocks(blocks: Vec<Block<f64>>) -> Factor<f64> {
        Factor::of(None, blocks, Vec::new())
    }

    fn floating() -> Factor<f64> {
        of_blocks(vec![block(Anchor::Floating)])
    }

    /// Two blocks tiling `5 + 1` variables, both claiming the one ground vertex.
    fn two_ground_blocks() -> Factor<f64> {
        of_blocks(vec![block(Anchor::Ground), block(Anchor::Ground)])
    }

    #[test]
    fn a_factor_of_valid_blocks_passes() {
        floating()
            .validate_structure()
            .unwrap_or_else(|error| panic!("fixture is valid: {error}"));
    }

    /// Every variant no single block could have raised.
    #[test]
    fn every_cross_block_error_variant_is_reachable() {
        #[allow(clippy::type_complexity)]
        let cases: Vec<(&str, fn() -> Factor<f64>, fn(&mut Factor<f64>), FactorError)> = vec![
            (
                "two blocks claim the one ground vertex",
                two_ground_blocks,
                |_| {},
                FactorError::MultipleGroundBlocks { grounded: 2 },
            ),
            (
                "permutation position out of bounds",
                floating,
                |f| {
                    f.permutation = Some(Permutation {
                        forward: vec![0, 1, 99],
                    });
                },
                FactorError::PermutationInvalid { position: 99 },
            ),
            (
                "permutation repeats a position",
                floating,
                |f| {
                    f.permutation = Some(Permutation {
                        forward: vec![0, 1, 1],
                    });
                },
                FactorError::PermutationInvalid { position: 1 },
            ),
            (
                "permutation shorter than the factor",
                floating,
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
