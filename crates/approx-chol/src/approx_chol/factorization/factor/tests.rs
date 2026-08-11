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

/// Every fact no single block can see; a block's own serde boundary owns the rest.
mod validation {
    use crate::approx_chol::factorization::exact::LowerTriangular;
    use crate::approx_chol::factorization::{
        anchor::Anchor,
        block::{Block, BlockDim},
        cholesky::Cholesky,
    };

    use super::*;

    /// Three variables, and a cholesky that is valid but arbitrary: nothing at this level
    /// reads it.
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

    #[test]
    fn a_factor_of_valid_blocks_passes() {
        floating()
            .validate_structure()
            .unwrap_or_else(|error| panic!("fixture is valid: {error}"));
    }

    /// Two blocks tiling `5 + 1` variables, both claiming the one ground vertex.
    #[test]
    fn a_second_block_cannot_claim_the_ground_vertex() {
        let factor = of_blocks(vec![block(Anchor::Ground), block(Anchor::Ground)]);

        assert_eq!(
            factor.validate_structure(),
            Err(FactorError::MultipleGroundBlocks { grounded: 2 })
        );
    }

    /// The reported position is the offending entry, or the map's length when it is too
    /// short to have one.
    #[test]
    fn a_permutation_that_does_not_cover_the_factor_is_rejected() {
        let cases = [
            ("a position out of bounds", vec![0, 1, 99], 99),
            ("a repeated position", vec![0, 1, 1], 1),
            ("shorter than the factor", vec![1, 0], 2),
        ];

        for (label, forward, position) in cases {
            let mut factor = floating();
            factor.permutation = Some(Permutation { forward });

            assert_eq!(
                factor.validate_structure(),
                Err(FactorError::PermutationInvalid { position }),
                "{label}"
            );
        }
    }
}
