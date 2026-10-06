use super::*;
use crate::approx_chol::factorization::cholesky::Cholesky;
use crate::approx_chol::factorization::exact::LowerTriangular;

#[test]
fn permutation_gather_matches_its_definition_and_scatter_inverts_it() {
    // A 2-cycle would make gather and scatter identical, hiding a reversed mapping.
    let forward = [2u32, 0, 1];
    let permutation = Permutation::from_order(forward.to_vec()).expect("not the identity");

    let original = [10.0_f64, 20.0, 30.0];
    let mut slots = [0.0_f64; 3];
    permutation.gather_into(&original, 0..3, &mut slots);
    for (position, &source) in forward.iter().enumerate() {
        assert_eq!(slots[position], original[source as usize]);
    }

    let mut values = [0.0_f64; 3];
    permutation.scatter_from(&slots, 0..3, &mut values);
    assert_eq!(values, original);
}

#[test]
fn permutation_of_identity_is_none() {
    assert!(Permutation::from_order(vec![0, 1, 2, 3]).is_none());
    assert!(Permutation::from_order(Vec::new()).is_none());
}

/// Exact over `rows` eliminated slots; any positive pivots do, since nothing checks accuracy.
fn exact(rows: usize) -> Cholesky<f64> {
    let values = match rows {
        1 => vec![2.0],
        _ => vec![2.0, -0.5, 1.5],
    };
    Cholesky::Exact(LowerTriangular { values })
}

/// The first two input vertices of `order` grounded, the last two floating: five slots for four.
fn mixed_in(order: [u32; 4]) -> Factor<f64> {
    Factor::from_blocks(
        Permutation::from_order(order.to_vec()),
        vec![Block::Grounded(exact(2)), Block::Floating(exact(1))],
        Vec::new(),
    )
}

fn mixed() -> Factor<f64> {
    mixed_in([3, 0, 2, 1])
}

/// The identity order still has a ground slot past the input, so it cannot solve in place.
#[test]
fn a_factor_reports_the_input_dimension_and_keeps_ground_slots_internal() {
    for order in [[3, 0, 2, 1], [0, 1, 2, 3]] {
        let factor = mixed_in(order);
        assert_eq!((factor.n(), factor.slots), (4, 5));

        let b = [1.0, -2.0, 0.5, 4.0];
        let x = factor.solve(&b).expect("solve");
        assert_eq!(x.len(), 4);

        let at = |position: usize| order[position] as usize;
        let mut grounded = [b[at(0)], b[at(1)], f64::NAN];
        Block::Grounded(exact(2)).solve(&mut grounded);
        let mut floating = [b[at(2)], b[at(3)]];
        Block::Floating(exact(1)).solve(&mut floating);
        let mut expected = [0.0; 4];
        for (position, value) in grounded[..2].iter().chain(&floating).enumerate() {
            expected[at(position)] = *value;
        }
        assert_eq!(x, expected, "order {order:?}");
    }
}

/// Every fact no single block can see.
mod validation {
    use super::*;

    /// Each block carries its own ground, so any number of grounded blocks is one system.
    #[test]
    fn grounded_blocks_need_not_share_a_ground() {
        let factor = Factor::of(
            None,
            vec![Block::Grounded(exact(2)), Block::Grounded(exact(1))],
            Vec::new(),
        );
        assert_eq!(factor.validate_structure(), Ok(()));
        assert_eq!((factor.n(), factor.slots), (3, 5));
    }

    /// Reports the offending entry, or the map's length when it is too short to have one.
    #[test]
    fn a_permutation_that_does_not_cover_the_input_is_rejected() {
        let cases = [
            ("a position out of bounds", vec![0, 1, 2, 99], 99),
            ("a repeated position", vec![0, 1, 1, 2], 1),
            ("shorter than the input", vec![1, 0, 2], 3),
            ("covering the ground slot too", vec![0, 1, 2, 3, 4], 5),
        ];

        for (label, forward, position) in cases {
            let mut factor = mixed();
            factor.permutation = Some(Permutation { forward });

            assert_eq!(
                factor.validate_structure(),
                Err(FactorError::PermutationInvalid { position }),
                "{label}"
            );
        }
    }

    #[test]
    fn a_corrupt_block_is_rejected_at_the_factor() {
        let factor = Factor::of(
            None,
            vec![Block::Floating(Cholesky::Exact(LowerTriangular {
                values: vec![0.0],
            }))],
            Vec::new(),
        );
        assert_eq!(
            factor.validate_structure(),
            Err(FactorError::ExactPivotInvalid { index: 0 })
        );
    }
}
