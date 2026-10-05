use super::*;
use crate::approx_chol::factorization::cholesky::tests::{approx, exact};

/// Only the ground slot separates the two: the same cholesky covers one input vertex
/// fewer when its last slot is the ground.
#[test]
fn a_grounded_block_has_one_slot_more_than_vertices() {
    for cholesky in [approx(), exact()] {
        let grounded = Block::Grounded(cholesky.clone());
        let floating = Block::Floating(cholesky);
        assert_eq!((grounded.vertices(), grounded.slots()), (2, 3));
        assert_eq!((floating.vertices(), floating.slots()), (3, 3));
    }
}

/// A floating block's solution is zero-mean, whatever the factor pinned.
#[test]
fn a_floating_solution_is_zero_mean() {
    let block = Block::Floating(approx());
    let mut slots = [1.0, -3.0, 2.5];
    block.solve(&mut slots);
    assert!(slots.iter().sum::<f64>().abs() < 1e-15, "{slots:?}");
}
