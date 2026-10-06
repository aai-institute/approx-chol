use super::*;
use crate::approx_chol::factorization::cholesky::tests::approx;

/// A floating block's solution is zero-mean, whatever the factor pinned.
#[test]
fn a_floating_solution_is_zero_mean() {
    let block = Block::Floating(approx());
    let mut slots = [1.0, -3.0, 2.5];
    block.solve(&mut slots);
    assert!(slots.iter().sum::<f64>().abs() < 1e-15, "{slots:?}");
}
