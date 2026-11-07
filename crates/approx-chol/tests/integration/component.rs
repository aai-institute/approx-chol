use approx_chol::factorize;
use approx_chol::input::{BipartiteFactorWidth2, FactorWidth1};
use rstest::rstest;

use crate::common::entry::only;

/// A bipartite component lists its larger side first, the one its factor eliminates, so its vertices follow the factor, not their numbering.
#[rstest]
#[case::right_larger(2, 6, &[2, 3, 4, 5, 6, 7, 0, 1])]
#[case::left_larger(6, 2, &[0, 1, 2, 3, 4, 5, 6, 7])]
fn a_component_lists_its_vertices_in_its_factors_order(
    #[case] n_left: usize,
    #[case] n_right: usize,
    #[case] expected: &[usize],
) {
    let (left, right): (Vec<u32>, Vec<u32>) = (0..n_left * n_right)
        .map(|k| ((k % n_left) as u32, (k / n_left) as u32))
        .unzip();
    let gram = BipartiteFactorWidth2::new(
        FactorWidth1::<f64, u32>::new(n_left, &left, None).expect("valid left"),
        FactorWidth1::new(n_right, &right, None).expect("valid right"),
    )
    .expect("valid gram");
    let factor = factorize(gram).expect("factors");
    assert!(only(&factor).vertices().eq(expected.iter().copied()));
}
