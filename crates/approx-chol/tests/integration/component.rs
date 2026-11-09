use approx_chol::input::{BipartiteFactorWidth2, FactorWidth1, FactorWidth2, Side};
use approx_chol::{factorize, factorize_complement, ApproxInverse, Config, ScalingPolicy};
use rstest::rstest;

use crate::common::entry::only;

/// A bipartite gram with one column per `(left, right)` edge.
fn gram<'a>(
    n_left: usize,
    n_right: usize,
    edges: &'a (Vec<u32>, Vec<u32>),
) -> BipartiteFactorWidth2<'a, f64, u32> {
    BipartiteFactorWidth2::new(
        FactorWidth1::<f64, u32>::new(n_left, &edges.0).expect("valid left"),
        FactorWidth1::new(n_right, &edges.1).expect("valid right"),
    )
    .expect("valid gram")
}

/// Every edge between `lefts` and `rights`.
fn complete(lefts: &[u32], rights: &[u32]) -> (Vec<u32>, Vec<u32>) {
    (lefts.iter())
        .flat_map(|&l| rights.iter().map(move |&r| (l, r)))
        .unzip()
}

fn vertices(factor: &ApproxInverse) -> Vec<Vec<usize>> {
    (factor.components())
        .map(|component| component.vertices().collect())
        .collect()
}

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
    let factor = factorize(gram(n_left, n_right, &(left, right))).expect("factors");
    assert!(only(&factor).vertices().eq(expected.iter().copied()));
}

/// Beside a component eliminating its left side, one eliminating its right side lists it first; both keep their own order.
#[test]
fn each_component_of_a_disconnected_input_leads_with_its_own_larger_side() {
    let (mut edges, larger_left) = (
        complete(&[0, 1], &[0, 1, 2, 3]),
        complete(&[2, 3, 4, 5], &[4, 5]),
    );
    edges.0.extend(larger_left.0);
    edges.1.extend(larger_left.1);
    let factor = factorize(gram(6, 6, &edges)).expect("factors");
    assert_eq!(
        vertices(&factor),
        [vec![6, 7, 8, 9, 0, 1], vec![2, 3, 4, 5, 10, 11]]
    );
}

/// A complement keeps each component's side in the kept side's numbering, listed where it interleaves with another's and a run where it does not.
#[rstest]
#[case::interleaved(Side::Left, [vec![0, 2], vec![1, 3]])]
#[case::runs(Side::Right, [vec![0, 1], vec![2, 3]])]
fn a_complements_components_keep_their_side(#[case] kept: Side, #[case] expected: [Vec<usize>; 2]) {
    let (mut edges, other) = (complete(&[0, 2], &[0, 1]), complete(&[1, 3], &[2, 3]));
    edges.0.extend(other.0);
    edges.1.extend(other.1);
    let factor = factorize_complement(
        gram(4, 4, &edges),
        kept,
        Config::default(),
        ScalingPolicy::default(),
    )
    .expect("factors");
    assert_eq!(vertices(&factor), expected);
}

/// A Gram settles its folded components after its frustrated ones; a frustrated one holding the lowest vertex still comes first.
#[test]
fn a_grams_components_are_listed_by_lowest_vertex() {
    let (a, b) = ([0u32, 1, 0, 3, 4, 3], [1u32, 2, 2, 4, 5, 5]);
    let x = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
    let y = [4.0, 5.0, 6.0, -1.0, -2.0, -3.0];
    let gram = FactorWidth2::new(6, (&a, &x), (&b, &y)).expect("valid gram");
    let factor = factorize(gram).expect("factors");
    let floats: Vec<bool> = factor.components().map(|c| c.floats()).collect();
    assert_eq!(vertices(&factor), [vec![0, 1, 2], vec![3, 4, 5]]);
    assert_eq!(floats, [false, true], "the frustrated triangle first");
}
