use approx_chol::input::{CsrRef, Side};
use approx_chol::{
    factorize_with, ApproxInverse, Clamp, Clamped, Complement, Config, FrustratedRoute, Real,
    ScalingPolicy, Sgdd, Threads,
};

use crate::common::bipartite::Bipartite;
use crate::common::dense;
use crate::common::graph::{Edge, Graph};
use crate::common::grid::grid;
use crate::common::lcg::Lcg;
use crate::common::scaling::{dld, exhausted};

/// `copies` disjoint 8×8 grids, a lone grounded vertex after each.
fn grids(copies: usize) -> Graph {
    let block = grid(8, 8);
    let stride = block.n + 1;
    let edges = (0..copies).flat_map(|c| {
        (block.edges.iter()).map(move |edge| Edge {
            a: c * stride + edge.a,
            b: c * stride + edge.b,
            ..*edge
        })
    });
    let mut graph = Graph::new(copies * stride, edges.collect());
    for c in 0..copies {
        graph.ground[c * stride + block.n] = 1.0;
    }
    graph
}

/// Each clamped component's vertices, in its factor's order, and its clamp.
fn component_clamps<T: Real>(factor: &ApproxInverse<T>) -> Vec<(Vec<usize>, Clamp)> {
    (factor.components())
        .filter_map(|c| Some((c.vertices().collect(), c.clamp()?.clone())))
        .collect()
}

fn read<'s>(clamps: impl Iterator<Item = Clamped<'s>>) -> Vec<(Vec<usize>, Clamp)> {
    clamps
        .map(|c| (c.vertices().collect(), c.clamp().clone()))
        .collect()
}

/// A clone factors under another `Config` as a fresh reduction would.
#[test]
fn a_cloned_reduction_factors_under_each_config() {
    let (ptrs, cols, vals) = dld::<f64>(&grids(2), &mut Lcg(0x5eed), 2.0);
    let csr = CsrRef::new(ptrs.len() - 1, &ptrs, &cols, &vals).expect("valid");
    let sgdd = Sgdd::reduce(csr, ScalingPolicy::default(), Threads::All).expect("reduces");
    let signed = Config {
        frustrated: FrustratedRoute::Signed,
        exact_max_dim: 0,
        seed: 3,
        ..Config::default()
    };
    for config in [Config::default(), signed] {
        let fresh = factorize_with(csr, config, ScalingPolicy::default()).expect("factors");
        let factor = sgdd.clone().factor(config, Threads::All);
        assert_eq!(format!("{factor:?}"), format!("{fresh:?}"));
    }
}

/// Before factoring, `clamps` reads each clamped component as its factor's `components` will: same order, vertices in the factor's order, same clamp.
#[test]
fn a_reductions_clamps_are_its_factors_components_clamps() {
    let (ptrs, cols, vals) = dld::<f64>(&grids(3), &mut Lcg(0x5eed), 2.0);
    let csr = CsrRef::new(ptrs.len() - 1, &ptrs, &cols, &vals).expect("valid");
    let sgdd = Sgdd::reduce(csr, exhausted(), Threads::All).expect("reduces");
    let clamps = read(sgdd.clamps());
    assert_eq!(clamps.len(), 3, "each grid clamped: {clamps:?}");
    let first: Vec<usize> = (0..64).collect();
    assert_eq!(clamps[0].0, first, "the first grid, in A's numbering");
    let factor = sgdd.factor(Config::default(), Threads::All);
    assert_eq!(clamps, component_clamps(&factor));
}

/// A reduction with nothing short of dominance has no clamps.
#[test]
fn a_dominant_reduction_has_no_clamps() {
    let graph = grid(6, 6);
    let (ptrs, cols, vals) = graph.csr(false);
    let csr = CsrRef::new(graph.n, &ptrs, &cols, &vals).expect("valid");
    let sgdd = Sgdd::reduce(csr, exhausted(), Threads::All).expect("reduces");
    assert_eq!(sgdd.clamps().count(), 0);
}

/// A complement's clamp is its whole block's, on the kept vertices in the kept side's numbering, as its factor's component reads it.
#[test]
fn a_complements_clamps_are_on_its_kept_side() {
    let gram = Bipartite::new(31, 12, 60, true);
    let whole = Sgdd::reduce(gram.view(), exhausted(), Threads::All).expect("reduces");
    let [(_, clamp)] = &read(whole.clamps())[..] else {
        panic!("one clamp")
    };
    for (kept, vertices) in [(Side::Left, 0..30), (Side::Right, 0..12)] {
        let complement =
            Complement::reduce_onto(gram.view(), kept, exhausted(), Threads::All).expect("reduces");
        let clamps = read(complement.clamps());
        let [(on_kept, of_kept)] = &clamps[..] else {
            panic!("one clamp: {clamps:?}")
        };
        let mut sorted = on_kept.clone();
        sorted.sort_unstable();
        assert!(sorted.into_iter().eq(vertices), "{kept:?}: {on_kept:?}");
        assert_eq!(of_kept, clamp, "{kept:?}");
        let factor = complement.factor(dense::dense(), Threads::All);
        assert_eq!(clamps, component_clamps(&factor), "{kept:?}");
    }
}
