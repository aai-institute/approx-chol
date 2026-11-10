use approx_chol::input::{CsrRef, Side};
use approx_chol::ScalingOutcome::Unsettled;
use approx_chol::{
    ApproxInverse, Complement, Component, Config, CoverSurplus, FrustratedRoute, Real,
    ScalingPolicy, SchurMode, Sgdd,
};
use rstest::rstest;

use crate::common::bipartite::Bipartite;
use crate::common::bits::bits;
use crate::common::grid::grid;
use crate::common::lcg::Lcg;
use crate::common::scaling::dld;
use crate::common::solve::Solve;
use crate::common::sparse::clique_tree;
use crate::common::threads;

/// The factor as shown, which a build under each budget matches, and the solve of `b` as bits, which its solve and each column of a four-column apply match under each budget; `kept` factors the complement.
fn solve_bits(g: &Bipartite, config: Config, kept: Option<Side>) -> (String, Vec<u64>) {
    let (gram, policy) = (g.view(), ScalingPolicy::default());
    let build = |threads| match kept {
        None => (Sgdd::reduce(gram, policy, threads).unwrap()).factor(config, threads),
        Some(side) => {
            (Complement::reduce_onto(gram, side, policy, threads).unwrap()).factor(config, threads)
        }
    };
    let [factor, rest @ ..] = threads::BUDGETS.map(build);
    let shown = format!("{factor:?}");
    for (threads, built) in threads::BUDGETS[1..].iter().zip(&rest) {
        assert_eq!(format!("{built:?}"), shown, "built at {threads:?}");
    }
    assert_eq!(factor.components().len(), 3);
    let b: Vec<f64> = (0..factor.n())
        .map(|i| ((i * 7919) % 101) as f64 / 50.0 - 1.0)
        .collect();
    let whole = bits(&factor.solve(&b).unwrap());
    for threads in threads::BUDGETS {
        assert_eq!(
            bits(&factor.solve_at(&b, threads).unwrap()),
            whole,
            "{threads:?}"
        );
    }
    let scale = [1.0, -0.5, 0.0, 3.0];
    let alone = scale.map(|s| {
        bits(
            &factor
                .solve(&b.iter().map(|&v| s * v).collect::<Vec<_>>())
                .unwrap(),
        )
    });
    for threads in threads::BUDGETS {
        let mut rows: Vec<[f64; 4]> = b.iter().map(|&v| scale.map(|s| s * v)).collect();
        factor.apply(&mut rows, threads).unwrap();
        for (j, alone) in alone.iter().enumerate() {
            let column: Vec<f64> = rows.iter().map(|row| row[j]).collect();
            assert_eq!(&bits(&column), alone, "column {j}, {threads:?}");
        }
    }
    (shown, whole)
}

#[rstest]
#[case::ac(Config::default())]
#[case::ac2(Config { sampling: clique_tree(3), ..Config::default() })]
#[case::exact(Config { complement: SchurMode::Exact, ..Config::default() })]
#[case::surplus_edges(Config { frustrated: FrustratedRoute::Cover(CoverSurplus::SurplusEdges), ..Config::default() })]
#[case::signed(Config { frustrated: FrustratedRoute::Signed, ..Config::default() })]
fn solves_are_bit_identical_across_thread_counts(
    #[case] config: Config,
    #[values(None, Some(Side::Left), Some(Side::Right))] kept: Option<Side>,
) {
    let g = Bipartite::three_groups();
    threads::across_pools(|| solve_bits(&g, config, kept));
}

/// A 200×200 grid's `D L D`, weights over four decades and `D` over two: singular, short of dominance, and past the gathered Perron product's grain.
fn scaled_grid<T: Real>() -> String {
    let mut lcg = Lcg(0x362);
    let mut graph = grid(200, 200);
    for edge in &mut graph.edges {
        edge.weight = 10f64.powf(4.0 * lcg.unit());
    }
    let (ptrs, columns, scaled) = dld::<T>(&graph, &mut lcg, 2.0);
    let csr = CsrRef::new(graph.n, &ptrs, &columns, &scaled).unwrap();
    let factor = |policy| {
        let build = |threads| {
            let sgdd = Sgdd::reduce(csr, policy, threads).unwrap();
            let clamps: Vec<String> = sgdd.clamps().map(|c| format!("{c:?}")).collect();
            (clamps, sgdd.factor(Config::default(), threads))
        };
        let [(clamps, factor), rest @ ..] = threads::BUDGETS.map(build);
        for (threads, (at, built)) in threads::BUDGETS[1..].iter().zip(rest) {
            assert_eq!(at, clamps, "clamps at {threads:?}");
            assert_eq!(format!("{built:?}"), format!("{factor:?}"), "{threads:?}");
        }
        factor
    };
    let at_once: ApproxInverse<T> = factor(ScalingPolicy::new(1e-9, 0).unwrap());
    let unsettled = |c: Component<'_, T>| c.clamp().is_some_and(|c| c.scaling == Unsettled);
    assert!(at_once.components().any(unsettled), "relax never ran");
    let stopped = factor(ScalingPolicy::new(1e-9, 8).unwrap());
    let settled = factor(ScalingPolicy::default());
    format!("{at_once:?}{stopped:?}{settled:?}")
}

/// One thread scatters the Perron product, more gather it: the factors, stopped after `0` and `8` iterations and settled, agree bit for bit, as do a stopped one's clamps, read off a product's bracket.
#[rstest]
#[case::f64(scaled_grid::<f64>)]
#[case::f32(scaled_grid::<f32>)]
fn a_relaxation_is_bit_identical_across_thread_counts(#[case] factors: fn() -> String) {
    threads::across_pools(factors);
}
