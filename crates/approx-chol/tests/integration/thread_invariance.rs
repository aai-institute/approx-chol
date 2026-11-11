use std::num::NonZeroUsize;

use approx_chol::input::{CsrRef, FactorWidth2, Side};
use approx_chol::ScalingOutcome::Unsettled;
use approx_chol::{
    ApproxInverse, BuildError, Complement, Component, Config, CoverSurplus, FrustratedRoute, Real,
    ScalingPolicy, SchurMode, Sgdd, Threads,
};
use rstest::rstest;

use crate::common::bipartite::Bipartite;
use crate::common::bits::bits;
use crate::common::disjoint::Disjoint;
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

/// Gram components short of dominance, alternately certified by a null vector and relaxed, each reduced in its task: the factor and its solve agree bit for bit across budgets, `Count(6)` among them, and exactly the relaxed ones stop unsettled when relax `stops` at once.
fn gram_blocks(sizes: &[usize], policy: ScalingPolicy, stops: bool) -> (String, Vec<u64>) {
    let gram = Disjoint::new(sizes, |k| k % 2 == 0);
    let six = Threads::Count(NonZeroUsize::new(6).unwrap());
    let build = |threads| {
        (Sgdd::reduce(gram.view(), policy, threads).unwrap()).factor(Config::default(), threads)
    };
    let factor = build(six);
    let shown = format!("{factor:?}");
    for threads in threads::BUDGETS {
        let built = format!("{:?}", build(threads));
        assert_eq!(built, shown, "built at {threads:?}");
    }
    assert_eq!(factor.components().len(), sizes.len());
    for (k, c) in factor.components().enumerate() {
        let unsettled = (c.clamp()).is_some_and(|c| c.scaling == Unsettled);
        assert_eq!(unsettled, stops && k % 2 == 1, "component {k}");
    }
    let b: Vec<f64> = (0..factor.n())
        .map(|i| ((i * 7919) % 101) as f64 / 50.0 - 1.0)
        .collect();
    (shown, bits(&factor.solve(&b).unwrap()))
}

/// 600 ten-vertex stars, past the split grain, four of whose centers sum their absolute row past `MAX/4` though each cell and diagonal is in range: the reduction of a block, not the assembly, fails, at the first such center in component order under every budget.
#[test]
fn a_gram_fails_at_its_first_failing_block_across_thread_counts() {
    let failing = [7, 150, 151, 599];
    let (mut a, mut b, mut x, mut y) = (vec![], vec![], vec![], vec![]);
    for k in 0..600u32 {
        for leaf in 1..10 {
            let (x_k, y_k) = match failing.contains(&k) {
                true => (2e153, -6e153),
                false => (1.0, -0.5),
            };
            a.push(10 * k);
            b.push(10 * k + leaf);
            x.push(x_k);
            y.push(y_k);
        }
    }
    let gram = FactorWidth2::new(6000, (&a, &x), (&b, &y)).unwrap();
    let first = BuildError::OutOfRange { vertex: 70 };
    threads::across_pools(|| {
        for threads in threads::BUDGETS {
            let error = Sgdd::reduce(gram, ScalingPolicy::default(), threads).err();
            assert_eq!(error, Some(first), "{threads:?}");
        }
    });
}

/// 200 small blocks, past `extract`'s grain.
const MANY: [usize; 200] = [64; 200];
/// Two relaxed blocks whose products pass the gather grain (2¹³ multiply-adds) under the `Count(3)` that `extract` leaves each task of `Count(6)`.
const NESTED: [usize; 8] = [64, 3000, 64, 3000, 64, 64, 64, 64];

#[rstest]
#[case::at_once(ScalingPolicy::new(1e-9, 0).unwrap(), true)]
#[case::settled(ScalingPolicy::default(), false)]
fn gram_blocks_are_bit_identical_across_thread_counts(
    #[values(&MANY[..], &NESTED[..])] sizes: &[usize],
    #[case] policy: ScalingPolicy,
    #[case] stops: bool,
) {
    threads::across_pools(|| gram_blocks(sizes, policy, stops));
}
