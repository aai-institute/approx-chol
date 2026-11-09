use approx_chol::{factorize_complement, factorize_with, ApproxInverse, Config, CoverSurplus};
use approx_chol::{FrustratedRoute, ScalingPolicy, SchurMode, Side};

use crate::common::bipartite::Bipartite;
use crate::common::bits::bits;
use crate::common::cycle::Cycle;
use crate::common::dense_residual::apply;
use crate::common::entry::{Entry, Types};
use crate::common::graph::Graph;
use crate::common::grid::light_row;
use crate::common::pcg::pcg_iterations;
use crate::common::solve::Solve;

/// In the order [`each`] gives its results.
const SURPLUSES: [CoverSurplus; 3] = [
    CoverSurplus::GroundVertex,
    CoverSurplus::SurplusEdges,
    CoverSurplus::Auto,
];

/// `f` of `base` routed through the cover under each of [`SURPLUSES`].
fn each<R>(base: Config, f: impl Fn(Config) -> R) -> [R; 3] {
    SURPLUSES.map(|surplus| {
        f(Config {
            frustrated: FrustratedRoute::Cover(surplus),
            ..base
        })
    })
}

fn sparse(complement: SchurMode) -> Config {
    Config {
        exact_max_dim: 0,
        complement,
        ..Config::default()
    }
}

/// A map of vectors, `A` or its preconditioner.
type Operator<'a> = Box<dyn Fn(&[f64]) -> Vec<f64> + 'a>;

/// One factor on a cover path, with the operator it preconditions.
struct Path<'a> {
    name: &'static str,
    n: usize,
    solve: Operator<'a>,
    apply: Operator<'a>,
}

impl<'a> Path<'a> {
    fn dense(name: &'static str, factor: ApproxInverse, m: Vec<Vec<f64>>) -> Self {
        Self {
            name,
            n: m.len(),
            solve: Box::new(move |r| factor.solve(r).unwrap()),
            apply: Box::new(move |x| apply(&m, x)),
        }
    }

    fn rhs(&self) -> Vec<f64> {
        (self.apply)(
            &(0..self.n)
                .map(|v| (v % 5) as f64 - 2.0)
                .collect::<Vec<_>>(),
        )
    }

    fn bits(&self) -> Vec<u64> {
        bits(&(self.solve)(&self.rhs()))
    }

    /// PCG's iterations to `1e-10`, if within 100.
    fn iterations(&self) -> Option<usize> {
        pcg_iterations(&self.apply, &self.solve, &self.rhs(), 1e-10, 100)
    }
}

/// `graph` factored whole through its CSR.
fn whole(graph: &Graph, config: Config) -> Vec<Path<'_>> {
    let factor = Entry::Csr.reduce(graph, Types::F64U32).factorize(&config);
    vec![Path {
        name: "whole block",
        n: graph.n,
        solve: Box::new(move |r| factor.solve(r)),
        apply: Box::new(|x| graph.apply(x)),
    }]
}

/// `g` factored whole and onto either side.
fn bipartite(g: &Bipartite, config: Config) -> Vec<Path<'static>> {
    let policy = ScalingPolicy::default();
    let complement = |side| factorize_complement(g.view(), side, config, policy).unwrap();
    vec![
        Path::dense(
            "whole",
            factorize_with(g.view(), config, policy).unwrap(),
            g.dense(),
        ),
        Path::dense("left", complement(Side::Left), g.complement(Side::Left)),
        Path::dense("right", complement(Side::Right), g.complement(Side::Right)),
    ]
}

fn all_bits(paths: &[Path<'_>]) -> Vec<Vec<u64>> {
    paths.iter().map(Path::bits).collect()
}

/// Each path's bits equal under `like` and differ under `unlike`.
fn assert_paths(actual: &[Path<'_>], like: &[Path<'_>], unlike: &[Path<'_>], label: &str) {
    for ((actual, like), unlike) in actual.iter().zip(like).zip(unlike) {
        assert_eq!(actual.bits(), like.bits(), "{label}, {}", actual.name);
        assert_ne!(actual.bits(), unlike.bits(), "{label}, {}", actual.name);
    }
}

fn assert_every_path_converges(paths: &[Path<'_>], label: &str) {
    for path in paths {
        let iterations = path.iterations();
        assert!(iterations.is_some(), "{label}, {}", path.name);
    }
}

/// A frustrated cycle whose pendants' `1e-200` surplus lies within the twist band, each pendant joined to two cycle vertices.
fn cycle(cycle: usize, pendants: usize) -> Bipartite {
    Cycle {
        cycle,
        pendants,
        wide: 2,
        right_pendants: 0,
    }
    .bipartite()
}

#[test]
fn a_balanced_block_solves_alike_under_every_surplus() {
    let mut grid = light_row(10, 1e-40);
    grid.edges.iter_mut().for_each(|edge| edge.positive = false);
    let [ground, edges, auto] = each(sparse(SchurMode::Same), |c| all_bits(&whole(&grid, c)));
    assert_eq!(ground, auto);
    assert_eq!(edges, auto);
    let mut g = Bipartite::frustrated();
    let (_, y) = g.values.as_mut().expect("weighted");
    y.iter_mut().for_each(|y| *y = -y.abs());
    for complement in [SchurMode::Exact, SchurMode::Same] {
        let [ground, edges, auto] = each(sparse(complement), |c| all_bits(&bipartite(&g, c)));
        assert_eq!(ground, auto, "{complement:?}");
        assert_eq!(edges, auto, "{complement:?}");
    }
}

#[test]
fn a_frustrated_block_on_the_dense_arm_solves_alike_under_every_surplus() {
    let grid = light_row(4, 1e-40);
    let [ground, edges, auto] = each(Config::default(), |c| all_bits(&whole(&grid, c)));
    assert_eq!(ground, auto);
    assert_eq!(edges, auto);
    let [ground, _, auto] = each(sparse(SchurMode::Same), |c| all_bits(&whole(&grid, c)));
    assert_ne!(ground, auto, "the approximate arm tells them apart");
    let g = cycle(8, 7);
    let [ground, edges, auto] = each(Config::default(), |c| all_bits(&bipartite(&g, c)));
    assert_eq!(ground, auto);
    assert_eq!(edges, auto);
}

#[test]
fn inside_the_twist_band_auto_carries_the_surplus_on_edges_and_the_ground_vertex_stalls() {
    let grid = light_row(10, 1e-40);
    let [ground, edges, auto] = each(sparse(SchurMode::Same), |c| whole(&grid, c));
    assert_paths(&auto, &edges, &ground, "grid");
    assert_every_path_converges(&auto, "Auto");
    assert_every_path_converges(&edges, "SurplusEdges");
    // As a ground the surplus gives the cover a common mode ~1e-43 of the trace, and PCG stalls at ‖r‖ = ‖b‖.
    assert_eq!(ground[0].iterations(), None);
    let g = cycle(30, 25);
    for complement in [SchurMode::Exact, SchurMode::Same] {
        let [ground, edges, auto] = each(sparse(complement), |c| bipartite(&g, c));
        assert_paths(&auto, &edges, &ground, &format!("{complement:?}"));
    }
}

#[test]
fn past_the_twist_band_auto_carries_the_surplus_to_the_ground_vertex_and_every_surplus_converges() {
    let grid = light_row(10, 1.0);
    let [ground, edges, auto] = each(sparse(SchurMode::Same), |c| whole(&grid, c));
    assert_paths(&auto, &ground, &edges, "grid");
    for (paths, surplus) in [&ground, &edges, &auto].into_iter().zip(SURPLUSES) {
        assert_every_path_converges(paths, &format!("grid, {surplus:?}"));
    }
    let g = Bipartite::frustrated();
    for complement in [SchurMode::Exact, SchurMode::Same] {
        let [ground, edges, auto] = each(sparse(complement), |c| bipartite(&g, c));
        assert_paths(&auto, &ground, &edges, &format!("{complement:?}"));
        for (paths, surplus) in [&ground, &edges, &auto].into_iter().zip(SURPLUSES) {
            assert_every_path_converges(paths, &format!("{complement:?}, {surplus:?}"));
        }
    }
}
