//! The full sweep behind #435: every fixture and route over `ε`, printed as a table, asserting nothing; the pivot column needs `serde`.
//! Columns: whether a component floats (`/S` signed); smallest pivot and `uᵀAu` over `max a_ii`; `uᵀAu · uᵀP⁻¹u`, near 1 where `P` tracks `A` along `u = 1/√n`; PCG iterations to 1e-8 and true residual on `b ⟂ 1` and a generic `b`, and the generic `‖x‖`; worst over seeds.
//! `SWEEP_FIXTURE=<prefix> cargo test --release -p approx-chol --all-features --test integration signed_near_balance -- --ignored --nocapture`

use approx_chol::input::Side;
use approx_chol::{Config, CoverSurplus, FrustratedRoute};

use super::{cycle, flipped_grid, right_hand_sides, state};
use crate::common::entry::{Entry, Factored, Types};
use crate::common::graph::{Edge, Graph};
use crate::common::lcg::Lcg;
use crate::common::pcg::pcg;
use crate::common::sparse::clique_tree;

/// #415's `frustrated_random` toward balance: its positive edges and its grounds scaled by `ε`.
fn random(n: usize, seed: u64, epsilon: f64) -> Graph {
    let mut lcg = Lcg(seed);
    let weight = |lcg: &mut Lcg| ((lcg.next_u64() % 1200) as f64 / 100.0 - 6.0).exp2();
    let mut edges: Vec<Edge> = (1..n)
        .map(|v| Edge::negative(v - 1, v, weight(&mut lcg)))
        .collect();
    while edges.len() < 4 * n {
        let (a, b) = (
            (lcg.next_u64() % n as u64) as usize,
            (lcg.next_u64() % n as u64) as usize,
        );
        if a != b {
            let w = weight(&mut lcg);
            let positive = lcg.next_u64() % 3 == 0;
            let w = if positive { epsilon * w } else { w };
            edges.push(Edge {
                positive,
                ..Edge::negative(a, b, w)
            });
        }
    }
    let mut graph = Graph::new(n, edges);
    (0..n)
        .step_by(10)
        .for_each(|v| graph.ground[v] = 0.5 * epsilon);
    graph
}

/// 200 left vertices of degree 6 on 80 right ones, right numbered after left, unit-free weights `1 + (k mod 7)/8` squared, every edge negative but every 97th positive at `ε`: a Gram without surplus, one signed edge per column.
fn bipartite(epsilon: f64) -> Graph {
    let mut edges = Vec::new();
    for i in 0..200 {
        for d in 0..6 {
            let k = edges.len();
            let (a, b) = (i, 200 + (i * 7 + d * 13) % 80);
            let magnitude = 1.0 + (k % 7) as f64 / 8.0;
            edges.push(match k % 97 == 0 {
                true => Edge::positive(a, b, epsilon),
                false => Edge::negative(a, b, magnitude * magnitude),
            });
        }
    }
    Graph::new(280, edges)
}

/// `graph`'s Schur complement onto `kept`, the rest independent, ungrounded and joined to each vertex once: each eliminated star's signed clique `w_a w_b / d`, its sign `σ_a σ_b`, renumbered from `kept.start`. As edges, so `A x` keeps a frustration below the diagonal's ulp, which a dense complement rounds away.
fn star_complement(graph: &Graph, kept: std::ops::Range<usize>) -> Graph {
    let mut stars = vec![Vec::new(); graph.n];
    for e in &graph.edges {
        let (v, u) = match kept.contains(&e.a) {
            true => (e.b, e.a),
            false => (e.a, e.b),
        };
        assert!(kept.contains(&u) && !kept.contains(&v), "bipartite");
        stars[v].push((u - kept.start, e.weight, e.sign()));
    }
    let mut edges = Vec::new();
    for star in stars.iter().filter(|s| !s.is_empty()) {
        let d: f64 = star.iter().map(|&(_, w, _)| w).sum();
        for (i, &(a, wa, sa)) in star.iter().enumerate() {
            for &(b, wb, sb) in &star[i + 1..] {
                assert_ne!(a, b, "one column per pair");
                edges.push(Edge {
                    a,
                    b,
                    weight: wa * wb / d,
                    positive: sa * sb < 0.0,
                });
            }
        }
    }
    Graph::new(kept.len(), edges)
}

/// One matrix: `a`, applied in `f64`, and the graph an entry point factors, whole or onto a side.
struct Fixture {
    name: &'static str,
    a: Graph,
    entry: Entry,
    stated: Graph,
    onto: Option<Side>,
}

impl Fixture {
    fn csr(name: &'static str, graph: Graph) -> Self {
        Self {
            name,
            a: graph.clone(),
            entry: Entry::Csr,
            stated: graph,
            onto: None,
        }
    }

    /// A bipartite Gram whole, or its complement `onto` a side.
    fn gram(whole: Graph, onto: Option<Side>) -> Self {
        let (a, name) = match onto {
            None => (whole.clone(), "bip whole"),
            Some(Side::Left) => (star_complement(&whole, 0..200), "bip onto L"),
            Some(Side::Right) => (star_complement(&whole, 200..280), "bip onto R"),
        };
        Self {
            name,
            a,
            entry: Entry::BipartiteFactorWidth2,
            stated: whole,
            onto,
        }
    }

    fn factorize(&self, config: Config, types: Types) -> Result<Factored, String> {
        let reduced = self.entry.reduce(&self.stated, types);
        reduced.try_factorize(&config, self.onto)
    }
}

struct Arm {
    name: &'static str,
    configs: Vec<Config>,
}

fn arms() -> Vec<Arm> {
    let approximate = |frustrated, copies, seeds: &[u64]| {
        (seeds.iter())
            .map(|&seed| Config {
                seed,
                sampling: clique_tree(copies),
                frustrated,
                exact_max_dim: 0,
                ..Config::default()
            })
            .collect()
    };
    let cover = FrustratedRoute::Cover(CoverSurplus::Auto);
    vec![
        Arm {
            name: "dense",
            configs: vec![Config {
                exact_max_dim: 2000,
                ..Config::default()
            }],
        },
        Arm {
            name: "signed/AC",
            configs: approximate(FrustratedRoute::Signed, 1, &[0, 1, 2]),
        },
        Arm {
            name: "signed/AC2",
            configs: approximate(FrustratedRoute::Signed, 2, &[0, 1, 2]),
        },
        Arm {
            name: "cover/AC",
            configs: approximate(cover, 1, &[0]),
        },
        Arm {
            name: "cover/AC2",
            configs: approximate(cover, 2, &[0]),
        },
    ]
}

fn fixtures(epsilon: f64) -> Vec<Fixture> {
    vec![
        Fixture::csr("cycle 500", cycle(500, epsilon)),
        Fixture::csr("grid 32 x1", flipped_grid(32, 1, epsilon)),
        Fixture::csr("grid 32 x5", flipped_grid(32, 5, epsilon)),
        Fixture::csr("random 800", random(800, 7, epsilon)),
        Fixture::gram(bipartite(epsilon), None),
        Fixture::gram(bipartite(epsilon), Some(Side::Left)),
        Fixture::gram(bipartite(epsilon), Some(Side::Right)),
    ]
}

/// The smallest stored pivot, NaN where one is missing or not finite and positive, or without `serde` to read it.
#[cfg_attr(not(feature = "serde"), allow(unused_variables))]
fn smallest_pivot(factored: &Factored) -> f64 {
    #[cfg(feature = "serde")]
    {
        let inverse = super::inverse_pivots(factored);
        let valid = !inverse.is_empty() && inverse.iter().all(|&d| d.is_finite() && d > 0.0);
        if valid {
            return 1.0 / inverse.iter().fold(0.0f64, |m, &d| m.max(d));
        }
    }
    f64::NAN
}

#[test]
#[ignore = "a measurement for #435: prints a table, asserts nothing"]
fn signed_near_balance_sweep() {
    // Past `ε_mach` down to underflow, where a ground could round away.
    let f64_tail = [1e-20, 1e-25, 1e-30, 1e-100, 1e-300, 1e-310, 0.0];
    let f32_tail = [1e-12, 1e-20, 1e-30, 1e-38, 1e-40, 0.0];
    let mut epsilons: Vec<(Types, f64)> = ((0..=16).map(|k| 10f64.powi(-k)))
        .chain(f64_tail)
        .map(|e| (Types::F64U32, e))
        .collect();
    let f32_head = (0..=8).map(|k| 10f64.powi(-k));
    epsilons.extend(f32_head.chain(f32_tail).map(|e| (Types::F32U32, e)));
    let only = std::env::var("SWEEP_FIXTURE").ok();
    println!(
        "fixture      | T   | eps    | arm        | state     | dmin/dmax | lam1/dmax | track    | it(perp) | res(perp) | it(gen) | res(gen) | |x|(gen)"
    );
    for (types, epsilon) in epsilons {
        let (tolerance, max) = (1e-8, 400);
        for fixture in fixtures(epsilon) {
            if only
                .as_deref()
                .is_some_and(|o| !fixture.name.starts_with(o))
            {
                continue;
            }
            let a = &fixture.a;
            let (generic, perp) = right_hand_sides(a.n);
            let ones = vec![1.0 / (a.n as f64).sqrt(); a.n];
            let a_ones = a.apply(&ones);
            let lambda = ones.iter().zip(&a_ones).map(|(u, v)| u * v).sum::<f64>();
            let diagonal_max = (0..a.n)
                .map(|i| {
                    let mut e = vec![0.0; a.n];
                    e[i] = 1.0;
                    a.apply(&e)[i]
                })
                .fold(0.0f64, f64::max);
            for arm in arms() {
                let mut row: Option<String> = None;
                let (mut worst_perp, mut worst_gen): ((usize, f64), (usize, f64)) =
                    ((0, 0.0), (0, 0.0));
                let (mut x_gen, mut pivot, mut track) = (0.0f64, f64::INFINITY, 0.0f64);
                let mut state_label = String::new();
                for &config in &arm.configs {
                    let factored = match fixture.factorize(config, types) {
                        Ok(factored) => factored,
                        Err(e) => {
                            row = Some(format!("build error: {e}"));
                            break;
                        }
                    };
                    let (floats, signed) = state(&factored);
                    state_label = format!(
                        "{}{}",
                        if floats { "float" } else { "ground" },
                        if signed { "/S" } else { "" }
                    );
                    pivot = pivot.min(smallest_pivot(&factored));
                    // `uᵀAu · uᵀP⁻¹u`: about 1 where `P` tracks `A` along `u`.
                    let p_ones = factored.solve(&ones);
                    let ratio = lambda * ones.iter().zip(&p_ones).map(|(u, v)| u * v).sum::<f64>();
                    track = track.max(ratio);
                    let solve = |r: &[f64]| factored.solve(r);
                    let run = pcg(|x| a.apply(x), solve, &perp, tolerance, max);
                    worst_perp = (
                        worst_perp.0.max(run.iterations.unwrap_or(usize::MAX)),
                        worst_perp.1.max(run.residual),
                    );
                    let run = pcg(|x| a.apply(x), solve, &generic, tolerance, max);
                    worst_gen = (
                        worst_gen.0.max(run.iterations.unwrap_or(usize::MAX)),
                        worst_gen.1.max(run.residual),
                    );
                    x_gen = x_gen.max(run.x.iter().map(|x| x * x).sum::<f64>().sqrt());
                }
                let iterations = |it: usize| match it {
                    usize::MAX => "  -".to_string(),
                    it => format!("{it:3}"),
                };
                let body = row.unwrap_or_else(|| {
                    format!(
                        "{state_label:9} | {:9.2e} | {:9.2e} | {track:8.2e} | {:>8} | {:9.2e} | {:>7} | {:8.2e} | {x_gen:8.2e}",
                        pivot / diagonal_max,
                        lambda / diagonal_max,
                        iterations(worst_perp.0),
                        worst_perp.1,
                        iterations(worst_gen.0),
                        worst_gen.1,
                    )
                });
                let scalar = match types {
                    Types::F64U32 | Types::F64U64 => "f64",
                    Types::F32U32 | Types::F32U64 => "f32",
                };
                println!(
                    "{:12} | {scalar} | {epsilon:6.0e} | {:10} | {body}",
                    fixture.name, arm.name,
                );
            }
        }
    }
}
