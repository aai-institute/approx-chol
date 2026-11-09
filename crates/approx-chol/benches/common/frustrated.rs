use std::collections::{HashMap, HashSet};
use std::num::NonZeroU32;

use approx_chol::{ApproxInverse, Config, CoverSurplus, FrustratedRoute, Sampling};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use crate::csr::{build, OwnedCsr};
use crate::panel::Panel;

/// Every approximate route a frustrated block takes, by name.
const ROUTES: [(&str, FrustratedRoute); 4] = [
    (
        "cover-ground",
        FrustratedRoute::Cover(CoverSurplus::GroundVertex),
    ),
    (
        "cover-edges",
        FrustratedRoute::Cover(CoverSurplus::SurplusEdges),
    ),
    ("cover-auto", FrustratedRoute::Cover(CoverSurplus::Auto)),
    ("signed", FrustratedRoute::Signed),
];

/// Copies per edge: AC, then AC2.
const COPIES: [u32; 2] = [1, 2];

/// Each route under each copy count, by name.
pub fn arms() -> Vec<(String, Config)> {
    let arm = |(name, route): (&str, FrustratedRoute)| {
        COPIES.map(move |copies| {
            let config = Config {
                frustrated: route,
                sampling: Sampling::CliqueTree {
                    copies: NonZeroU32::new(copies).expect("a copy"),
                },
                ..Config::default()
            };
            (format!("{name}/k{copies}"), config)
        })
    };
    ROUTES.into_iter().flat_map(arm).collect()
}

#[derive(Clone, Copy, Debug)]
pub enum Weights {
    Unit,
    /// `2^u`, `u` uniform in `[−10, 10)`.
    LogUniform,
}

impl Weights {
    fn draw(self, rng: &mut SmallRng) -> f64 {
        match self {
            Self::Unit => 1.0,
            Self::LogUniform => rng.random_range(-10.0..10.0f64).exp2(),
        }
    }
}

/// A frustrated input and its operator `A`.
pub enum Fixture {
    Csr(OwnedCsr),
    Slopes(Panel),
}

impl Fixture {
    pub fn n(&self) -> usize {
        match self {
            Self::Csr(csr) => csr.csr().n(),
            Self::Slopes(panel) => panel.n(),
        }
    }

    /// `A`'s off-diagonal entries `(i, j, a_ij)`, `i < j`, duplicates summed.
    fn off_diagonal(&self) -> HashMap<(usize, usize), f64> {
        let mut entries = HashMap::new();
        match self {
            Self::Csr(csr) => {
                let csr = csr.csr();
                for i in 0..csr.n() {
                    let span = csr.row_ptrs()[i] as usize..csr.row_ptrs()[i + 1] as usize;
                    let row = csr.col_indices()[span.clone()]
                        .iter()
                        .zip(&csr.values()[span]);
                    for (&j, &a) in row.filter(|&(&j, _)| j as usize > i) {
                        *entries.entry((i, j as usize)).or_insert(0.0) += a;
                    }
                }
            }
            Self::Slopes(panel) => {
                for (worker, firm, value) in panel.observations() {
                    *entries.entry((worker, firm)).or_insert(0.0) += value;
                }
            }
        }
        entries.retain(|_, a| *a != 0.0);
        entries
    }

    /// `A`'s stored nonzeros, both triangles.
    pub fn nnz(&self) -> usize {
        match self {
            Self::Csr(csr) => csr.csr().values().len(),
            Self::Slopes(panel) => panel.n() + 2 * self.off_diagonal().len(),
        }
    }

    pub fn factor(&self, config: Config) -> ApproxInverse {
        match self {
            Self::Csr(csr) => build(csr.csr(), config),
            Self::Slopes(panel) => build(panel.gram(), config),
        }
    }

    pub fn apply(&self, x: &[f64]) -> Vec<f64> {
        match self {
            Self::Csr(csr) => csr.matvec(x),
            Self::Slopes(panel) => {
                let mut out = vec![0.0; x.len()];
                for (worker, firm, value) in panel.observations() {
                    let dot = x[worker] + value * x[firm];
                    out[worker] += dot;
                    out[firm] += value * dot;
                }
                out
            }
        }
    }

    /// Whether no `±1` diagonal congruence makes every off-diagonal nonpositive: a positive entry joins two vertices of opposite sign, a negative one two alike.
    fn frustrated(&self) -> bool {
        let mut parent: Vec<(usize, bool)> = (0..self.n()).map(|v| (v, false)).collect();
        fn root(parent: &mut [(usize, bool)], v: usize) -> (usize, bool) {
            let (mut top, mut parity) = (v, false);
            while parent[top].0 != top {
                (top, parity) = (parent[top].0, parity != parent[top].1);
            }
            let (mut at, mut rest) = (v, parity);
            while parent[at].0 != at {
                let (up, flip) = parent[at];
                parent[at] = (top, rest);
                (at, rest) = (up, rest != flip);
            }
            (top, parity)
        }
        self.off_diagonal().into_iter().any(|((i, j), a)| {
            let ((ri, si), (rj, sj)) = (root(&mut parent, i), root(&mut parent, j));
            let unlike = a > 0.0;
            if ri == rj {
                return (si != sj) != unlike;
            }
            parent[ri] = (rj, (si != sj) != unlike);
            false
        })
    }
}

/// Every fixture near `n` vertices, by name; panics on one that a congruence would balance.
pub fn fixtures(n: usize) -> Vec<(String, Fixture)> {
    let side2 = (n as f64).sqrt().round() as usize;
    let side3 = (n as f64).cbrt().round() as usize;
    let mut fixtures = Vec::new();
    for weights in [Weights::Unit, Weights::LogUniform] {
        for positive in [0.001, 0.01, 0.1, 0.5] {
            let name = format!("grid2d_{side2}x{side2}/f{positive}/{weights:?}");
            fixtures.push((name, grid(&[side2, side2], positive, weights)));
        }
    }
    let name = format!("grid3d_{side3}^3/f0.1/Unit");
    fixtures.push((name, grid(&[side3, side3, side3], 0.1, Weights::Unit)));
    let name = format!("random_{n}_m4n/f0.5/LogUniform");
    fixtures.push((name, random(n, 0.5, Weights::LogUniform)));
    let mut rng = SmallRng::seed_from_u64(0x510E);
    let workers = n * 19 / 20;
    let slopes = Panel::with_slope(workers, n - workers, &mut rng);
    fixtures.push((
        format!("slopes_{workers}x{}", n - workers),
        Fixture::Slopes(slopes),
    ));
    // Kept side within the default `exact_max_dim`, so its complement is formed dense.
    let slopes = Panel::with_slope(workers, 24, &mut rng);
    fixtures.push((format!("slopes_{workers}x24"), Fixture::Slopes(slopes)));
    for (name, fixture) in &fixtures {
        assert!(fixture.frustrated(), "{name} is balanced at n = {n}");
    }
    fixtures
}

/// Edges `(a, b)` weighted by `weights`, a `positive` fraction of them positive off-diagonals, and 1% of the vertices grounded, as SDD CSR.
fn sdd(
    n: usize,
    edges: impl Iterator<Item = (usize, usize)>,
    positive: f64,
    weights: Weights,
) -> Fixture {
    let mut rng = SmallRng::seed_from_u64(0xF5);
    let mut rows: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
    let mut diagonal = vec![0.0; n];
    for (a, b) in edges {
        let w = weights.draw(&mut rng);
        let sign = if rng.random_bool(positive) { 1.0 } else { -1.0 };
        rows[a].push((b as u32, sign * w));
        rows[b].push((a as u32, sign * w));
        diagonal[a] += w;
        diagonal[b] += w;
    }
    for (vertex, row) in rows.iter_mut().enumerate() {
        let ground = if rng.random_bool(0.01) {
            weights.draw(&mut rng)
        } else {
            0.0
        };
        row.push((vertex as u32, diagonal[vertex] + ground));
        row.sort_unstable_by_key(|&(j, _)| j);
    }
    Fixture::Csr(OwnedCsr::from_rows(rows))
}

/// A grid of `dims`, row-major.
fn grid(dims: &[usize], positive: f64, weights: Weights) -> Fixture {
    let n: usize = dims.iter().product();
    let strides: Vec<usize> = (0..dims.len())
        .map(|d| dims[d + 1..].iter().product())
        .collect();
    let edges = (0..n).flat_map(|v| {
        (0..dims.len())
            .filter(|&d| (v / strides[d]) % dims[d] + 1 < dims[d])
            .map(|d| (v, v + strides[d]))
            .collect::<Vec<_>>()
    });
    sdd(n, edges, positive, weights)
}

/// A spanning path plus random edges to `m = 4n` in all, no pair twice: `n ≥ 9`, or `4n` pairs do not exist.
fn random(n: usize, positive: f64, weights: Weights) -> Fixture {
    assert!(
        4 * n <= n * (n - 1) / 2,
        "{n} vertices hold fewer than 4n pairs"
    );
    let mut rng = SmallRng::seed_from_u64(0xA11);
    let mut seen: HashSet<(usize, usize)> = (1..n).map(|v| (v - 1, v)).collect();
    let mut edges: Vec<_> = (1..n).map(|v| (v - 1, v)).collect();
    while edges.len() < 4 * n {
        let (a, b) = (rng.random_range(0..n), rng.random_range(0..n));
        let pair = (a.min(b), a.max(b));
        if a != b && seen.insert(pair) {
            edges.push(pair);
        }
    }
    sdd(n, edges.into_iter(), positive, weights)
}
