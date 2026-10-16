#[path = "common/csr.rs"]
mod csr;
#[path = "common/grid.rs"]
mod grid;

use approx_chol::input::{BipartiteGram, FactorWidth1, Weights};
use approx_chol::{factorize, CliqueRule, Config, Multiplicity, ScalingPolicy, Sgdd, SplitCount};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use csr::{reduce, OwnedCsr};
use grid::grid_laplacian;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use std::collections::BTreeSet;

fn barabasi_albert(n: usize, m: usize, seed: u64) -> OwnedCsr {
    assert!(m >= 1 && n > m);
    let mut rng = SmallRng::seed_from_u64(seed);

    let mut adj: Vec<BTreeSet<u32>> = vec![BTreeSet::new(); n];
    let mut degree_list: Vec<u32> = Vec::new();

    for i in 0..m {
        for j in (i + 1)..m {
            adj[i].insert(j as u32);
            adj[j].insert(i as u32);
            degree_list.push(i as u32);
            degree_list.push(j as u32);
        }
    }

    for v in m..n {
        let mut targets = BTreeSet::new();
        while targets.len() < m {
            if degree_list.is_empty() {
                let t = rng.random_range(0..v as u32);
                targets.insert(t);
            } else {
                let idx = rng.random_range(0..degree_list.len());
                targets.insert(degree_list[idx]);
            }
        }
        for &t in &targets {
            adj[v].insert(t);
            adj[t as usize].insert(v as u32);
            degree_list.push(v as u32);
            degree_list.push(t);
        }
    }

    OwnedCsr::laplacian(
        adj.iter()
            .map(|neighbors| neighbors.iter().map(|&u| u as usize)),
    )
}

fn split(k: u32) -> Multiplicity {
    Multiplicity::Split(SplitCount::new(k).expect("k ≥ 2"))
}

fn bench_factorization_grid(c: &mut Criterion) {
    let mut group = c.benchmark_group("factorization_grid");
    group.sample_size(10);

    let ac2 = Config {
        approximate: split(2),
        ..Config::default()
    };
    // 1000x1000 holds 5M nonzeros, so its timing runs out of cache.
    for &size in &[50, 100, 200, 1000] {
        let lap = grid_laplacian(size, size);
        for (label, config) in [("AC", Config::default()), ("AC2", ac2)] {
            group.bench_with_input(
                BenchmarkId::new(label, format!("{size}x{size}")),
                &lap,
                |b, lap| b.iter(|| factorize(&reduce(lap.csr()), &config)),
            );
        }
    }
    group.finish();
}

fn bench_factorization_powerlaw(c: &mut Criterion) {
    let mut group = c.benchmark_group("factorization_powerlaw");
    group.sample_size(10);

    for &(n, m, label) in &[
        (5_000, 3, "5k_m3"),
        (5_000, 10, "5k_m10"),
        (10_000, 3, "10k_m3"),
        (10_000, 10, "10k_m10"),
        (20_000, 5, "20k_m5"),
    ] {
        let lap = barabasi_albert(n, m, 0xDEAD);
        let config = Config::default();
        group.bench_with_input(BenchmarkId::new("AC", label), &lap, |b, lap| {
            b.iter(|| factorize(&reduce(lap.csr()), &config));
        });
    }
    group.finish();
}

/// A whole factorize of a bipartite block, not main's bare sampler: no sampler is public.
fn bench_bipartite_stars(c: &mut Criterion) {
    const STARS: usize = 10_000;

    let mut group = c.benchmark_group("bipartite_stars");
    for degree in [4usize, 10, 32] {
        let mut rng = SmallRng::seed_from_u64(0x5EED);
        // Every star on the same `degree` kept vertices, as main's; only weights vary.
        let stars: Vec<u32> = (0..STARS * degree).map(|i| (i / degree) as u32).collect();
        let kept: Vec<u32> = (0..STARS * degree).map(|i| (i % degree) as u32).collect();
        let weights: Vec<f64> = (0..STARS * degree)
            .map(|_| rng.random_range(0.25..1.25f64))
            .collect();
        let gram = BipartiteGram::new(
            FactorWidth1::new(STARS, &stars, None).expect("valid side"),
            FactorWidth1::new(degree, &kept, None).expect("valid side"),
            Weights::of(&weights).expect("valid weights"),
        )
        .expect("valid gram");
        let sgdd: Sgdd =
            Sgdd::from_bipartite_gram(gram, ScalingPolicy::default()).expect("reduces");

        group.throughput(criterion::Throughput::Elements(STARS as u64));
        for (label, copies) in [("AC", Multiplicity::Single), ("AC2 k=4", split(4))] {
            // Dense would take the complement exact: the kept block has only `degree` vertices.
            let config = Config {
                complement: CliqueRule::Sampled(copies),
                approximate: copies,
                dense_up_to: 0,
                ..Config::default()
            };
            group.bench_with_input(BenchmarkId::new(label, degree), &sgdd, |b, sgdd| {
                b.iter(|| factorize(sgdd, &config));
            });
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_factorization_grid,
    bench_factorization_powerlaw,
    bench_bipartite_stars,
);
criterion_main!(benches);
