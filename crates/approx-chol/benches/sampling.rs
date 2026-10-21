#[path = "common/csr.rs"]
mod csr;
#[path = "common/grid.rs"]
mod grid;

use approx_chol::input::{BipartiteFactorWidth2, FactorWidth1};
use approx_chol::{factorize_with, Backend, Config};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use csr::{build, OwnedCsr};
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

fn bench_factorization_grid(c: &mut Criterion) {
    let mut group = c.benchmark_group("factorization_grid");
    group.sample_size(10);

    let ac2 = Config {
        split_merge: Some(2),
        ..Config::default()
    };
    // 1000x1000 holds 5M nonzeros, so its timing runs out of cache.
    for &size in &[50, 100, 200, 1000] {
        let lap = grid_laplacian(size, size);
        for (label, config) in [("AC", Config::default()), ("AC2", ac2)] {
            group.bench_with_input(
                BenchmarkId::new(label, format!("{size}x{size}")),
                &lap,
                |b, lap| b.iter(|| build(lap.csr(), config)),
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
            b.iter(|| build(lap.csr(), config));
        });
    }
    group.finish();
}

/// A whole factorize of a bipartite block, reduction included, not main's bare sampler: no sampler is public.
fn bench_bipartite_stars(c: &mut Criterion) {
    const STARS: usize = 10_000;

    let mut group = c.benchmark_group("bipartite_stars");
    for degree in [4usize, 10, 32] {
        let mut rng = SmallRng::seed_from_u64(0x5EED);
        // Every star on the same `degree` kept vertices, as main's; only scales vary.
        let stars: Vec<u32> = (0..STARS * degree).map(|i| (i / degree) as u32).collect();
        let kept: Vec<u32> = (0..STARS * degree).map(|i| (i % degree) as u32).collect();
        let scales: Vec<f64> = (0..STARS * degree)
            .map(|_| rng.random_range(0.25..1.25f64).sqrt())
            .collect();
        let gram = BipartiteFactorWidth2::scaled(
            FactorWidth1::new(STARS, &stars, None).expect("valid side"),
            FactorWidth1::new(degree, &kept, None).expect("valid side"),
            &scales,
        )
        .expect("valid gram");

        group.throughput(criterion::Throughput::Elements(STARS as u64));
        for (label, split_merge) in [("AC", None), ("AC2 k=4", Some(4))] {
            // Dense would take the complement exact: the kept block has only `degree` vertices.
            let config = Config {
                split_merge,
                backend: Backend::Approximate,
                ..Config::default()
            };
            group.bench_with_input(BenchmarkId::new(label, degree), &gram, |b, &gram| {
                b.iter(|| factorize_with(gram, config).expect("factors"));
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
