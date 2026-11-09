#[path = "common/csr.rs"]
mod csr;
#[path = "common/grid.rs"]
mod grid;
#[path = "common/powerlaw.rs"]
mod powerlaw;
#[path = "common/stars.rs"]
mod stars;

use std::num::NonZeroU32;

use approx_chol::{factorize_with, Config, Sampling, ScalingPolicy};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use csr::build;
use grid::grid_laplacian;
use powerlaw::barabasi_albert;
use stars::{Stars, STARS};

fn bench_factorization_grid(c: &mut Criterion) {
    let mut group = c.benchmark_group("factorization_grid");
    group.sample_size(10);

    let ac2 = Config {
        sampling: Sampling::CliqueTree {
            copies: NonZeroU32::new(2).unwrap(),
        },
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
    let mut group = c.benchmark_group("bipartite_stars");
    for degree in [4usize, 10, 32] {
        let stars = Stars::new(degree);
        let gram = stars.gram();

        group.throughput(criterion::Throughput::Elements(STARS as u64));
        for (label, copies) in [("AC", 1), ("AC2 k=4", 4)] {
            // Dense would take the complement exact: the kept block has only `degree` vertices.
            let config = Config {
                sampling: Sampling::CliqueTree {
                    copies: NonZeroU32::new(copies).unwrap(),
                },
                exact_max_dim: 0,
                ..Config::default()
            };
            group.bench_with_input(BenchmarkId::new(label, degree), &gram, |b, &gram| {
                b.iter(|| factorize_with(gram, config, ScalingPolicy::default()).expect("factors"));
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
