mod common;

use approx_chol::low_level::Builder;
use approx_chol::Config;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use std::collections::BTreeSet;

use common::grid::GridLaplacian;
use common::{grid_laplacian, OrPanic};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Barabasi-Albert preferential-attachment graph -> Laplacian CSR.
fn barabasi_albert(n: usize, m: usize, seed: u64) -> GridLaplacian {
    assert!(m >= 1 && n > m);
    let mut rng = SmallRng::seed_from_u64(seed);

    let mut adj: Vec<BTreeSet<u32>> = vec![BTreeSet::new(); n];
    let mut degree_list: Vec<u32> = Vec::new();

    // Seed: complete graph on vertices 0..m
    for i in 0..m {
        for j in (i + 1)..m {
            adj[i].insert(j as u32);
            adj[j].insert(i as u32);
            degree_list.push(i as u32);
            degree_list.push(j as u32);
        }
    }

    // Growth: each new vertex picks m distinct targets proportional to degree
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

    // Build CSR Laplacian
    let mut row_ptrs: Vec<u32> = Vec::with_capacity(n + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    row_ptrs.push(0);

    for (v, neighbors) in adj.iter().enumerate().take(n) {
        let v_u32 = v as u32;
        let degree = neighbors.len() as f64;

        for &nbr in neighbors {
            if nbr < v_u32 {
                col_indices.push(nbr);
                values.push(-1.0);
            }
        }
        col_indices.push(v_u32);
        values.push(degree);
        for &nbr in neighbors {
            if nbr > v_u32 {
                col_indices.push(nbr);
                values.push(-1.0);
            }
        }
        row_ptrs.push(col_indices.len() as u32);
    }

    GridLaplacian {
        row_ptrs,
        col_indices,
        values,
        n: n as u32,
    }
}

// ---------------------------------------------------------------------------
// End-to-end: grid Laplacian (uniform degree ~4)
// ---------------------------------------------------------------------------

fn bench_factorization_grid(c: &mut Criterion) {
    let mut group = c.benchmark_group("factorization_grid");
    group.sample_size(10);

    for &size in &[50, 100, 200] {
        let lap = grid_laplacian(size, size);
        let config = Config::default();
        let builder = Builder::new(config);

        group.bench_with_input(
            BenchmarkId::new("AC", format!("{size}x{size}")),
            &lap,
            |b, lap| {
                b.iter(|| {
                    builder
                        .build(lap.as_csr().or_panic("grid_laplacian must build valid CSR"))
                        .or_panic("factorization should succeed")
                });
            },
        );

        let ac2_config = Config {
            split_merge: Some(2),
            ..Default::default()
        };
        let ac2_builder = Builder::new(ac2_config);

        group.bench_with_input(
            BenchmarkId::new("AC2", format!("{size}x{size}")),
            &lap,
            |b, lap| {
                b.iter(|| {
                    ac2_builder
                        .build(lap.as_csr().or_panic("grid_laplacian must build valid CSR"))
                        .or_panic("factorization should succeed")
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// End-to-end: Barabasi-Albert power-law graph (hub degrees >> 1000)
// ---------------------------------------------------------------------------

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
        let builder = Builder::new(config);

        group.bench_with_input(BenchmarkId::new("AC", label), &lap, |b, lap| {
            b.iter(|| {
                builder
                    .build(lap.as_csr().or_panic("grid_laplacian must build valid CSR"))
                    .or_panic("factorization should succeed")
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_factorization_grid,
    bench_factorization_powerlaw,
);
criterion_main!(benches);
