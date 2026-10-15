#[path = "common/csr.rs"]
mod csr;
#[path = "common/grid.rs"]
mod grid;

use std::hint::black_box;
use std::time::Duration;

use approx_chol_next::{factorize, Config};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use csr::{reduce, OwnedCsr};
use grid::grid_laplacian;

/// `k` interleaved paths: the only shape whose blocks are not contiguous.
fn interleaved_paths(n: usize, k: usize) -> OwnedCsr {
    OwnedCsr::from_rows((0..n).map(|vertex| {
        let mut row = Vec::with_capacity(3);
        if vertex >= k {
            row.push((vertex - k, -1.0));
        }
        let diagonal = row.len();
        row.push((vertex, 0.0));
        if vertex + k < n {
            row.push((vertex + k, -1.0));
        }
        row[diagonal].1 = (row.len() - 1) as f64;
        row
    }))
}

fn bench_solve_for_size(c: &mut Criterion, size: usize) {
    let lap = grid_laplacian(size, size);
    let factor = factorize(&reduce(lap.csr()), &Config::default());
    let n = factor.n();

    let mut rhs = vec![0.0f64; n];
    rhs[0] = 1.0;
    rhs[n - 1] = -1.0;

    let mut group = c.benchmark_group(format!("solve_grid_{size}x{size}"));
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(1));

    group.bench_with_input(BenchmarkId::new("solve", n), &n, |b, _| {
        b.iter(|| black_box(factor.solve(black_box(&rhs)).expect("solve should succeed")));
    });

    group.finish();
}

/// Guards the per-block gather, solve and scatter against the grid solves above.
fn bench_disconnected_solve(c: &mut Criterion, n: usize, k: usize) {
    let lap = interleaved_paths(n, k);
    let factor = factorize(&reduce(lap.csr()), &Config::default());
    let dim = factor.n();

    let mut rhs = vec![0.0f64; dim];
    rhs[0] = 1.0;
    rhs[dim - 1] = -1.0;

    let mut group = c.benchmark_group(format!("disconnected_solve_k{k}"));
    group.sample_size(50);
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(2));
    group.bench_with_input(BenchmarkId::new("solve", n), &n, |b, _| {
        b.iter(|| black_box(factor.solve(black_box(&rhs)).expect("solve should succeed")));
    });
    group.finish();
}

fn bench_solve(c: &mut Criterion) {
    // 1000x1000 holds 5M nonzeros, so its timing runs out of cache.
    for size in [100, 200, 1000] {
        bench_solve_for_size(c, size);
    }
    for n in [64usize, 256, 1024, 4096, 16384] {
        bench_disconnected_solve(c, n, 4);
    }
}

criterion_group!(benches, bench_solve);
criterion_main!(benches);
