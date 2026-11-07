#[path = "common/csr.rs"]
mod csr;
#[path = "common/grid.rs"]
mod grid;

use std::hint::black_box;
use std::time::Duration;

use approx_chol::input::{BipartiteFactorWidth2, FactorWidth1};
use approx_chol::{factorize, ApproxInverse, Config, MemBuffer, MemStack, Threads};
use criterion::{criterion_group, criterion_main, Bencher, BenchmarkId, Criterion};
use csr::{build, OwnedCsr};
use grid::grid_laplacian;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// The solve alone: `x` and the scratch allocated once, `b` copied in per run.
fn timed_solve(b: &mut Bencher, factor: &ApproxInverse, rhs: &[f64]) {
    let mut x = rhs.to_vec();
    let mut scratch = MemBuffer::new(factor.scratch(1, Threads::All));
    b.iter(|| {
        x.copy_from_slice(black_box(rhs));
        factor
            .apply_with(&mut x, Threads::All, MemStack::new(&mut scratch))
            .expect("solve");
        black_box(&x);
    });
}

/// A `K`-column block applied in place: the rows and the scratch allocated once, the right-hand sides copied in per run.
fn timed_apply<const K: usize>(
    b: &mut Bencher,
    factor: &ApproxInverse,
    rhs: &[f64],
    threads: Threads,
) {
    let columns: Vec<[f64; K]> = (rhs.iter().enumerate())
        .map(|(i, &v)| std::array::from_fn(|j| v * (1.0 + ((i + j) % 3) as f64)))
        .collect();
    let mut rows = columns.clone();
    let mut scratch = MemBuffer::new(factor.scratch(K, threads));
    b.iter(|| {
        rows.copy_from_slice(black_box(&columns));
        factor
            .apply_with(&mut rows, threads, MemStack::new(&mut scratch))
            .expect("apply");
        black_box(&rows);
    });
}

/// `k` interleaved paths: the only shape whose blocks are not contiguous.
fn interleaved_paths(n: usize, k: usize) -> OwnedCsr {
    OwnedCsr::laplacian((0..n).map(|vertex| {
        [vertex.checked_sub(k), Some(vertex + k).filter(|&u| u < n)]
            .into_iter()
            .flatten()
    }))
}

fn bench_solve_for_size(c: &mut Criterion, size: usize) {
    let lap = grid_laplacian(size, size);
    let factor = build(lap.csr(), Config::default());
    let n = factor.n();

    let mut rhs = vec![0.0f64; n];
    rhs[0] = 1.0;
    rhs[n - 1] = -1.0;

    let mut group = c.benchmark_group(format!("solve_grid_{size}x{size}"));
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(1));

    group.bench_function(BenchmarkId::new("solve", n), |b| {
        timed_solve(b, &factor, &rhs)
    });

    group.finish();
}

/// Guards the per-block gather, solve and scatter against the grid solves above.
fn bench_disconnected_solve(c: &mut Criterion, n: usize, k: usize) {
    let lap = interleaved_paths(n, k);
    let factor = build(lap.csr(), Config::default());
    let dim = factor.n();

    let mut rhs = vec![0.0f64; dim];
    rhs[0] = 1.0;
    rhs[dim - 1] = -1.0;

    let mut group = c.benchmark_group(format!("disconnected_solve_k{k}"));
    group.sample_size(50);
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(2));
    group.bench_function(BenchmarkId::new("solve", n), |b| {
        timed_solve(b, &factor, &rhs)
    });
    group.finish();
}

/// A worker-firm panel as `within` passes it: the worker side is eliminated, the firm side's complement factored.
fn bench_panel_solve(c: &mut Criterion, workers: usize, firms: usize) {
    const OBSERVATIONS_PER_WORKER: usize = 5;
    let mut rng = SmallRng::seed_from_u64(0x5EED);
    let worker: Vec<u32> = (0..workers * OBSERVATIONS_PER_WORKER)
        .map(|i| (i / OBSERVATIONS_PER_WORKER) as u32)
        .collect();
    let firm: Vec<u32> = (0..worker.len())
        .map(|_| rng.random_range(0..firms as u32))
        .collect();
    let gram = BipartiteFactorWidth2::new(
        FactorWidth1::new(workers, &worker, None).expect("valid side"),
        FactorWidth1::new(firms, &firm, None).expect("valid side"),
    )
    .expect("valid gram");
    let factor = factorize(gram).expect("factors");
    let rhs: Vec<f64> = (0..factor.n())
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();

    let mut group = c.benchmark_group("panel_solve");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(2));
    group.bench_function(
        BenchmarkId::new("solve", format!("{workers}x{firms}")),
        |b| timed_solve(b, &factor, &rhs),
    );
    if workers >= 200_000 {
        let one = Threads::Count(std::num::NonZeroUsize::MIN);
        for (budget, threads) in [("all", Threads::All), ("one", one)] {
            let id = |k: usize| {
                BenchmarkId::new(format!("apply_k{k}_{budget}"), format!("{workers}x{firms}"))
            };
            group.bench_function(id(1), |b| timed_apply::<1>(b, &factor, &rhs, threads));
            group.bench_function(id(4), |b| timed_apply::<4>(b, &factor, &rhs, threads));
            group.bench_function(id(8), |b| timed_apply::<8>(b, &factor, &rhs, threads));
        }
    }
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
    // Below and past the elimination's split grain of 2^17 multiply-adds.
    for (workers, firms) in [(20_000, 1_000), (200_000, 10_000)] {
        bench_panel_solve(c, workers, firms);
    }
}

criterion_group!(benches, bench_solve);
criterion_main!(benches);
