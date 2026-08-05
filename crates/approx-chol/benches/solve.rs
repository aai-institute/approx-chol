mod common;

use std::hint::black_box;
use std::time::Duration;

use approx_chol::low_level::Builder;
use approx_chol::{Config, CsrRef, Factor};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use common::grid_laplacian;

/// `k` interleaved path Laplacians: vertex `i` neighbours `i - k` and `i + k`, so
/// component membership maximally interleaves with input numbering. The only shape
/// where the block permutation is non-identity, and the worst case for it.
struct InterleavedPaths {
    row_ptrs: Vec<u32>,
    col_indices: Vec<u32>,
    values: Vec<f64>,
    n: u32,
}

impl InterleavedPaths {
    fn as_csr(&self) -> Result<CsrRef<'_>, approx_chol::Error> {
        CsrRef::new(&self.row_ptrs, &self.col_indices, &self.values, self.n)
    }
}

fn interleaved_paths(n: usize, k: usize) -> InterleavedPaths {
    let mut row_ptrs = Vec::with_capacity(n + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    row_ptrs.push(0);
    for vertex in 0..n {
        let mut degree = 0.0f64;
        if vertex >= k {
            col_indices.push((vertex - k) as u32);
            values.push(-1.0);
            degree += 1.0;
        }
        let diagonal_slot = col_indices.len();
        col_indices.push(vertex as u32);
        values.push(0.0);
        if vertex + k < n {
            col_indices.push((vertex + k) as u32);
            values.push(-1.0);
            degree += 1.0;
        }
        values[diagonal_slot] = degree;
        row_ptrs.push(col_indices.len() as u32);
    }
    InterleavedPaths {
        row_ptrs,
        col_indices,
        values,
        n: n as u32,
    }
}

fn bench_solve_for_size(c: &mut Criterion, size: usize) {
    let lap = grid_laplacian(size, size);
    let factor: Factor<f64> = Builder::new(Config::default())
        .build(lap.as_csr().expect("grid_laplacian must build valid CSR"))
        .expect("factorization should succeed");
    let n = factor.n();

    let mut rhs = vec![0.0f64; n];
    rhs[0] = 1.0;
    rhs[n - 1] = -1.0;

    let mut group = c.benchmark_group(format!("solve_grid_{size}x{size}"));
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(1));

    let mut work_projected = vec![0.0f64; n];
    group.bench_with_input(BenchmarkId::new("solve_into", n), &n, |b, _| {
        b.iter(|| {
            factor
                .solve_into(black_box(&rhs), black_box(&mut work_projected))
                .expect("solve_into should succeed");
            black_box(&work_projected);
        });
    });

    let mut work_in_place = vec![0.0f64; n];
    group.bench_with_input(BenchmarkId::new("solve_in_place", n), &n, |b, _| {
        b.iter(|| {
            work_in_place.copy_from_slice(&rhs);
            factor
                .solve_in_place(black_box(&mut work_in_place))
                .expect("solve_in_place should succeed");
            black_box(&work_in_place);
        });
    });

    group.finish();
}

/// Guards the per-component path: the permutation round trip plus one block solve
/// per component, against the connected grid solves above.
fn bench_disconnected_solve(c: &mut Criterion, n: usize, k: usize) {
    let lap = interleaved_paths(n, k);
    let factor: Factor<f64> = Builder::new(Config::default())
        .build(
            lap.as_csr()
                .expect("interleaved paths must build valid CSR"),
        )
        .expect("disconnected factorization should succeed");
    let dim = factor.n();

    let mut rhs = vec![0.0f64; dim];
    rhs[0] = 1.0;
    rhs[dim - 1] = -1.0;
    let mut work = vec![0.0f64; dim];

    let mut group = c.benchmark_group(format!("disconnected_solve_k{k}"));
    group.sample_size(50);
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(2));
    group.bench_with_input(BenchmarkId::new("solve_into", n), &n, |b, _| {
        b.iter(|| {
            factor
                .solve_into(black_box(&rhs), black_box(&mut work))
                .expect("solve_into should succeed");
            black_box(&work);
        });
    });
    group.finish();
}

fn bench_solve(c: &mut Criterion) {
    bench_solve_for_size(c, 100);
    bench_solve_for_size(c, 200);
    for n in [64usize, 256, 1024, 4096, 16384] {
        bench_disconnected_solve(c, n, 4);
    }
}

criterion_group!(benches, bench_solve);
criterion_main!(benches);
