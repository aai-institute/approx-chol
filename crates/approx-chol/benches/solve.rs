mod common;

use std::hint::black_box;
use std::time::Duration;

use approx_chol::{Builder, Config, Factor};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use common::{grid_laplacian, OrPanic};

fn bench_solve_for_size(c: &mut Criterion, size: usize) {
    let lap = grid_laplacian(size, size);
    let factor: Factor<f64> = Builder::new(Config::default())
        .build(lap.as_csr())
        .or_panic("factorization should succeed");
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
                .or_panic("solve_into should succeed");
            black_box(&work_projected);
        });
    });

    let mut work_no_projection = vec![0.0f64; n];
    group.bench_with_input(
        BenchmarkId::new("solve_into_no_projection", n),
        &n,
        |b, _| {
            b.iter(|| {
                factor
                    .solve_into_with_projection(
                        black_box(&rhs),
                        black_box(&mut work_no_projection),
                        false,
                    )
                    .or_panic("solve_into_with_projection should succeed");
                black_box(&work_no_projection);
            });
        },
    );

    let mut work_in_place = vec![0.0f64; n];
    group.bench_with_input(BenchmarkId::new("solve_in_place", n), &n, |b, _| {
        b.iter(|| {
            work_in_place.copy_from_slice(&rhs);
            factor
                .solve_in_place(black_box(&mut work_in_place))
                .or_panic("solve_in_place should succeed");
            black_box(&work_in_place);
        });
    });

    group.finish();
}

fn bench_solve(c: &mut Criterion) {
    bench_solve_for_size(c, 100);
    bench_solve_for_size(c, 200);
}

criterion_group!(benches, bench_solve);
criterion_main!(benches);
