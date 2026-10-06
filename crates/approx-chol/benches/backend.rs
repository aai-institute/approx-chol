mod common;

use approx_chol::low_level::Builder;
use approx_chol::{Backend, Config, ExactFailure, Factor};
use common::grid::GridLaplacian;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

type Shape = (&'static str, fn(usize) -> GridLaplacian);

/// Dense cost depends only on `n` and the sampler's on fill, so these bracket the crossover.
fn shapes() -> [Shape; 2] {
    [
        ("path", |n| common::grid_laplacian(1, n)),
        ("complete", complete_laplacian),
    ]
}

/// The densest block either backend can be handed.
fn complete_laplacian(n: usize) -> GridLaplacian {
    let mut row_ptrs = Vec::with_capacity(n + 1);
    let mut col_indices = Vec::with_capacity(n * n);
    let mut values = Vec::with_capacity(n * n);
    row_ptrs.push(0);
    for row in 0..n {
        for col in 0..n {
            col_indices.push(col as u32);
            values.push(if row == col { (n - 1) as f64 } else { -1.0 });
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

/// Unbounded `max_dim`: `Backend::default` would route the larger sizes to the other arm.
fn backends() -> [(&'static str, Backend); 2] {
    [
        ("approximate", Backend::Approximate),
        (
            "exact",
            Backend::ExactBelow {
                max_dim: usize::MAX,
                on_failure: ExactFailure::FallBackToApproximate,
            },
        ),
    ]
}

const DIMENSIONS: [usize; 7] = [8, 16, 24, 32, 64, 128, 256];

fn bench_backend_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("backend_build");
    for (shape, build_lap) in shapes() {
        for n in DIMENSIONS {
            let lap = build_lap(n);
            // CSR validation would swamp the exact arm at the small sizes compared here.
            let csr = lap.as_csr().expect("valid CSR");
            for (label, backend) in backends() {
                let builder = Builder::<f64>::new(Config {
                    backend,
                    ..Config::default()
                });
                let id = BenchmarkId::new(format!("{shape}/{label}"), n);
                group.bench_with_input(id, &csr, |b, csr| {
                    b.iter(|| builder.build(*csr).expect("factorization should succeed"));
                });
            }
        }
    }
    group.finish();
}

fn bench_backend_solve(c: &mut Criterion) {
    let mut group = c.benchmark_group("backend_solve");
    for (shape, build_lap) in shapes() {
        for n in DIMENSIONS {
            let lap = build_lap(n);
            let mut rhs = vec![0.0; n];
            rhs[0] = 1.0;
            rhs[n - 1] = -1.0;
            for (label, backend) in backends() {
                let factor: Factor<f64> = Builder::new(Config {
                    backend,
                    ..Config::default()
                })
                .build(lap.as_csr().expect("valid CSR"))
                .expect("factorization should succeed");
                let mut work = vec![0.0; factor.n()];
                let mut scratch = vec![0.0; factor.scratch_len()];

                let id = BenchmarkId::new(format!("{shape}/{label}"), n);
                group.bench_with_input(id, &rhs, |b, rhs| {
                    b.iter(|| {
                        work.copy_from_slice(rhs);
                        factor
                            .solve_in_place(&mut work, &mut scratch)
                            .expect("solve should succeed")
                    });
                });
            }
        }
    }
    group.finish();
}

criterion_group!(benches, bench_backend_build, bench_backend_solve);
criterion_main!(benches);
