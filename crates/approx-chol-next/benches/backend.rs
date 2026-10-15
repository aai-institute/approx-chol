#[path = "common/csr.rs"]
mod csr;
#[path = "common/grid.rs"]
mod grid;

use approx_chol_next::{factorize, Config};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use csr::{reduce, OwnedCsr};

type Shape = (&'static str, fn(usize) -> OwnedCsr);

/// Dense cost depends only on `n` and the sampler's on fill, so these bracket the crossover.
fn shapes() -> [Shape; 2] {
    [
        ("path", |n| grid::grid_laplacian(1, n)),
        ("complete", complete_laplacian),
    ]
}

/// The densest block either Cholesky can be handed.
fn complete_laplacian(n: usize) -> OwnedCsr {
    OwnedCsr::from_rows((0..n).map(|row| {
        (0..n)
            .map(|col| (col, if row == col { (n - 1) as f64 } else { -1.0 }))
            .collect()
    }))
}

/// Unbounded `dense_up_to`: the default would route the larger sizes to the other arm.
fn choleskies() -> [(&'static str, Config); 2] {
    [
        (
            "approximate",
            Config {
                dense_up_to: 0,
                ..Config::default()
            },
        ),
        (
            "exact",
            Config {
                dense_up_to: usize::MAX,
                ..Config::default()
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
            // CSR shape checks would swamp the exact arm at the small sizes compared here.
            let csr = lap.csr();
            for (label, config) in choleskies() {
                let id = BenchmarkId::new(format!("{shape}/{label}"), n);
                group.bench_with_input(id, &csr, |b, csr| {
                    b.iter(|| factorize(&reduce(*csr), &config));
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
            for (label, config) in choleskies() {
                let factor = factorize(&reduce(lap.csr()), &config);
                let id = BenchmarkId::new(format!("{shape}/{label}"), n);
                group.bench_with_input(id, &rhs, |b, rhs| {
                    b.iter(|| factor.solve(rhs).expect("solve should succeed"));
                });
            }
        }
    }
    group.finish();
}

criterion_group!(benches, bench_backend_build, bench_backend_solve);
criterion_main!(benches);
