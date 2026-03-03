mod common;

use approx_chol::{Builder, Config};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use common::{grid_laplacian, OrPanic};

fn bench_approx_chol_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("approx_chol_build");
    group.sample_size(10);

    for size in [100, 200] {
        let lap = grid_laplacian(size, size);
        let config = Config::default();
        let builder = Builder::new(config);

        group.bench_with_input(
            BenchmarkId::new("grid", format!("{}x{}", size, size)),
            &lap,
            |b, lap| {
                b.iter(|| {
                    builder
                        .build(lap.as_csr())
                        .or_panic("factorization should succeed")
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_approx_chol_build);
criterion_main!(benches);
