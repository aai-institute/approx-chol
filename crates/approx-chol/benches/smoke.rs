mod common;

use std::time::Duration;

use approx_chol::{Builder, Config, SplitMerge};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};

use common::{grid_laplacian, GridLaplacian};

fn run_build_and_solve(lap: &GridLaplacian, config: Config) {
    let builder = Builder::new(config);
    let factor = builder.build(lap.as_csr()).unwrap();

    let n = factor.n();
    let mut rhs = vec![0.0; n];
    rhs[0] = 1.0;
    rhs[n - 1] = -1.0;
    let mut work = vec![0.0; n];
    factor.solve_into(&rhs, &mut work);
    assert!(work.iter().all(|x| x.is_finite()));
}

fn bench_approx_chol_smoke(c: &mut Criterion) {
    let mut group = c.benchmark_group("approx_chol_mini_smoke");
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_millis(50));
    group.measurement_time(Duration::from_millis(100));

    for size in [70, 100] {
        let lap = grid_laplacian(size, size);
        let ac = Config::default();
        let ac2 = Config {
            seed: 42,
            split_merge: Some(SplitMerge { split: 2, merge: 2 }),
            ..Default::default()
        };

        group.bench_with_input(
            BenchmarkId::new("AC", format!("{size}x{size}")),
            &lap,
            |b, lap| {
                b.iter(|| run_build_and_solve(lap, ac));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("AC2", format!("{size}x{size}")),
            &lap,
            |b, lap| {
                b.iter(|| run_build_and_solve(lap, ac2));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_approx_chol_smoke);
criterion_main!(benches);
