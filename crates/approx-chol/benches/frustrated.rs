#[path = "common/csr.rs"]
#[allow(
    dead_code,
    reason = "frustrated fixtures are signed, never a Laplacian"
)]
mod csr;
#[path = "common/frustrated.rs"]
#[allow(dead_code, reason = "only the A/B applies A")]
mod frustrated;
#[path = "common/panel.rs"]
#[allow(dead_code, reason = "the frustrated fixtures draw only slope panels")]
mod panel;

use std::hint::black_box;
use std::time::Duration;

use approx_chol::{MemBuffer, MemStack, Threads};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use frustrated::{arms, fixtures};

const N: usize = 40_000;

/// Build and one solve per frustrated arm, on every fixture.
fn bench_frustrated(c: &mut Criterion) {
    for (name, fixture) in fixtures(N) {
        let mut group = c.benchmark_group(format!("frustrated/{name}"));
        group.sample_size(10);
        group.warm_up_time(Duration::from_millis(200));
        let rhs: Vec<f64> = (0..fixture.n()).map(|i| (i % 7) as f64 - 3.0).collect();
        for (arm, config) in arms() {
            group.bench_function(BenchmarkId::new("build", &arm), |b| {
                b.iter(|| fixture.factor(config));
            });
            let factor = fixture.factor(config);
            let mut x = rhs.clone();
            let mut scratch = MemBuffer::new(factor.scratch(1, Threads::All));
            group.bench_function(BenchmarkId::new("apply", &arm), |b| {
                b.iter(|| {
                    x.copy_from_slice(black_box(&rhs));
                    factor
                        .apply_with(&mut x, Threads::All, MemStack::new(&mut scratch))
                        .expect("apply");
                    black_box(&x);
                });
            });
        }
        group.finish();
    }
}

criterion_group!(benches, bench_frustrated);
criterion_main!(benches);
