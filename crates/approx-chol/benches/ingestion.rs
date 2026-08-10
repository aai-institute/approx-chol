//! Full build over a degree sweep, the denominator every ingestion change is read
//! against. There is no ingestion-only arm: since #84 the augmentation decision
//! precedes graph construction, so the deficient matrix that used to isolate
//! ingestion now aborts before building anything.

mod common;

use std::hint::black_box;
use std::time::Duration;

use approx_chol::low_level::Builder;
use approx_chol::Config;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use common::shapes::sweep;

fn bench_shapes(c: &mut Criterion) {
    let mut group = c.benchmark_group("ingest");
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(3));

    let builder = Builder::<f64>::new(Config::default());
    for (label, lap) in sweep() {
        let csr = lap.as_csr_ref();
        builder.build(csr).expect("sweep shapes must factorize");
        group.bench_function(BenchmarkId::new("full_build", label), |b| {
            b.iter(|| black_box(builder.build(csr)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_shapes);
criterion_main!(benches);
