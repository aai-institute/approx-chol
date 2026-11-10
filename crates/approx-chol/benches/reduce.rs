#[path = "common/csr.rs"]
#[allow(dead_code, reason = "`build` and `laplacian` serve the factor benches")]
mod csr;

use std::time::Duration;

use approx_chol::{ScalingPolicy, Sgdd, Threads};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};
use csr::OwnedCsr;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// `D L D`, `L` a ring plus `chords` random edges a vertex with weights over `decades`, `D` over two: singular and short of dominance, so `Sgdd::reduce` runs the Perron inverse iteration.
fn scaled_ring(n: usize, chords: usize, decades: f64, seed: u64) -> OwnedCsr {
    let mut rng = SmallRng::seed_from_u64(seed);
    let d: Vec<f64> = (0..n)
        .map(|_| 10f64.powf(rng.random_range(0.0..2.0)))
        .collect();
    let mut edges = Vec::with_capacity(n * (chords + 1));
    for i in 0..n {
        for k in 0..=chords {
            let j = match k {
                0 => (i + 1) % n,
                _ => rng.random_range(0..n),
            };
            let w = 10f64.powf(rng.random_range(0.0..decades));
            if j != i {
                edges.push((i, j, w));
            }
        }
    }
    OwnedCsr::congruent_laplacian(&d, edges)
}

/// #362's case: about five edges a vertex, weights over six decades; an iteration takes seconds.
fn bench_reduce_scaled_ring(c: &mut Criterion) {
    let mut group = c.benchmark_group("sgdd_reduce");
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_secs(1));
    for (n, label) in [(100_000, "100k"), (400_000, "400k")] {
        let a = scaled_ring(n, 4, 6.0, 0x362);
        let csr = a.csr();
        group.bench_with_input(BenchmarkId::new("w6", label), &csr, |b, &csr| {
            b.iter(|| Sgdd::reduce(csr, ScalingPolicy::default(), Threads::All).expect("reduces"));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_reduce_scaled_ring);
criterion_main!(benches);
