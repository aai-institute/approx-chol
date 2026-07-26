//! Ingestion in isolation, without widening the crate's API.
//!
//! A matrix whose *last* row is not diagonally dominant runs every ingestion pass
//! — canonicality check, diagonal scan, mirror pairing — and then errors in the
//! augmentation loop before any factorization work starts. Each shape is measured
//! both ways, aborted and completed, so an ingestion change can be read as a
//! fraction of the build it is part of rather than in the abstract.
//!
//! The aborted arm still includes `Builder::build`'s index narrowing, so it
//! overstates ingestion proper by an O(nnz) constant.

use std::hint::black_box;
use std::time::Duration;

use approx_chol::low_level::Builder;
use approx_chol::{Config, Error, OwnedCsr};
use criterion::measurement::WallTime;
use criterion::{criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion};

/// Connected canonical Laplacian where vertex `i` neighbours `i-bandwidth..=i+bandwidth`,
/// so row degree is `2 * bandwidth` independently of `n`.
///
/// With `deficient`, the last row's diagonal is short by one — the deficit that
/// aborts the build once ingestion has completed.
fn banded_laplacian(n: usize, bandwidth: usize, deficient: bool) -> OwnedCsr {
    let mut row_ptrs = Vec::with_capacity(n + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    row_ptrs.push(0);
    for vertex in 0..n {
        let low = vertex.saturating_sub(bandwidth);
        let high = (vertex + bandwidth).min(n - 1);
        let mut degree = 0.0f64;
        let diagonal_slot = col_indices.len() + (vertex - low);
        for other in low..=high {
            col_indices.push(other);
            if other == vertex {
                values.push(0.0);
            } else {
                values.push(-1.0);
                degree += 1.0;
            }
        }
        values[diagonal_slot] = if deficient && vertex + 1 == n {
            degree - 1.0
        } else {
            degree
        };
        row_ptrs.push(col_indices.len());
    }
    OwnedCsr::try_from_usize(&row_ptrs, &col_indices, &values, n)
        .expect("banded laplacian must be valid CSR")
}

/// A deficient shape aborts in the augmentation loop, once every ingestion pass
/// has run; the complete one is the denominator for that fraction. Any other
/// outcome would mean the timed region skips the passes this benchmark measures.
fn bench_shape(
    group: &mut BenchmarkGroup<'_, WallTime>,
    id: BenchmarkId,
    lap: &OwnedCsr,
    deficient: bool,
) {
    let builder = Builder::<f64>::new(Config::default());
    match (builder.build(lap.as_csr_ref()), deficient) {
        (Err(Error::NotDiagonallyDominant { row }), true) if row + 1 == lap.as_csr_ref().n() => {}
        (Ok(_), false) => {}
        (outcome, _) => panic!("unexpected build outcome: {outcome:?}"),
    }

    group.bench_function(id, |b| {
        b.iter(|| black_box(builder.build(lap.as_csr_ref())));
    });
}

/// Row degree rises at fixed `n`, then `n` rises at fixed degree. Both matter:
/// mirror pairing is per-entry, but ingestion also allocates once per row.
fn bench_shapes(c: &mut Criterion) {
    let mut group = c.benchmark_group("ingest");
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(3));

    // The last pair is within's shape: small, dense-ish, every off-diagonal present.
    for (n, bandwidth) in [
        (1024usize, 2usize),
        (1024, 16),
        (1024, 128),
        (4608, 4),
        (128, 128),
        (512, 512),
    ] {
        let label = format!("n{n}_deg{}", 2 * bandwidth.min(n - 1));
        for (arm, deficient) in [("ingest_only", true), ("full_build", false)] {
            let lap = banded_laplacian(n, bandwidth, deficient);
            bench_shape(&mut group, BenchmarkId::new(arm, &label), &lap, deficient);
        }
    }
    group.finish();
}

criterion_group!(benches, bench_shapes);
criterion_main!(benches);
