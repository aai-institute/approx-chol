mod common;

use std::hint::black_box;
use std::time::Duration;

use approx_chol::CsrRef;
use criterion::{criterion_group, criterion_main, Criterion};

use common::grid_laplacian;

fn bench_to_owned_u32_for_size(c: &mut Criterion, size: usize) {
    let lap = grid_laplacian(size, size);
    let csr_u32 = lap.as_csr();

    let row_ptrs_usize: Vec<usize> = lap.row_ptrs.iter().map(|&v| v as usize).collect();
    let col_indices_usize: Vec<usize> = lap.col_indices.iter().map(|&v| v as usize).collect();
    let csr_usize =
        CsrRef::new(&row_ptrs_usize, &col_indices_usize, &lap.values, lap.n).expect("valid CSR");

    let mut group = c.benchmark_group(format!("csr_to_owned_u32_{size}x{size}"));
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(1));

    group.bench_function("u32_passthrough_baseline", |b| {
        b.iter(|| {
            let csr = black_box(csr_u32);
            black_box(csr.row_ptrs().len());
        });
    });

    group.bench_function("u32_input_to_owned_u32", |b| {
        b.iter(|| {
            let converted = black_box(csr_u32)
                .to_owned_u32()
                .expect("u32 indices must fit in u32");
            black_box(converted.as_ref().row_ptrs().len());
        });
    });

    group.bench_function("usize_input_to_owned_u32", |b| {
        b.iter(|| {
            let converted = black_box(csr_usize)
                .to_owned_u32()
                .expect("usize indices from this grid must fit in u32");
            black_box(converted.as_ref().row_ptrs().len());
        });
    });

    group.finish();
}

fn bench_to_owned_u32(c: &mut Criterion) {
    bench_to_owned_u32_for_size(c, 100);
    bench_to_owned_u32_for_size(c, 200);
}

criterion_group!(benches, bench_to_owned_u32);
criterion_main!(benches);
