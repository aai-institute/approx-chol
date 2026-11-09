#[path = "common/csr.rs"]
mod csr;
#[path = "common/grid.rs"]
mod grid;
#[path = "common/panel.rs"]
mod panel;

use std::hint::black_box;
use std::time::Duration;

use approx_chol::{
    factorize, ApplyError, ApproxInverse, ColumnMajor, Columns, Config, MatrixMut, MemBuffer,
    MemStack, Threads,
};
use criterion::{criterion_group, criterion_main, Bencher, BenchmarkId, Criterion};
use csr::{build, OwnedCsr};
use grid::grid_laplacian;
use panel::Panel;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// The solve alone: `x` and the scratch allocated once, `b` copied in per run.
fn timed_solve(b: &mut Bencher, factor: &ApproxInverse, rhs: &[f64]) {
    timed_block(b, factor, 1, Threads::All, rhs.to_vec(), |x, stack| {
        factor.apply_with(x, Threads::All, stack)
    });
}

/// Column `j` of a block over `rhs`, scaled apart from the others.
fn scaled(rhs: &[f64], j: usize) -> impl Iterator<Item = f64> + '_ {
    (rhs.iter().enumerate()).map(move |(i, &v)| v * (1.0 + ((i + j) % 3) as f64))
}

/// A block of `columns` columns stored as `data`, applied through `apply`: the data and the scratch allocated once, the right-hand sides copied in per run.
fn timed_block<D: Clone>(
    b: &mut Bencher,
    factor: &ApproxInverse,
    columns: usize,
    threads: Threads,
    data: D,
    apply: impl Fn(&mut D, &mut MemStack) -> Result<(), ApplyError>,
) {
    let mut block = data.clone();
    let mut scratch = MemBuffer::new(factor.scratch(columns, threads));
    b.iter(|| {
        block.clone_from(black_box(&data));
        apply(&mut block, MemStack::new(&mut scratch)).expect("apply");
        black_box(&block);
    });
}

/// A `K`-column block applied in place.
fn timed_apply<const K: usize>(
    b: &mut Bencher,
    factor: &ApproxInverse,
    rhs: &[f64],
    threads: Threads,
) {
    let mut rows = vec![[0.0; K]; rhs.len()];
    for j in 0..K {
        (rows.iter_mut().zip(scaled(rhs, j))).for_each(|(row, v)| row[j] = v);
    }
    timed_block(b, factor, K, threads, rows, |rows, stack| {
        factor.apply_with(rows, threads, stack)
    });
}

/// Columns stored apart, reached only through `read_column` and `write_column`.
struct Apart<'c>(&'c mut [Vec<f64>]);

impl MatrixMut<f64> for Apart<'_> {
    fn rows(&self) -> usize {
        self.0.first().map_or(0, Vec::len)
    }

    fn columns(&self) -> usize {
        self.0.len()
    }

    fn read_column(&self, column: usize, into: &mut [f64]) {
        into.copy_from_slice(&self.0[column]);
    }

    fn write_column(&mut self, column: usize, from: &[f64]) {
        self.0[column].copy_from_slice(from);
    }
}

/// How a block stores its columns one by one.
#[derive(Clone, Copy)]
enum Layout {
    ColumnMajor,
    Columns,
    Apart,
}

/// A `k`-column block stored column by column in `layout`.
fn timed_columns(
    b: &mut Bencher,
    factor: &ApproxInverse,
    rhs: &[f64],
    k: usize,
    threads: Threads,
    layout: Layout,
) {
    let n = rhs.len();
    let columns: Vec<Vec<f64>> = (0..k).map(|j| scaled(rhs, j).collect()).collect();
    let apply = |block: &mut dyn MatrixMut<f64>, stack: &mut MemStack| {
        factor.apply_with(block, threads, stack)
    };
    match layout {
        Layout::ColumnMajor => {
            timed_block(b, factor, k, threads, columns.concat(), |data, stack| {
                apply(
                    &mut ColumnMajor::new(data, n).expect("n rows to a column"),
                    stack,
                )
            })
        }
        Layout::Columns => timed_block(b, factor, k, threads, columns, |columns, stack| {
            apply(&mut Columns::new(columns).expect("n rows each"), stack)
        }),
        Layout::Apart => timed_block(b, factor, k, threads, columns, |columns, stack| {
            apply(&mut Apart(columns), stack)
        }),
    }
}

/// `k` interleaved paths: the only shape whose blocks are not contiguous.
fn interleaved_paths(n: usize, k: usize) -> OwnedCsr {
    OwnedCsr::laplacian((0..n).map(|vertex| {
        [vertex.checked_sub(k), Some(vertex + k).filter(|&u| u < n)]
            .into_iter()
            .flatten()
    }))
}

fn bench_solve_for_size(c: &mut Criterion, size: usize) {
    let lap = grid_laplacian(size, size);
    let factor = build(lap.csr(), Config::default());
    let n = factor.n();

    let mut rhs = vec![0.0f64; n];
    rhs[0] = 1.0;
    rhs[n - 1] = -1.0;

    let mut group = c.benchmark_group(format!("solve_grid_{size}x{size}"));
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(1));

    group.bench_function(BenchmarkId::new("solve", n), |b| {
        timed_solve(b, &factor, &rhs)
    });

    group.finish();
}

/// Guards the per-block gather, solve and scatter against the grid solves above.
fn bench_disconnected_solve(c: &mut Criterion, n: usize, k: usize) {
    let lap = interleaved_paths(n, k);
    let factor = build(lap.csr(), Config::default());
    let dim = factor.n();

    let mut rhs = vec![0.0f64; dim];
    rhs[0] = 1.0;
    rhs[dim - 1] = -1.0;

    let mut group = c.benchmark_group(format!("disconnected_solve_k{k}"));
    group.sample_size(50);
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(2));
    group.bench_function(BenchmarkId::new("solve", n), |b| {
        timed_solve(b, &factor, &rhs)
    });
    group.finish();
}

/// A worker-firm panel as `within` passes it: the worker side is eliminated, the firm side's complement factored.
fn bench_panel_solve(c: &mut Criterion, workers: usize, firms: usize) {
    let mut rng = SmallRng::seed_from_u64(0x5EED);
    let panel = Panel::new(workers, firms, &mut rng);
    let factor = factorize(panel.gram()).expect("factors");
    let rhs: Vec<f64> = (0..factor.n())
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();

    let mut group = c.benchmark_group("panel_solve");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(2));
    group.bench_function(
        BenchmarkId::new("solve", format!("{workers}x{firms}")),
        |b| timed_solve(b, &factor, &rhs),
    );
    if workers >= 200_000 {
        let one = Threads::Count(std::num::NonZeroUsize::MIN);
        for (budget, threads) in [("all", Threads::All), ("one", one)] {
            let id = |k: usize| {
                BenchmarkId::new(format!("apply_k{k}_{budget}"), format!("{workers}x{firms}"))
            };
            group.bench_function(id(1), |b| timed_apply::<1>(b, &factor, &rhs, threads));
            group.bench_function(id(4), |b| timed_apply::<4>(b, &factor, &rhs, threads));
            group.bench_function(id(8), |b| timed_apply::<8>(b, &factor, &rhs, threads));
        }
    }
    group.finish();
    if workers >= 200_000 {
        bench_column_layouts(c, &factor, &rhs, &format!("{workers}x{firms}"));
    }
}

/// Layouts stored column by column on the panel, against `panel_solve`'s interleaved `apply_k*`: below, at and past each pack's width.
fn bench_column_layouts(c: &mut Criterion, factor: &ApproxInverse, rhs: &[f64], panel: &str) {
    let mut group = c.benchmark_group("column_layouts");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(2));
    let one = Threads::Count(std::num::NonZeroUsize::MIN);
    let layouts = [
        ("column_major", Layout::ColumnMajor),
        ("columns", Layout::Columns),
        ("apart", Layout::Apart),
    ];
    for (budget, threads) in [("all", Threads::All), ("one", one)] {
        for k in [1, 3, 4, 7, 8] {
            for (name, layout) in layouts {
                let id = BenchmarkId::new(format!("{name}_k{k}_{budget}"), panel);
                group.bench_function(id, |b| timed_columns(b, factor, rhs, k, threads, layout));
            }
        }
    }
    group.finish();
}

/// `components` interleaved paths of `len` vertices: a budget spread over many small listed blocks, as within's pairs split.
fn bench_many_components(c: &mut Criterion, components: usize, len: usize) {
    let lap = interleaved_paths(components * len, components);
    let factor = build(lap.csr(), Config::default());
    let rhs: Vec<f64> = (0..factor.n()).map(|i| (i % 7) as f64 - 3.0).collect();
    let mut group = c.benchmark_group("many_components");
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(2));
    let one = Threads::Count(std::num::NonZeroUsize::MIN);
    for (budget, threads) in [("all", Threads::All), ("one", one)] {
        let id = |k: usize| {
            BenchmarkId::new(
                format!("apply_k{k}_{budget}"),
                format!("{components}x{len}"),
            )
        };
        group.bench_function(id(1), |b| timed_apply::<1>(b, &factor, &rhs, threads));
        group.bench_function(id(4), |b| timed_apply::<4>(b, &factor, &rhs, threads));
    }
    group.finish();
}

fn bench_solve(c: &mut Criterion) {
    // 1000x1000 holds 5M nonzeros, so its timing runs out of cache.
    for size in [100, 200, 1000] {
        bench_solve_for_size(c, size);
    }
    for n in [64usize, 256, 1024, 4096, 16384] {
        bench_disconnected_solve(c, n, 4);
    }
    // Below and past the elimination's split grain of 2^17 multiply-adds.
    for (workers, firms) in [(20_000, 1_000), (200_000, 10_000)] {
        bench_panel_solve(c, workers, firms);
    }
    bench_many_components(c, 1000, 100);
}

criterion_group!(benches, bench_solve);
criterion_main!(benches);
