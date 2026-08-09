mod common;

use std::hint::black_box;
use std::time::Instant;

use approx_chol::low_level::Builder;
use approx_chol::Config;
use common::grid::GridLaplacian;

const N: usize = 160_000;
// Degree 12, not the grid benches' 4: ingestion is ~17% of the build at degree 4
// and negligible past it, and `within`'s factors sit here.
const HALF_BANDWIDTH: usize = 6;
const RUNS: usize = 9;

fn banded_laplacian(n: usize, half: usize) -> GridLaplacian {
    let mut row_ptrs = Vec::with_capacity(n + 1);
    let mut col_indices = Vec::with_capacity(n * (2 * half + 1));
    let mut values = Vec::with_capacity(n * (2 * half + 1));
    row_ptrs.push(0);
    for row in 0..n {
        let lo = row.saturating_sub(half);
        let hi = (row + half).min(n - 1);
        let degree = (hi - lo) as f64;
        for col in lo..=hi {
            col_indices.push(col as u32);
            values.push(if col == row { degree } else { -1.0 });
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

fn main() {
    let lap = banded_laplacian(N, HALF_BANDWIDTH);
    let csr = lap.as_csr().expect("banded_laplacian must build valid CSR");
    let builder = Builder::<f64>::new(Config::default());

    let mut best = u128::MAX;
    for _ in 0..RUNS {
        let start = Instant::now();
        let factor = builder.build(csr).expect("factorization should succeed");
        // Min, not mean: contention only ever slows a run.
        best = best.min(start.elapsed().as_nanos());
        black_box(&factor);
    }

    // Derived from the consts so a retuned workload cannot keep comparing against
    // a reference measured on the old one.
    println!("WALLCLOCK_WORKLOAD=banded/n={N}/half={HALF_BANDWIDTH}/runs={RUNS}");
    println!("WALLCLOCK_BEST_NS={best}");
}
