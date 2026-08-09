//! Headline factorization for the non-blocking wall-clock alert (#60). Prints
//! `WALLCLOCK_BEST_NS=<ns>` for `scripts/check-wallclock-reference.py`.
//!
//! Best-of-N, not mean: runner contention can only slow a run, so the minimum is
//! the cleanest estimate. Banded at degree 12 rather than the degree-4 grid every
//! other bench uses, because ingestion is ~17% of the build at degree 4 and
//! negligible past it — degree 12 is where `within`'s factors actually sit, so the
//! tracked number moves with the sampler rather than with ingestion.

mod common;

use std::hint::black_box;
use std::time::Instant;

use approx_chol::low_level::Builder;
use approx_chol::Config;
use common::grid::GridLaplacian;

const N: usize = 160_000;
const HALF_BANDWIDTH: usize = 6;
const RUNS: usize = 5;

/// Banded Laplacian on a path: vertex `i` joins `i ± 1 ..= i ± HALF_BANDWIDTH`.
fn banded_laplacian(n: usize, half: usize) -> GridLaplacian {
    let mut row_ptrs = Vec::with_capacity(n + 1);
    let mut col_indices = Vec::with_capacity(n * (2 * half + 1));
    let mut values = Vec::with_capacity(n * (2 * half + 1));
    row_ptrs.push(0);
    for row in 0..n {
        let lo = row.saturating_sub(half);
        let hi = (row + half).min(n - 1);
        for col in lo..=hi {
            col_indices.push(col as u32);
            values.push(if col == row { (hi - lo) as f64 } else { -1.0 });
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
        best = best.min(start.elapsed().as_nanos());
        black_box(&factor);
    }

    println!("WALLCLOCK_BEST_NS={best}");
}
