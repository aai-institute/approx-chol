#[path = "common/csr.rs"]
mod csr;

use std::hint::black_box;
use std::time::Instant;

use approx_chol_next::{factorize, Config};
use csr::{reduce, OwnedCsr};

const N: usize = 160_000;
// Degree 12, where `within`'s factors sit; at degree 4 ingestion is ~17% of the build.
const HALF_BANDWIDTH: usize = 6;
const RUNS: usize = 9;

fn banded_laplacian(n: usize, half: usize) -> OwnedCsr {
    OwnedCsr::from_rows((0..n).map(|row| {
        let lo = row.saturating_sub(half);
        let hi = (row + half).min(n - 1);
        let degree = (hi - lo) as f64;
        (lo..=hi)
            .map(|col| (col, if col == row { degree } else { -1.0 }))
            .collect()
    }))
}

fn main() {
    let lap = banded_laplacian(N, HALF_BANDWIDTH);
    let config = Config::default();

    let mut best = u128::MAX;
    for _ in 0..RUNS {
        let start = Instant::now();
        let factor = factorize(&reduce(lap.csr()), &config);
        // Min, not mean: contention only ever slows a run.
        best = best.min(start.elapsed().as_nanos());
        black_box(&factor);
    }

    // Whole `Config` so a retuned bench never compares against a stale reference.
    let config = format!("{config:?}").replace(' ', "");
    println!("WALLCLOCK_WORKLOAD=banded/n={N}/half={HALF_BANDWIDTH}/runs={RUNS}/f64/{config}");
    println!("WALLCLOCK_BEST_NS={best}");
}
