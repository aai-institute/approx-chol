//! Each frustrated arm on each fixture: PCG iterations, `nnz(L)`, work and the build's peak memory.

#[path = "common/csr.rs"]
#[allow(
    dead_code,
    reason = "frustrated fixtures are signed, never a Laplacian"
)]
mod csr;
#[path = "common/frustrated.rs"]
mod frustrated;
#[path = "common/panel.rs"]
#[allow(dead_code, reason = "the frustrated fixtures draw only slope panels")]
mod panel;
#[path = "../tests/integration/common/pcg.rs"]
#[allow(dead_code, reason = "the bench reads only the iteration count")]
mod pcg;
#[path = "common/peak.rs"]
mod peak;

use approx_chol::{ApproxInverse, Config, Threads};
use frustrated::{arms, fixtures, Fixture};
use pcg::pcg_iterations;
use peak::peak_while;
use serde_json::Value;

/// Vertices per fixture, unless `FRUSTRATED_AB_N` says otherwise.
const N: usize = 100_000;
const SEEDS: u64 = 5;
const TOLERANCE: f64 = 1e-8;
const MAX_ITERATIONS: usize = 2000;

/// Stored entries of `L`, diagonal included, read from the factor's serde payload: each block's dense packed rows or approximate steps' headers and neighbors, its eliminated rows and their pivots, and the lone vertices' pivots; panics on a payload of another shape.
fn nnz_l(factor: &ApproxInverse) -> usize {
    let payload = serde_json::to_value(factor).expect("serializes");
    let len = |value: &Value, what: &str| match value.as_array() {
        Some(items) => items.len(),
        None => panic!("the payload has no {what}"),
    };
    let body = &payload[1];
    let blocks = (body[1].as_object())
        .and_then(|by_index| by_index.values().next())
        .expect("blocks under their index type");
    let mut count = len(&body[2], "lone pivots");
    for block in blocks.as_array().expect("a list of blocks") {
        let triangular = block["cholesky"]["triangular"].as_object();
        let steps = |steps: &Value| {
            len(&steps["headers"], "headers") + len(&steps["neighbors"], "neighbors")
        };
        count += match triangular.and_then(|t| t.iter().next()) {
            Some((kind, dense)) if kind == "Dense" => len(&dense[0], "packed rows"),
            Some((kind, pinned)) if ["Folded", "Cover"].contains(&kind.as_str()) => {
                steps(&pinned[0])
            }
            Some((kind, signed)) if kind == "Signed" => steps(signed),
            other => panic!("a triangular factor, not {other:?}"),
        };
        count += match &block["elimination"] {
            Value::Null => 0,
            elimination => {
                len(&elimination["cross"]["indices"], "eliminated rows")
                    + len(&elimination["inv_pivot"], "eliminated pivots")
            }
        };
    }
    assert!(count > 0, "a factor stores entries");
    count
}

fn median(mut values: Vec<usize>) -> usize {
    values.sort_unstable();
    values[values.len() / 2]
}

/// One arm's line: medians over the seeds, and the largest peak.
fn measure(name: &str, fixture: &Fixture, b: &[f64], arm: &str, config: Config) -> String {
    let nnz_a = fixture.nnz();
    let (mut iterations, mut nnz, mut work, mut peak) = (vec![], vec![], vec![], 0);
    for seed in 0..SEEDS {
        let (factor, bytes) = peak_while(|| fixture.factor(Config { seed, ..config }));
        peak = peak.max(bytes);
        let precondition = |r: &[f64]| {
            let mut z = r.to_vec();
            factor.apply(&mut z, Threads::All).expect("applies");
            z
        };
        let apply = |x: &[f64]| fixture.apply(x);
        let taken = pcg_iterations(apply, precondition, b, TOLERANCE, MAX_ITERATIONS);
        let taken = taken.unwrap_or(usize::MAX);
        let entries = nnz_l(&factor);
        iterations.push(taken);
        nnz.push(entries);
        work.push(taken.saturating_mul(entries + nnz_a));
    }
    let shown = |v: usize| match v {
        usize::MAX => "-".to_string(),
        v => v.to_string(),
    };
    format!(
        "{name}\t{arm}\t{}\t{}\t{nnz_a}\t{}\t{peak}",
        shown(median(iterations)),
        median(nnz),
        shown(median(work))
    )
}

fn main() {
    let n = std::env::var("FRUSTRATED_AB_N").map_or(N, |n| n.parse().expect("a vertex count"));
    println!("FRUSTRATED_AB n={n} seeds={SEEDS} tolerance={TOLERANCE:e} max={MAX_ITERATIONS}");
    println!("fixture\tarm\titerations\tnnz_L\tnnz_A\twork\tpeak_bytes");
    for (name, fixture) in fixtures(n) {
        let want: Vec<f64> = (0..fixture.n())
            .map(|i| ((7 * i + 3) as f64).sin())
            .collect();
        let b = fixture.apply(&want);
        for (arm, config) in arms() {
            println!("{}", measure(&name, &fixture, &b, &arm, config));
        }
    }
}
