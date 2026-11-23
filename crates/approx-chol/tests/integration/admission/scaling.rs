use crate::common::clamps::Clamps;
use approx_chol::input::{BipartiteFactorWidth2, CsrRef, FactorWidth1, FactorWidth2, Side};
use approx_chol::{
    factorize, factorize_complement, ApproxInverse, BuildError, Config, Real, ScalingOutcome,
    ScalingPolicy, Sgdd, Threads,
};
use rstest::rstest;

use crate::common::entry::only;
use crate::common::gram::dense_csr;
use crate::common::lcg::Lcg;
use crate::common::solve::Solve;

use super::{reduce, reduce_with};

/// `D L D` for the graph `edges` on `n` vertices, `D` within `spread` of `I`: GDD by construction, `R = D⁻¹`.
fn dld(
    n: usize,
    edges: &[(usize, usize)],
    spread: f64,
    seed: usize,
) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    // Arithmetic only, no libm: the same bits on every platform.
    let d: Vec<f64> = (0..n)
        .map(|i| 1.0 + spread * (((7919 * i + 104_729 * seed) % 1000) as f64 / 1000.0 - 0.5))
        .collect();
    congruent_laplacian(&d, edges, &vec![1.0; edges.len()])
}

/// `D L_w D` with `D` and the weights log-uniform over `[2^-bits, 1)`.
fn weighted_dld(
    n: usize,
    edges: &[(usize, usize)],
    bits: u32,
    seed: u64,
) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    let mut lcg = Lcg(seed.wrapping_mul(2_654_435_761).wrapping_add(17));
    let d: Vec<f64> = (0..n).map(|_| log_uniform(&mut lcg, bits)).collect();
    let weights: Vec<f64> = edges.iter().map(|_| log_uniform(&mut lcg, bits)).collect();
    congruent_laplacian(&d, edges, &weights)
}

/// Log-uniform over `[2^-bits, 1)`; powers of two come from their bits, so no libm.
fn log_uniform(lcg: &mut Lcg, bits: u32) -> f64 {
    let exponent = (lcg.unit() * f64::from(bits)) as u64 + 1;
    (1.0 + lcg.unit()) * f64::from_bits((1023 - exponent) << 52)
}

/// `D L_w D`: off-diagonal `−w_ij d_i d_j`, diagonal `d_i (Σ_j w_ij) d_i`.
fn congruent_laplacian(
    d: &[f64],
    edges: &[(usize, usize)],
    weights: &[f64],
) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); d.len()];
    let mut degrees = vec![0.0; d.len()];
    for (&(i, j), &w) in edges.iter().zip(weights) {
        rows[i].push((j, -(w * d[i] * d[j])));
        rows[j].push((i, -(w * d[j] * d[i])));
        degrees[i] += w;
        degrees[j] += w;
    }
    let (mut row_ptrs, mut columns, mut values) = (vec![0u32], Vec::new(), Vec::new());
    for (i, row) in rows.iter_mut().enumerate() {
        row.push((i, d[i] * degrees[i] * d[i]));
        row.sort_by_key(|&(j, _)| j);
        for &(j, value) in row.iter() {
            columns.push(j as u32);
            values.push(value);
        }
        row_ptrs.push(columns.len() as u32);
    }
    (row_ptrs, columns, values)
}

fn path(n: usize) -> Vec<(usize, usize)> {
    (1..n).map(|i| (i - 1, i)).collect()
}

fn ring(n: usize) -> Vec<(usize, usize)> {
    let mut edges = path(n);
    edges.push((n - 1, 0));
    edges
}

fn grid(k: usize) -> Vec<(usize, usize)> {
    let edges = crate::common::grid::grid(k, k).edges;
    edges.iter().map(|edge| (edge.a, edge.b)).collect()
}

/// A ~5000-step CG run stalls past 4× the settle band; 6144 steps cover this path, which 2048 do not (#194).
#[rstest]
#[case::spread_1e_6(1e-6)]
#[case::spread_1e_3(1e-3)]
#[case::spread_1(1.0)]
fn a_settled_dld_path_is_not_reported(#[case] spread: f64) {
    let n = 5000;
    let (row_ptrs, columns, values) = dld(n, &path(n), spread, 0);
    let policy = ScalingPolicy::new(1e-9, 6144).expect("a valid policy");
    let factor = reduce_with(&row_ptrs, &columns, &values, policy).expect("admitted");
    let block = only(&factor);
    assert_eq!(factor.clamps(), [], "spread {spread:e}");
    assert!(!block.grounding_unproven(), "spread {spread:e}");
    assert!(block.floats(), "spread {spread:e}");
}

/// The path above, unsettled at the default 2048 steps: a reduction under 6144 settles whatever `Config` factors it.
#[test]
fn a_reduction_keeps_its_own_scaling_when_factored() {
    let n = 5000;
    let (row_ptrs, columns, values) = dld(n, &path(n), 1.0, 0);
    let csr = CsrRef::new(n, &row_ptrs, &columns, &values).expect("valid shape");
    let unsettled = factorize(csr).expect("admitted");
    assert!(only(&unsettled).grounding_unproven());
    let [clamp] = &unsettled.clamps()[..] else {
        panic!("one clamp: {:?}", unsettled.clamps())
    };
    assert_eq!(clamp.scaling, ScalingOutcome::Unsettled, "{clamp:?}");
    let scaling = ScalingPolicy::new(1e-9, 6144).expect("a valid policy");
    let factor = Sgdd::reduce(csr, scaling, Threads::All)
        .expect("admitted")
        .factor(Config::default(), Threads::All);
    assert_eq!(factor.clamps(), []);
    assert!(!only(&factor).grounding_unproven());
}

/// A rounded `D L D` triangle whose `y = 1` certifies at eigen-residual `7.7e-15`, past the band: one Noda step already meets INI 1's tolerance, and only a polish settles it; kept, row 0's `7.3e-15` surplus grounded it.
fn rounded_triangle() -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    let (b, c) = (8.000_000_000_000_12, 18.000_000_000_000_27);
    let off = -6.000_000_000_000_179;
    let values = vec![2.0, -2.0, -3.0, -2.0, b, off, -3.0, off, c];
    (vec![0, 3, 6, 9], vec![0, 1, 2, 0, 1, 2, 0, 1, 2], values)
}

/// `k × k` grid, each vertex's right then down edge, `D` and weights log-uniform over `[2^-bits, 1)`, off-diagonals raised by `1 + η`: `D L_w D` with `ρ(N) = 1 + η`.
fn weighted_raised(k: usize, eta: f64, bits: u32, seed: u64) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    let n = k * k;
    let mut lcg = Lcg(seed.wrapping_mul(2_654_435_761).wrapping_add(17));
    let d: Vec<f64> = (0..n).map(|_| log_uniform(&mut lcg, bits)).collect();
    let right = |v: usize| (v % k + 1 < k).then_some(v + 1);
    let down = |v: usize| (v + k < n).then_some(v + k);
    let edges = (0..n).flat_map(|v| {
        [right(v), down(v)]
            .into_iter()
            .flatten()
            .map(move |w| (v, w))
    });
    let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut degrees = vec![0.0; n];
    for (i, j) in edges {
        let w = log_uniform(&mut lcg, bits);
        rows[i].push((j, -w * (1.0 + eta)));
        rows[j].push((i, -w * (1.0 + eta)));
        degrees[i] += w;
        degrees[j] += w;
    }
    let (mut row_ptrs, mut columns, mut values) = (vec![0u32], Vec::new(), Vec::new());
    for (i, row) in rows.iter_mut().enumerate() {
        row.push((i, degrees[i]));
        row.sort_by_key(|&(j, _)| j);
        for &(j, x) in row.iter() {
            columns.push(j as u32);
            values.push(x * d[i] * d[j]);
        }
        row_ptrs.push(columns.len() as u32);
    }
    (row_ptrs, columns, values)
}

/// Settled within 16ε, then `F·A·F` rounds a few ε more: this grid's 17.3ε is rounding, not a deficit. The singular wide-weight grid left Noda steps no budget to settle in until they handed over near it.
#[rstest]
#[case::grid(dld(22 * 22, &grid(22), 1e-3, 30))]
#[case::triangle(rounded_triangle())]
#[case::singular_wide_weights(weighted_raised(30, 0.0, 18, 1))]
fn a_settled_scaling_with_rounding_deficit_is_not_clamped(
    #[case] (row_ptrs, columns, values): (Vec<u32>, Vec<u32>, Vec<f64>),
) {
    let factor = reduce(&row_ptrs, &columns, &values).expect("admitted");
    let block = only(&factor);
    assert_eq!(factor.clamps(), []);
    assert!(!block.grounding_unproven());
    assert!(block.floats());
}

/// Ranked by violation, a stall restart resumed from a stale best, adopted nothing, and gave up at 3568 of the budget.
#[test]
fn a_weighted_grid_settles_after_a_stall_restart() {
    let k = 24;
    let (row_ptrs, columns, values) = weighted_dld(k * k, &grid(k), 24, 2);
    let policy = ScalingPolicy::new(1e-9, 8000).expect("a valid policy");
    let factor = reduce_with(&row_ptrs, &columns, &values, policy).expect("admitted");
    assert!(!only(&factor).grounding_unproven());
}

/// A restart's CG could not beat best's eigen-residual within its patience; ending there gave up at 5144 of 12000.
#[test]
fn a_stall_before_any_adoption_keeps_iterating() {
    let k = 30;
    let (row_ptrs, columns, values) = weighted_dld(k * k, &grid(k), 24, 14);
    let policy = ScalingPolicy::new(1e-9, 12_000).expect("a valid policy");
    let factor = reduce_with(&row_ptrs, &columns, &values, policy).expect("admitted");
    assert!(!only(&factor).grounding_unproven());
}

/// A settled eigen-residual is relative to the Perron vector's peak, so a row at `μ_i` may sit `band/μ_i` off.
#[test]
fn a_settled_scaling_with_an_uneven_perron_vector_is_not_clamped() {
    let k = 30;
    let (row_ptrs, columns, values) = weighted_dld(k * k, &grid(k), 10, 0);
    let factor = reduce(&row_ptrs, &columns, &values).expect("admitted");
    assert_eq!(factor.clamps(), []);
    assert!(!only(&factor).grounding_unproven());
}

/// A `k × k` `D L D` grid with its edges raised by `1 + η`, so `ρ(N) = 1 + η`.
fn raised(k: usize, eta: f64) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    let (row_ptrs, columns, mut values) = dld(k * k, &grid(k), 1.0, 0);
    for (i, window) in row_ptrs.windows(2).enumerate() {
        let row = window[0] as usize..window[1] as usize;
        for (value, &j) in values[row.clone()].iter_mut().zip(&columns[row]) {
            if j as usize != i {
                *value *= 1.0 + eta;
            }
        }
    }
    (row_ptrs, columns, values)
}

/// The one clamp of an input under the default policy, as `(scaling, violation)`.
fn only_clamp(
    (row_ptrs, columns, values): (Vec<u32>, Vec<u32>, Vec<f64>),
) -> (ScalingOutcome, f64) {
    let factor = reduce(&row_ptrs, &columns, &values).expect("admitted");
    let [clamp] = &factor.clamps()[..] else {
        panic!("one clamp: {} of them", factor.clamps().len())
    };
    (clamp.scaling, clamp.violation)
}

/// `ρ(N) = 1 + η` within the tolerance: certified, its clamp `ρ − 1` up to rounding (#341). Below half the tolerance the first solve certified, then a shift pinned under `ρ` clamped `957×` that on grid64; above it the first shift sits under `ρ` and the block ended unsettled.
/// Wide weights: a Noda `y` with a lower top but a larger eigen-residual was ranked out, leaving a clamp of `3.1 η`.
#[rstest]
#[case::below_half_tolerance(raised(64, 1e-12), 1e-12)]
#[case::above_half_tolerance(raised(64, 7e-10), 7e-10)]
#[case::wide_below_half_tolerance(raised(150, 1e-12), 1e-12)]
#[case::wide_weights(weighted_raised(30, 3e-14, 12, 2), 3e-14)]
fn a_block_within_the_tolerance_is_clamped_by_its_excess(
    #[case] input: (Vec<u32>, Vec<u32>, Vec<f64>),
    #[case] eta: f64,
) {
    let report = only_clamp(input);
    assert_eq!(report.0, ScalingOutcome::Certified, "{report:?}");
    assert!(report.1 <= eta + 64.0 * f64::EPSILON, "{report:?}");
}

/// `ρ(N) = 1 + η` just past the tolerance: refuted, its clamp near `ρ − 1`; a Rayleigh quotient that refuted inside a Noda step dropped it and kept a clamp of `2.4e-6`.
#[test]
fn a_block_just_past_the_tolerance_is_clamped_near_its_excess() {
    let eta = 1.1e-9;
    let report = only_clamp(raised(16, eta));
    assert_eq!(report.0, ScalingOutcome::Refuted, "{report:?}");
    assert!(report.1 <= 2.0 * eta, "{report:?}");
}

/// A `k × k` unit grid with its edges raised by `1 + η`, a path of `l` vertices hung from vertex 0 by weight `w`, each tail row's diagonal `4×` its off-diagonal sum, all as `D A D`: the tail's Perron components fall to `1e-21` and below, so `low` stays near `0.2`.
fn dominant_tail(k: usize, l: usize, w: f64, eta: f64) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    let core = k * k;
    let mut edges: Vec<(usize, usize, f64)> = grid(k)
        .into_iter()
        .map(|(i, j)| (i, j, 1.0 + eta))
        .collect();
    edges.extend((0..l).map(|t| {
        (
            if t == 0 { 0 } else { core + t - 1 },
            core + t,
            if t == 0 { w } else { 1.0 },
        )
    }));
    let n = core + l;
    let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut sums = vec![0.0; n];
    for &(i, j, x) in &edges {
        rows[i].push((j, -x));
        rows[j].push((i, -x));
        let weight = if i < core && j < core { 1.0 } else { x };
        sums[i] += weight;
        sums[j] += weight;
    }
    let d: Vec<f64> = (0..n)
        .map(|i| 1.0 + ((7919 * i) % 1000) as f64 / 1000.0 - 0.5)
        .collect();
    let (mut row_ptrs, mut columns, mut values) = (vec![0u32], Vec::new(), Vec::new());
    for (i, row) in rows.iter_mut().enumerate() {
        row.push((i, if i < core { sums[i] } else { 4.0 * sums[i] }));
        row.sort_by_key(|&(j, _)| j);
        for &(j, x) in row.iter() {
            columns.push(j as u32);
            values.push(x * d[i] * d[j]);
        }
        row_ptrs.push(columns.len() as u32);
    }
    (row_ptrs, columns, values)
}

/// `ρ(N) = 1 + η` proven past the band, whatever eigen-residual some `y` shows: not settled, and certified where clamped. Codex's `f32` star `N = [[0, a, a], [a, 0, 0], [a, 0, 0]]` has a `y` with eigen-residual under the band and `ρ − 1 = 2.09e-6`; the grid's dominant tail kept `low` near `0.2`, so the endgame ran to the budget.
#[test]
fn a_block_proven_past_the_band_is_not_settled() {
    let a = 0.707_108_259_201_049_8_f64 as f32;
    let values = [1.0, -a, -a, -a, 1.0, -a, 1.0];
    let policy = ScalingPolicy::new(1e-3, 2048).expect("a valid policy");
    let star =
        reduce_with(&[0, 3, 5, 7], &[0, 1, 2, 0, 1, 0, 2], &values, policy).expect("admitted");
    // Within `f32`'s row band, so no clamp: only the float decision is left unproven.
    assert_eq!(star.clamps(), []);
    assert!(only(&star).grounding_unproven());
    let eta = 3e-10;
    let (row_ptrs, columns, values) = dominant_tail(30, 40, 1e-12, eta);
    let tail = reduce(&row_ptrs, &columns, &values).expect("admitted");
    let [clamp] = &tail.clamps()[..] else {
        panic!("one clamp: {} of them", tail.clamps().len())
    };
    let report = (clamp.scaling, clamp.violation);
    assert_eq!(report.0, ScalingOutcome::Certified, "{report:?}");
    assert!(report.1 <= eta + 64.0 * f64::EPSILON, "{report:?}");
    assert!(only(&tail).grounding_unproven());
    // Codex's `f64` triangle, `ρ − 1 ≈ 4.9e-15`: its floor passed `1 + band` at step 86 of an endgame that ran to the budget.
    let (x, y, z) = (
        -0.730_319_049_786_269_3,
        -0.067_578_704_168_472_17,
        -0.632_190_571_245_819_9,
    );
    let values = [1.0, x, y, x, 1.0, z, y, z, 1.0];
    let policy = ScalingPolicy::new(1e-9, 1024).expect("a valid policy");
    let triangle = reduce_with(&[0, 3, 6, 9], &[0, 1, 2, 0, 1, 2, 0, 1, 2], &values, policy)
        .expect("admitted");
    let [clamp] = &triangle.clamps()[..] else {
        panic!("one clamp: {} of them", triangle.clamps().len())
    };
    let report = (clamp.scaling, clamp.violation);
    assert_eq!(report.0, ScalingOutcome::Certified, "{report:?}");
    assert!(report.1 <= 4.9e-15 + 64.0 * f64::EPSILON, "{report:?}");
    assert!(only(&triangle).grounding_unproven());
}

/// `f32` rounds a singular block's `ρ(N)` ulps past `1`, so only a bracket within the `16ε` band certifies; under `1e-9` these spent 2048 steps and reported.
#[rstest]
#[case::path(path(1000), 1)]
#[case::grid(grid(30), 0)]
fn a_settled_f32_scaling_is_judged_against_its_band(
    #[case] edges: Vec<(usize, usize)>,
    #[case] seed: u64,
) {
    let n = edges
        .iter()
        .map(|&(i, j)| i.max(j) + 1)
        .max()
        .expect("edges");
    let (row_ptrs, columns, values) = weighted_dld(n, &edges, 10, seed);
    let values: Vec<f32> = values.iter().map(|&v| v as f32).collect();
    let factor = reduce(&row_ptrs, &columns, &values).expect("admitted");
    let block = only(&factor);
    assert_eq!(factor.clamps(), []);
    assert!(!block.grounding_unproven());
    assert!(block.floats());
}

/// A null vector spanning `2^28` puts `μ²` across `2^112`, past `f32`'s range when peaked at `1`; every entry of `V` is in range.
#[test]
fn a_wide_f32_gram_null_vector_scales_in_range() {
    let n = 1000u32;
    let mut lcg = Lcg(0);
    let u: Vec<f64> = (0..n).map(|_| log_uniform(&mut lcg, 28)).collect();
    let a: Vec<u32> = (0..n).collect();
    let b: Vec<u32> = (0..n).map(|i| (i + 1) % n).collect();
    let x: Vec<f32> = b.iter().map(|&j| u[j as usize] as f32).collect();
    let y: Vec<f32> = a.iter().map(|&i| -u[i as usize] as f32).collect();
    let gram = FactorWidth2::new(n as usize, (&a, &x), (&b, &y)).expect("valid columns");
    let factor = factorize(gram).expect("admitted");
    assert!(only(&factor).floats());
}

/// The forest composes each `u_i` from rounded ratios, so a long cycle's rows miss by `20–35ε` though `V` is exact (#210); a miss a few bands wide, beside that rounding, is still one.
#[rstest]
#[case::ring(ring(3000), 0, 0.0)]
#[case::ring_missed(ring(3000), 0, 1e-10)]
#[case::ring_missed_by_bands(ring(10_000), 6, 100.0 * f64::EPSILON)]
#[case::grid(grid(300), 2, 0.0)]
#[case::grid_missed(grid(300), 2, 1e-10)]
fn an_exact_gram_null_vector_reports_only_a_real_miss(
    #[case] edges: Vec<(usize, usize)>,
    #[case] seed: u64,
    #[case] miss: f64,
) {
    let n = (edges.iter())
        .map(|&(i, j)| i.max(j) + 1)
        .max()
        .expect("edges");
    let mut lcg = Lcg(seed);
    let u: Vec<f64> = (0..n).map(|_| log_uniform(&mut lcg, 24)).collect();
    let a: Vec<u32> = edges.iter().map(|&(i, _)| i as u32).collect();
    let b: Vec<u32> = edges.iter().map(|&(_, j)| j as u32).collect();
    let mut x: Vec<f64> = edges.iter().map(|&(_, j)| u[j]).collect();
    let y: Vec<f64> = edges.iter().map(|&(i, _)| -u[i]).collect();
    *x.last_mut().expect("edges") *= 1.0 + miss;
    // The missed edge `(p, q)` weighs `u_p²/Σ_{j~q} u_j²` in row `q`, which a seed can drive to `0`.
    let (p, q) = *edges.last().expect("edges");
    let neighbours: f64 = (edges.iter())
        .filter_map(|&(i, j)| (i == q).then_some(j).or((j == q).then_some(i)))
        .map(|j| u[j] * u[j])
        .sum();
    let footprint = miss * u[p] * u[p] / neighbours;
    assert!(footprint >= miss / 4.0, "footprint {footprint:e}");
    let gram = FactorWidth2::new(n, (&a, &x), (&b, &y)).expect("valid columns");
    let factor = factorize(gram).expect("admitted");
    let clamps = factor.clamps();
    assert_eq!(clamps.is_empty(), miss == 0.0, "{clamps:?}");
    assert!(
        (clamps.iter()).all(|c| c.violation > footprint / 2.0),
        "{clamps:?}"
    );
}

/// `u = 2^{0, −500, −1000, −500}`: the forest forms `2^-1000` below where FMA recovers a product's rounding, so that row's shift is unknown and its miss stays reported.
#[test]
fn a_null_vector_rounding_lost_below_range_keeps_its_report() {
    let u = [1.0, 2f64.powi(-500), 2f64.powi(-1000), 2f64.powi(-500)];
    let edges = [(0u32, 1u32), (1, 2), (2, 3), (3, 0)];
    let (a, b): (Vec<u32>, Vec<u32>) = edges.iter().copied().unzip();
    // Each column scaled so its entries are `2^±250`, keeping every diagonal in range.
    let scale = |i: u32, j: u32| 2f64.powi(250) / u[i as usize].max(u[j as usize]);
    let mut x: Vec<f64> = (edges.iter())
        .map(|&(i, j)| scale(i, j) * u[j as usize])
        .collect();
    let y: Vec<f64> = (edges.iter())
        .map(|&(i, j)| -scale(i, j) * u[i as usize])
        .collect();
    let miss = 1e-10;
    x[1] *= 1.0 + miss;
    let gram = FactorWidth2::new(4, (&a, &x), (&b, &y)).expect("valid columns");
    let factor = factorize(gram).expect("admitted");
    let [clamp] = &factor.clamps()[..] else {
        panic!("one clamp: {:?}", factor.clamps())
    };
    assert!(clamp.violation > miss / 4.0, "{clamp:?}");
}

/// What [`a_ground_is_kept_only_past_its_tracked_band`] hangs on the 3000-ring of seed 0.
#[derive(Debug)]
enum Beside {
    /// A triangle `t, t + 1, t + 2` with `u = 1`, bridged to the ring's vertex 1500 with weight `bridge²` at `t`, a ground `64ε` stated at `t + 1` and its closing column `(t + 2, t)` missing by `66ε`.
    Triangle { bridge: f64 },
    /// A ground `40ε·(u₁₄₅² + u₁₄₇²)` stated at vertex 146.
    Stated,
    /// As `Stated`, and a pendant `3000` on `2999` with `u = u₂₉₉₉·2⁻¹⁰⁰⁰`, its column scaled by `2⁴⁰⁰`: the forest forms `u` below where it tracks rounding, so row `2999`'s shift is unknown.
    Pendant,
}

/// The 3000-ring of seed 0, whose `u` rounding clamps row `0` by `~25ε` unreported (#210), its twin a surplus at `2999` that the tracked rounding explains, not a ground; a row whose shift is unknown keeps the flat clamp. A triangle's closing column missing by `66ε` leaves a ground `~33ε` of row `t + 2`, past its `21ε` band but within it less the flat clamp, twinned with a deficit at `t`: a heavy bridge keeps that deficit within `t`'s band, so the ground is kept; a light one reports it, and its twin goes with it (#513). `e_s` at the stated row `s` responds by `u_s²/Σ_i g_i u_i²` over the kept grounds `g`.
#[rstest]
#[case::beside_rounding(Beside::Triangle { bridge: 30.0 }, false, true)]
#[case::twin_of_a_reported_deficit(Beside::Triangle { bridge: 0.1 }, true, false)]
#[case::rounding_twin_explained(Beside::Stated, false, false)]
#[case::rounding_twin_beside_an_unknown_shift(Beside::Pendant, false, false)]
fn a_ground_is_kept_only_past_its_tracked_band(
    #[case] beside: Beside,
    #[case] reported: bool,
    #[case] kept: bool,
) {
    let n = 3000;
    let mut lcg = Lcg(0);
    let mut u: Vec<f64> = (0..n).map(|_| log_uniform(&mut lcg, 24)).collect();
    let mut edges = ring(n);
    let (t, m) = (n, n / 2);
    match beside {
        Beside::Triangle { .. } => {
            u.extend([1.0; 3]);
            edges.extend([(t, t + 1), (t + 1, t + 2), (t + 2, t), (m, t)]);
        }
        Beside::Stated => {}
        Beside::Pendant => {
            u.push(u[n - 1] * 2f64.powi(-1000));
            edges.push((n - 1, n));
        }
    }
    let mut a: Vec<u32> = edges.iter().map(|&(i, _)| i as u32).collect();
    let mut b: Vec<u32> = edges.iter().map(|&(_, j)| j as u32).collect();
    let mut x: Vec<f64> = edges.iter().map(|&(_, j)| u[j]).collect();
    let mut y: Vec<f64> = edges.iter().map(|&(i, _)| -u[i]).collect();
    // `33ε` of its row: past its band, within it less the ring's flat clamp.
    let miss = 66.0 * f64::EPSILON;
    let (s, stated) = match beside {
        Beside::Triangle { bridge } => {
            x[n + 2] *= 1.0 + miss;
            // `t`'s diagonal is `2 + bridge²`.
            x[n + 3] *= bridge / u[m];
            y[n + 3] *= bridge / u[m];
            // `32ε` of its row, past the `19ε` band of no entry.
            (t + 1, 64.0 * f64::EPSILON)
        }
        Beside::Stated | Beside::Pendant => {
            if matches!(beside, Beside::Pendant) {
                x[n] *= 2f64.powi(400);
                y[n] *= 2f64.powi(400);
            }
            (
                146,
                40.0 * f64::EPSILON * (u[145] * u[145] + u[147] * u[147]),
            )
        }
    };
    a.push(s as u32);
    b.push(s as u32 + 1);
    x.push(stated.sqrt());
    y.push(0.0);
    let gram = FactorWidth2::new(u.len(), (&a, &x), (&b, &y)).expect("valid columns");
    let factor = factorize(gram).expect("admitted");
    let clamps = factor.clamps();
    assert_eq!(clamps.is_empty(), !reported, "{clamps:?}");
    let mut e = vec![0.0; u.len()];
    e[s] = 1.0;
    let response = factor.solve(&e).expect("solves")[s];
    let grounds = if kept { stated + miss } else { stated };
    let ratio = response * grounds;
    assert!((ratio - 1.0).abs() < 0.05, "{ratio}");
}

#[test]
fn a_policy_reports_what_it_was_built_from() {
    let default = ScalingPolicy::default();
    assert_eq!(
        (default.tolerance(), default.max_iterations()),
        (1e-9, 2048)
    );
    let policy = ScalingPolicy::new(0.25, 7).expect("a valid policy");
    assert_eq!((policy.tolerance(), policy.max_iterations()), (0.25, 7));
}

/// Two interleaved pairs no scaling makes dominant: each is one clamp, its component's vertices ascending, listed by first vertex.
#[test]
fn clamps_list_each_component_by_its_first_vertex() {
    let row_ptrs = [0, 2, 4, 6, 8];
    let columns = [0, 2, 1, 3, 0, 2, 1, 3];
    let values = [1.0, -3.0, 1.0, -3.0, -3.0, 1.0, -3.0, 1.0];
    let factor = reduce(&row_ptrs, &columns, &values).expect("admitted clamped");
    let listed: Vec<_> = (factor.clamps().iter())
        .map(|clamp| (clamp.vertices.clone(), clamp.scaling))
        .collect();
    let refuted = ScalingOutcome::Refuted;
    assert_eq!(listed, [(vec![0, 2], refuted), (vec![1, 3], refuted)]);
}

/// `odd` on vertices 1, 3, 5 beside a dominant path on 0, 2, 4, its block's local row `i` vertex `2i + 1`: what fails at the block's row 1 must name vertex 3.
fn interleaved_error<T: Real>(odd: [[T; 3]; 3]) -> BuildError {
    let even = |i: usize, j: usize| match i.abs_diff(j) {
        0 => T::one() + T::one(),
        1 => -T::one(),
        _ => T::zero(),
    };
    let entry = |i: usize, j: usize| match (i % 2, j % 2) {
        (0, 0) => even(i / 2, j / 2),
        (1, 1) => odd[i / 2][j / 2],
        _ => T::zero(),
    };
    let dense: Vec<Vec<T>> = (0..6)
        .map(|i| (0..6).map(|j| entry(i, j)).collect())
        .collect();
    let (row_ptrs, columns, values) = dense_csr(&dense);
    reduce(&row_ptrs, &columns, &values).expect_err("out of range")
}

/// Rows 1 and 2 at diagonal `2^-960` share an edge of `2^1000`: at `y = 1` row 1's ratio `Σ_j|a_1j|/√(a_11 a_jj)` passes `f64`.
#[test]
fn a_relaxation_whose_first_ratio_overflows_is_out_of_range_at_its_vertex() {
    let (tiny, huge) = (2f64.powi(-960), 2f64.powi(1000));
    let error = interleaved_error([[1.0, -1.0, 0.0], [-1.0, tiny, -huge], [0.0, -huge, tiny]]);
    assert_eq!(error, BuildError::OutOfRange { vertex: 3 });
}

/// Rows 0 and 2 couple past dominance; their relaxation scales row 1, at `2^100`, so far down that its stored edge `−2^-100` to row 2 falls below `f32`'s floor.
#[test]
fn a_scaled_entry_below_the_floor_is_out_of_range_at_its_stored_vertex() {
    let (big, small) = (2f32.powi(100), 2f32.powi(-100));
    let error = interleaved_error([[1.0, 0.0, -3.0], [0.0, big, -small], [-3.0, -small, 1.0]]);
    assert_eq!(error, BuildError::OutOfRange { vertex: 3 });
}

/// A lone column on left 0 and right 0, then a path `p_0 … p_30` from left 1, alternating sides, column `k` loading `p_k` with `1/4` and `p_{k+1}` with `4`: its null vector falls by `16` a step, so the last columns' scaled entries fall below `f32`'s floor.
fn geometric_path() -> (Vec<u32>, Vec<u32>, Vec<f32>, Vec<f32>) {
    let (mut left, mut right, mut x, mut y) = (vec![0], vec![0], vec![1.0], vec![-1.0]);
    for k in 0..30u32 {
        let (l, r, xl, yr) = match k % 2 {
            0 => (k / 2 + 1, k / 2 + 1, 0.25, 4.0),
            _ => (k.div_ceil(2) + 1, (k - 1) / 2 + 1, 4.0, 0.25),
        };
        left.push(l);
        right.push(r);
        x.push(xl);
        y.push(yr);
    }
    (left, right, x, y)
}

/// Left vertices `0..17`, right `17..33`. The first stored row holding a scaled entry below the floor is the stored side's 15th: left 15, vertex 15, when the left side is stored, as whole and kept right; right 15, vertex 32, when the right is, its kept left side rotated after it. Each is its block's local row 14.
#[test]
fn a_bipartite_block_names_the_stored_row_in_the_callers_numbering() {
    let (left, right, x, y) = geometric_path();
    let left = FactorWidth1::valued(17, &left, &x).expect("a valid side");
    let right = FactorWidth1::valued(16, &right, &y).expect("a valid side");
    let v = BipartiteFactorWidth2::new(left, right).expect("a valid gram");
    let complement = |kept| {
        factorize_complement(v, kept, Config::default(), ScalingPolicy::default()).map(|_| ())
    };
    let whole = factorize(v).map(|_: ApproxInverse<f32>| ());
    assert_eq!(whole, Err(BuildError::OutOfRange { vertex: 15 }));
    assert_eq!(
        complement(Side::Right),
        Err(BuildError::OutOfRange { vertex: 15 })
    );
    assert_eq!(
        complement(Side::Left),
        Err(BuildError::OutOfRange { vertex: 32 })
    );
}
