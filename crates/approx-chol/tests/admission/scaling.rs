use approx_chol::input::Gram;
use approx_chol::{factorize, ScalingPolicy};
use rstest::rstest;

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
    let mut draws = Draws(seed.wrapping_mul(2_654_435_761).wrapping_add(17));
    let d: Vec<f64> = (0..n).map(|_| draws.log_uniform(bits)).collect();
    let weights: Vec<f64> = edges.iter().map(|_| draws.log_uniform(bits)).collect();
    congruent_laplacian(&d, edges, &weights)
}

/// A fixed LCG; powers of two come from their bits, so no libm.
struct Draws(u64);

impl Draws {
    fn unit(&mut self) -> f64 {
        self.0 = (self.0)
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn log_uniform(&mut self, bits: u32) -> f64 {
        let exponent = (self.unit() * f64::from(bits)) as u64 + 1;
        (1.0 + self.unit()) * f64::from_bits((1023 - exponent) << 52)
    }
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

fn grid(k: usize) -> Vec<(usize, usize)> {
    let edges = crate::grid::grid(k, k).edges;
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
    let block = &factor.blocks()[0];
    assert_eq!(block.scaling_report(), None, "spread {spread:e}");
    assert!(block.floats(), "spread {spread:e}");
}

/// Settled within 16ε, then `F·A·F` rounds a few ε more: this grid's 17.3ε is rounding, not a deficit.
#[test]
fn a_settled_scaling_with_rounding_deficit_is_not_clamped() {
    let k = 22;
    let (row_ptrs, columns, values) = dld(k * k, &grid(k), 1e-3, 30);
    let factor = reduce(&row_ptrs, &columns, &values).expect("admitted");
    assert_eq!(factor.blocks()[0].scaling_report(), None);
}

/// Ranked by violation, a stall restart resumed from a stale best, adopted nothing, and gave up at 3568 of the budget.
#[test]
fn a_weighted_grid_settles_after_a_stall_restart() {
    let k = 24;
    let (row_ptrs, columns, values) = weighted_dld(k * k, &grid(k), 24, 2);
    let policy = ScalingPolicy::new(1e-9, 8000).expect("a valid policy");
    let factor = reduce_with(&row_ptrs, &columns, &values, policy).expect("admitted");
    let report = factor.blocks()[0].scaling_report();
    assert!(report.is_none_or(|r| r.settled), "{report:?}");
}

/// A restart's CG could not beat best's eigen-residual within its patience; ending there gave up at 5144 of 12000.
#[test]
fn a_stall_before_any_adoption_keeps_iterating() {
    let k = 30;
    let (row_ptrs, columns, values) = weighted_dld(k * k, &grid(k), 24, 14);
    let policy = ScalingPolicy::new(1e-9, 12_000).expect("a valid policy");
    let factor = reduce_with(&row_ptrs, &columns, &values, policy).expect("admitted");
    let report = factor.blocks()[0].scaling_report();
    assert!(report.is_none_or(|r| r.settled), "{report:?}");
}

/// A settled eigen-residual is relative to the Perron vector's peak, so a row at `μ_i` may sit `band/μ_i` off.
#[test]
fn a_settled_scaling_with_an_uneven_perron_vector_is_not_clamped() {
    let k = 30;
    let (row_ptrs, columns, values) = weighted_dld(k * k, &grid(k), 10, 0);
    let factor = reduce(&row_ptrs, &columns, &values).expect("admitted");
    assert_eq!(factor.blocks()[0].scaling_report(), None);
}

/// A Gram cycle whose null vector `u = 2^min(i, L−1−i)` peaks mid-cycle; its error is relative per entry, so no row's band grows by `1/μ_i`.
#[test]
fn a_gram_null_vector_deficit_past_its_row_rounding_is_reported() {
    let (cycle, miss) = (20u32, 1e-10);
    let u = |i: u32| f64::from(1u32 << i.min(cycle - 1 - i));
    let a: Vec<u32> = (0..cycle).collect();
    let b: Vec<u32> = (0..cycle).map(|i| (i + 1) % cycle).collect();
    let mut x: Vec<f64> = b.iter().map(|&j| u(j)).collect();
    let y: Vec<f64> = a.iter().map(|&i| -u(i)).collect();
    x[cycle as usize - 1] *= 1.0 + miss;
    let gram = Gram::factor_width_2(cycle as usize, &a, &b, &x, &y).expect("valid columns");
    let factor = factorize(gram).expect("admitted");
    let report = factor.blocks()[0].scaling_report().expect("reported");
    assert!(report.violation > miss / 10.0, "{report:?}");
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
    let block = &factor.blocks()[0];
    assert_eq!(block.scaling_report(), None);
    assert!(block.floats());
}

/// A null vector spanning `2^28` puts `μ²` across `2^112`, past `f32`'s range when peaked at `1`; every entry of `V` is in range.
#[test]
fn a_wide_f32_gram_null_vector_scales_in_range() {
    let n = 1000u32;
    let mut draws = Draws(0);
    let u: Vec<f64> = (0..n).map(|_| draws.log_uniform(28)).collect();
    let a: Vec<u32> = (0..n).collect();
    let b: Vec<u32> = (0..n).map(|i| (i + 1) % n).collect();
    let x: Vec<f32> = b.iter().map(|&j| u[j as usize] as f32).collect();
    let y: Vec<f32> = a.iter().map(|&i| -u[i as usize] as f32).collect();
    let gram = Gram::factor_width_2(n as usize, &a, &b, &x, &y).expect("valid columns");
    let factor = factorize(gram).expect("admitted");
    assert!(factor.blocks()[0].floats());
}
