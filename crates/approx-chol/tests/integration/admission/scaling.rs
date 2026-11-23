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

/// Settled within 16ε, then `F·A·F` rounds a few ε more: this grid's 17.3ε is rounding, not a deficit.
#[test]
fn a_settled_scaling_with_rounding_deficit_is_not_clamped() {
    let k = 22;
    let (row_ptrs, columns, values) = dld(k * k, &grid(k), 1e-3, 30);
    let factor = reduce(&row_ptrs, &columns, &values).expect("admitted");
    assert_eq!(factor.clamps(), []);
    assert!(!only(&factor).grounding_unproven());
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
            if let Some(pendant) = x.get_mut(n) {
                *pendant *= 2f64.powi(400);
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
