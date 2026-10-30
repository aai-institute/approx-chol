use super::cycle::Cycle;
use super::dense::dense;
use super::dense_residual::apply;
use super::gram::{dense_csr, matrix};
use super::lcg::Lcg;
use approx_chol::input::{
    BipartiteFactorWidth2, CsrRef, FactorWidth1, FactorWidth2, Laplacian, Sddm,
};
use approx_chol::{factorize, factorize_with, Config, Factor, Real, ScalingPolicy};
use rstest::rstest;

fn grounded<T: Real>(factor: &Factor<T>) -> bool {
    factor.blocks().iter().all(|block| !block.floats())
}

/// The chain's null vector `(1, 1e200, 1e400)` passes `f64`; floating, `x₀` is `4.9e174`.
#[test]
fn csr_chain_whose_row_surplus_outweighs_its_coupling_grounds() {
    let values = [1e200, -1.0, -1.0, 1e200, -1.0, -1.0, 1e-200];
    let csr = CsrRef::new(&[0u32, 2, 5, 7], &[0, 1, 0, 1, 2, 1, 2], &values, 3).unwrap();
    let factor = factorize(csr).unwrap();
    assert_eq!(factor.solve(&[1e200, -1.0, 0.0]).unwrap(), [1.0, 0.0, 0.0]);
    assert!(grounded(&factor));
}

#[test]
fn gram_chain_whose_row_surplus_outweighs_its_coupling_grounds() {
    let (x, y) = ([1e100, 1e100], [-1e-100, -1e-100]);
    let gram = FactorWidth2::new(3, (&[0u32, 1], &x), (&[1u32, 2], &y)).unwrap();
    let factor = factorize(gram).unwrap();
    let solution = factor.solve(&[1e200, 1e200, -1.0]).unwrap();
    let error = (solution.iter().zip([1.0, 1.0, 0.0])).map(|(s, e): (&f64, f64)| (s - e).abs());
    assert!(error.fold(0.0, f64::max) <= 1e-15, "{solution:?}");
    assert!(grounded(&factor));
}

/// `edges` with unit weights, plus vertex `n` hung off vertex 0 by `1e-17` and grounded by `2e-17`.
fn weak_vertex_solves_as_its_sddm_view(n: u32, edges: &[(u32, u32)]) {
    let (coupling, surplus) = (1e-17f64, 2e-17f64);
    let size = n as usize + 1;
    let (mut u, mut v): (Vec<u32>, Vec<u32>) = edges.iter().copied().unzip();
    u.push(0);
    v.push(n);
    let mut weights = vec![1.0; edges.len()];
    weights.push(coupling);
    let mut ground = vec![0.0; size];
    ground[n as usize] = surplus;
    let laplacian = Laplacian::weighted(size, &u, &v, &weights).unwrap();
    let sddm = factorize(Sddm::new(laplacian, &ground).unwrap()).unwrap();
    let (mut a, mut b) = (u.clone(), v.clone());
    a.push(n);
    b.push(0);
    let mut x: Vec<f64> = weights.iter().map(|w| w.sqrt()).collect();
    let mut y: Vec<f64> = x.iter().map(|x| -x).collect();
    x.push(surplus.sqrt());
    y.push(0.0);
    let gram = factorize(FactorWidth2::new(size, (&a, &x), (&b, &y)).unwrap()).unwrap();
    let dense = matrix(size, &a, &b, &x, &y);
    let (ptrs, columns, values) = dense_csr(&dense);
    let csr = factorize(CsrRef::new(&ptrs, &columns, &values, size as u32).unwrap()).unwrap();
    let exact = (0..size).map(|i| [1.0, -1.0, 0.5, 2.0][i % 4]);
    let rhs: Vec<f64> = (dense.iter())
        .map(|row| row.iter().zip(exact.clone()).map(|(a, x)| a * x).sum())
        .collect();
    let expected = sddm.solve(&rhs).unwrap();
    for (label, factor) in [("gram", &gram), ("csr", &csr)] {
        let solution = factor.solve(&rhs).unwrap();
        let gap = (solution.iter().zip(&expected)).map(|(s, e)| (s - e).abs());
        let gap = gap.fold(0.0, f64::max);
        assert!(gap <= 1e-12, "{label}: {gap:e} off {expected:?}");
        assert!(grounded(factor), "{label} floats");
    }
}

/// A triangle, and a path past the default `exact_max_dim: 24`, so the grounded block is factored by AC.
#[rstest]
#[case::triangle(3, vec![(0, 1), (1, 2), (0, 2)])]
#[case::long_path(40, (0..39).map(|i| (i, i + 1)).collect())]
fn weakly_grounded_vertex_solves_as_its_sddm_view(#[case] n: u32, #[case] edges: Vec<(u32, u32)>) {
    weak_vertex_solves_as_its_sddm_view(n, &edges);
}

/// A star whose centre's `f64` row sum rounds up at every leaf, `d₀` that rounded sum; `frustrated` adds `+1` between leaves 1 and 2.
fn rounding_up_star(leaves: usize, frustrated: bool) -> Vec<Vec<f64>> {
    let mut sum = 0.0f64;
    let weights: Vec<f64> = (0..leaves)
        .map(|_| {
            let next = sum + 1.0;
            let ulp = f64::from_bits(next.to_bits() + 1) - next;
            // Just past half an ulp below `next`; below a power of two the spacing halves, so no gain there.
            let t = match (next as u64).is_power_of_two() {
                true => 1.0,
                false => 1.0 - ulp / 2.0 + f64::EPSILON,
            };
            sum += t;
            t
        })
        .collect();
    let mut a = vec![vec![0.0; leaves + 1]; leaves + 1];
    a[0][0] = sum;
    for (leaf, &t) in (1..).zip(&weights) {
        (a[0][leaf], a[leaf][0], a[leaf][leaf]) = (-t, -t, t);
    }
    if frustrated {
        (a[1][2], a[2][1]) = (1.0, 1.0);
        a[1][1] += 1.0;
        a[2][2] += 1.0;
    }
    a
}

/// The centre's exact surplus over `ε tr M`; every other row's is zero.
fn centre_surplus_over_band(a: &[Vec<f64>]) -> f64 {
    let leaves = a.len() - 1;
    let rounding: f64 = a[0][1..].iter().map(|&t| t.abs() - 1.0).sum();
    let surplus = (a[0][0] - leaves as f64) - rounding;
    let trace: f64 = (0..a.len()).map(|i| a[i][i]).sum();
    surplus / (f64::EPSILON * trace)
}

/// `‖Ax − b‖∞ / (‖A‖∞ ‖x‖∞)` for `x` the dense factor's solve of `b`, which must not float.
fn backward_error(a: &[Vec<f64>], rhs: &[f64]) -> f64 {
    let (ptrs, columns, values) = dense_csr(a);
    let csr = CsrRef::new(&ptrs, &columns, &values, a.len() as u32).unwrap();
    let factor: Factor = factorize_with(csr, dense(), ScalingPolicy::default()).unwrap();
    assert!(!factor.blocks()[0].floats());
    let x = factor.solve(rhs).unwrap();
    let residual = (apply(a, &x).into_iter().zip(rhs)).map(|(r, b)| (r - b).abs());
    let a_norm = (a.iter()).map(|row| row.iter().map(|a| a.abs()).sum::<f64>());
    let norm = a_norm.fold(0.0, f64::max) * x.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    residual.fold(0.0, f64::max) / norm
}

/// The pseudoinverse a floating block returns misses `b`'s mean, `≈ 1` here.
#[test]
fn csr_star_whose_row_sum_rounds_up_to_its_diagonal_grounds() {
    let a = rounding_up_star(100, false);
    let band = centre_surplus_over_band(&a);
    assert!(band > 5.0, "{band}");
    let rhs: Vec<f64> = (0..a.len()).map(|i| (i % 5) as f64 - 1.0).collect();
    let backward = backward_error(&a, &rhs);
    assert!(backward <= 16.0 * f64::EPSILON, "{backward:e}");
}

/// Ungrounded, a frustrated block's two-sheet cover is still definite on `[z; −z]`.
#[test]
fn frustrated_star_whose_surplus_rounds_to_zero_solves_its_matrix() {
    let a = rounding_up_star(100, true);
    let band = centre_surplus_over_band(&a);
    assert!(band > 5.0, "{band}");
    let exact: Vec<f64> = (0..a.len()).map(|i| (i % 5) as f64 - 2.0).collect();
    let backward = backward_error(&a, &apply(&a, &exact));
    assert!(backward <= 16.0 * f64::EPSILON, "{backward:e}");
}

/// Path `0 —r²— 1 —1— 2`, vertex 2 grounded by `1` = its coupling: `x = [2, 2, 1]`.
fn heavy_path_solves_grounded<T: Real>(r: T) {
    let (one, two) = (T::one(), T::one() + T::one());
    let w = r * r;
    let values = [w, -w, -w, w + one, -one, -one, two];
    let csr = CsrRef::new(&[0u32, 2, 5, 7], &[0, 1, 0, 1, 2, 1, 2], &values, 3).unwrap();
    let (x, y) = ([r, one, one], [-r, -one, T::zero()]);
    let gram = FactorWidth2::new(3, (&[0u32, 1, 2], &x), (&[1u32, 2, 0], &y)).unwrap();
    for (label, factor) in [("csr", factorize(csr)), ("gram", factorize(gram))] {
        let factor = factor.unwrap();
        let solution = factor.solve(&[T::zero(), one, T::zero()]).unwrap();
        assert_eq!(solution, [two, two, one], "{label}");
        assert!(grounded(&factor), "{label}");
    }
}

/// `1ᵀM1 = 1` is within `5ε tr M` once `r² ≥ 1/(10ε)`, yet vertex 2's surplus is past its row's band.
#[test]
fn heavy_path_whose_light_end_grounds_by_its_coupling_grounds() {
    heavy_path_solves_grounded(2f32.powi(10));
    heavy_path_solves_grounded(2f64.powi(25));
}

/// The path above with column `2r e₀ − r e₁`, dominant only once scaled by its null vector `(1, 2, 2)`: `x = [1, 2, 1]`.
#[test]
fn scaled_heavy_gram_path_grounds() {
    fn solves<T: Real>(r: T) {
        let (one, two) = (T::one(), T::one() + T::one());
        let (x, y) = ([two * r, one, one], [-r, -one, T::zero()]);
        let gram = FactorWidth2::new(3, (&[0u32, 1, 2], &x), (&[1u32, 2, 0], &y)).unwrap();
        let factor = factorize(gram).unwrap();
        let solution = factor.solve(&[T::zero(), one, T::zero()]).unwrap();
        let error = (solution.iter().zip([one, two, one])).map(|(&s, e)| ((s - e) / e).abs());
        let error = error.fold(T::zero(), T::max);
        assert!(error <= T::from(1e-3).unwrap(), "{solution:?}");
        assert!(grounded(&factor));
    }
    solves(2f32.powi(10));
    solves(2f64.powi(25));
}

/// A cycle missing by `1e-10`, within `√ε`, closes: it floats, its miss clamped and reported; grounded, `x₀` was `-5e12`.
#[rstest]
fn gram_cycle_missing_within_its_tolerance_floats_and_reports(#[values(1e-10, -1e-10)] miss: f64) {
    let factor = factorize(Cycle::<f64>::doubling(20).missing(miss).gram()).unwrap();
    let block = &factor.blocks()[0];
    let report = block.scaling_report().expect("reported");
    assert!(report.violation > 1e-11, "{report:?}");
    assert!(block.floats());
}

/// `Σ|a₀ⱼ| = 1 + 0.75ε` rounds up to `1 + ε` in `f32`; `d₀ = 1 + 10ε` is `4.6ε tr M`, `1 + 11ε` is `5.1ε`.
#[rstest]
#[case::in_band(10.0, true)]
#[case::past_band(11.0, false)]
fn f32_star_just_past_the_band_grounds(#[case] ulps: f32, #[case] floats: bool) {
    let (eps, w) = (f32::EPSILON, 0.75 * f32::EPSILON);
    let values = [1.0 + ulps * eps, -1.0, -w, -1.0, 1.0, -w, w];
    let csr = CsrRef::new(&[0u32, 3, 5, 7], &[0, 1, 2, 0, 1, 0, 2], &values, 3).unwrap();
    let factor: Factor<f32> = factorize(csr).unwrap();
    assert_eq!(factor.blocks()[0].floats(), floats);
}

/// Worker slopes `t − t̄` beside year intercepts, the first or last worker's year 3 off by `δ`: the last closes its cycles after the year hubs' many lookups.
#[rstest]
fn near_aliased_slope_floats(
    #[values((400, 1e-9), (400, -1e-9), (4000, 1e-12), (4000, -1e-12))] case: (u32, f64),
    #[values(false, true)] last: bool,
) {
    let ((workers, delta), years) = (case, 10u32);
    let mean = f64::from(years - 1) / 2.0;
    let rows: Vec<u32> = (0..workers * years).map(|k| k / years).collect();
    let cols: Vec<u32> = (0..workers * years).map(|k| k % years).collect();
    let mut x: Vec<f64> = cols.iter().map(|&t| f64::from(t) - mean).collect();
    let at = if last { x.len() - years as usize } else { 0 };
    x[at + 3] += delta;
    let gram = BipartiteFactorWidth2::new(
        FactorWidth1::new(workers as usize, &rows, Some(&x)).unwrap(),
        FactorWidth1::new(years as usize, &cols, None).unwrap(),
    )
    .unwrap();
    let factor = factorize(gram).unwrap();
    assert_eq!(factor.blocks().len(), 1);
    assert!(factor.blocks()[0].floats());
}

/// A cycle of `2⁵⁰` whose rows sit `residues` off their coupling, and a pendant grounded by `1`: rounding outweighs the ground in `Σs⁺` or `1ᵀM1` (#323).
#[rstest]
fn rounding_residues_beside_a_light_ground_do_not_float_it(
    #[values([0.5, -0.5], [-0.5, -0.5])] residues: [f64; 2],
) {
    let (n, w) = (8, 2f64.powi(50));
    let mut a = vec![vec![0.0; n + 1]; n + 1];
    for i in 0..n {
        let j = (i + 1) % n;
        (a[i][j], a[j][i]) = (-w, -w);
        a[i][i] = 2.0 * w + residues[i % 2];
    }
    (a[0][n], a[n][0]) = (-1.0, -1.0);
    a[0][0] += 1.0;
    a[n][n] = 2.0;
    let rhs: Vec<f64> = (0..a.len()).map(|i| (i % 5) as f64 - 1.0).collect();
    let backward = backward_error(&a, &rhs);
    assert!(backward <= 16.0 * f64::EPSILON, "{backward:e}");
}

/// A stated ground of `10⁻¹¹ a_00` grounds a ring of `10⁵` rounded links, from either end of its column.
#[rstest]
fn light_ground_on_long_ring_grounds(#[values(false, true)] swapped: bool) {
    let ring = Cycle::random(100_000, 1).grounded(0, 1e-11);
    let ring = if swapped { ring.swapped() } else { ring };
    let factor: Factor = factorize(ring.gram()).unwrap();
    assert!(!factor.blocks()[0].floats());
}

/// A cycle missing past `√ε` states the ring definite: it grounds whatever the scaling iteration leaves.
#[rstest]
fn ring_missing_past_its_tolerance_grounds(
    #[values(1, 2, 3)] seed: u64,
    #[values(2e-8, -1e-7, 1e-6)] miss: f64,
) {
    let factor: Factor = factorize(Cycle::random(1000, seed).missing(miss).gram()).unwrap();
    assert!(!factor.blocks()[0].floats());
}

/// Values rounded to `f32` close the ring within `√ε`: its null vector scales it, no iteration runs.
#[test]
fn f32_rounded_ring_keeps_its_null_vector() {
    let ring = Cycle::random(10_000, 1);
    let factor: Factor<f32> =
        factorize_with(ring.gram(), Config::default(), no_iterations()).unwrap();
    let block = &factor.blocks()[0];
    assert_eq!(block.scaling_report(), None);
    assert!(block.floats());
}

/// An `f32` ring missing by `10⁻²` is past `√ε`: it falls to `relax`, which here may not iterate, so it is reported.
#[test]
fn f32_ring_missing_past_its_tolerance_is_reported() {
    let ring = Cycle::random(10_000, 1).missing(1e-2);
    let factor: Factor<f32> =
        factorize_with(ring.gram(), Config::default(), no_iterations()).unwrap();
    let report = factor.blocks()[0].scaling_report();
    assert!(report.is_some_and(|report| !report.settled), "{report:?}");
}

fn no_iterations() -> ScalingPolicy {
    ScalingPolicy::new(1e-9, 0).unwrap()
}

/// `petals` rings of four columns through hub 0, the last one's closing column off by `1 + miss`: a short cycle in a block of `3·petals + 1`.
fn flower(petals: u32, miss: f64) -> (usize, [Vec<u32>; 2], [Vec<f64>; 2]) {
    let mut lcg = Lcg(1);
    let mut draw = move || lcg.unit() + 0.5;
    let u: Vec<f64> = (0..=3 * petals).map(|_| draw()).collect();
    let (mut a, mut b, mut x, mut y) = (vec![], vec![], vec![], vec![]);
    for p in 0..petals {
        let ring = [0, 3 * p + 1, 3 * p + 2, 3 * p + 3, 0];
        for (&i, &j) in ring.iter().zip(&ring[1..]) {
            let r = draw();
            a.push(i);
            b.push(j);
            x.push(u[j as usize] * r);
            y.push(-u[i as usize] * r);
        }
    }
    *x.last_mut().unwrap() *= 1.0 + miss;
    (u.len(), [a, b], [x, y])
}

/// A short cycle missing by `10⁻¹³` among `20000` clamps its petal's row past band, so it is reported; the hub's band covers its degree.
#[rstest]
#[case::closed(0.0, false)]
#[case::missing(-1e-13, true)]
fn short_cycle_miss_in_a_large_block_is_reported(#[case] miss: f64, #[case] reported: bool) {
    let (n, [a, b], [x, y]) = flower(20_000, miss);
    let factor: Factor = factorize(FactorWidth2::new(n, (&a, &x), (&b, &y)).unwrap()).unwrap();
    let block = &factor.blocks()[0];
    assert!(block.floats());
    assert_eq!(block.scaling_report().is_some(), reported);
}
