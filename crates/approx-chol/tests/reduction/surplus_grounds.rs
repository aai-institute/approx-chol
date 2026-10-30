use super::cycle::Cycle;
use super::dense::dense;
use super::dense_residual::apply;
use super::gram::{dense_csr, matrix};
use super::lcg::Lcg;
use approx_chol::input::{
    BipartiteFactorWidth2, CsrRef, FactorWidth1, FactorWidth2, Laplacian, Sddm,
};
use approx_chol::{factorize, factorize_with, Config, Factor, Real, ScalingPolicy};

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

#[test]
fn weakly_grounded_vertex_solves_as_its_sddm_view() {
    weak_vertex_solves_as_its_sddm_view(3, &[(0, 1), (1, 2), (0, 2)]);
}

/// Past the default `exact_max_dim: 24`, so the grounded block is factored by AC.
#[test]
fn weakly_grounded_vertex_on_a_long_path_solves_as_its_sddm_view() {
    let path: Vec<(u32, u32)> = (0..39).map(|i| (i, i + 1)).collect();
    weak_vertex_solves_as_its_sddm_view(40, &path);
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
#[test]
fn gram_cycle_missing_within_its_tolerance_floats_and_reports() {
    for miss in [1e-10, -1e-10] {
        let factor = factorize(Cycle::<f64>::doubling(20).missing(miss).gram()).unwrap();
        let block = &factor.blocks()[0];
        let report = block.scaling_report().expect("reported");
        assert!(report.violation > 1e-11, "{miss}: {report:?}");
        assert!(block.floats(), "{miss}");
    }
}

/// `Σ|a₀ⱼ| = 1 + 0.75ε` rounds up to `1 + ε` in `f32`; `d₀ = 1 + 10ε` is `4.6ε tr M`, `1 + 11ε` is `5.1ε`.
#[test]
fn f32_star_just_past_the_band_grounds() {
    let (eps, w) = (f32::EPSILON, 0.75 * f32::EPSILON);
    for (ulps, floats) in [(10.0, true), (11.0, false)] {
        let values = [1.0 + ulps * eps, -1.0, -w, -1.0, 1.0, -w, w];
        let csr = CsrRef::new(&[0u32, 3, 5, 7], &[0, 1, 2, 0, 1, 0, 2], &values, 3).unwrap();
        let factor: Factor<f32> = factorize(csr).unwrap();
        assert_eq!(factor.blocks()[0].floats(), floats, "d₀ = 1 + {ulps}ε");
    }
}

/// Worker slopes `t − t̄` beside year intercepts, the first or last worker's year 3 off by `δ`: the last closes its cycles after the year hubs' many lookups.
#[test]
fn near_aliased_slope_floats() {
    let cases = [(400u32, 1e-9), (400, -1e-9), (4000, 1e-12), (4000, -1e-12)];
    for ((workers, delta), last) in cases.into_iter().flat_map(|c| [(c, false), (c, true)]) {
        let years = 10u32;
        let mean = f64::from(years - 1) / 2.0;
        let (mut rows, mut cols, mut x) = (vec![], vec![], vec![]);
        for w in 0..workers {
            for t in 0..years {
                rows.push(w);
                cols.push(t);
                x.push(f64::from(t) - mean);
            }
        }
        let at = if last {
            x.len() - years as usize + 3
        } else {
            3
        };
        x[at] += delta;
        let gram = BipartiteFactorWidth2::new(
            FactorWidth1::new(workers as usize, &rows, Some(&x)).unwrap(),
            FactorWidth1::new(years as usize, &cols, None).unwrap(),
        )
        .unwrap();
        let factor = factorize(gram).unwrap();
        assert_eq!(factor.blocks().len(), 1);
        assert!(factor.blocks()[0].floats(), "{workers} {delta} {last}");
    }
}

/// A cycle of `2⁵⁰` whose rows sit `residues` off their coupling, and a pendant grounded by `1`: rounding outweighs the ground in `Σs⁺` or `1ᵀM1` (#323).
#[test]
fn rounding_residues_beside_a_light_ground_do_not_float_it() {
    for residues in [[0.5, -0.5], [-0.5, -0.5]] {
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
        assert!(
            backward <= 16.0 * f64::EPSILON,
            "{residues:?}: {backward:e}"
        );
    }
}

/// `u` errs by the roundings along the ring, past a fixed `(k + 19)ε` band from ~10⁵ vertices: it floats, a clamp reported.
#[test]
fn rounded_long_ring_floats() {
    for seed in 1..=4 {
        let factor: Factor = factorize(Cycle::random(100_000, seed).gram()).unwrap();
        let block = &factor.blocks()[0];
        assert!(block.floats(), "{seed}");
        let violation = block.scaling_report().map_or(0.0, |r| r.violation);
        assert!(violation < 1e3 * f64::EPSILON, "{seed}: {violation:e}");
    }
}

/// Values rounded to `f32` close the ring within `f32`'s rounding: its null vector scales it, no iteration runs.
#[test]
fn f32_rounded_ring_keeps_its_null_vector() {
    let no_iterations = ScalingPolicy::new(1e-9, 0).unwrap();
    let factor: Factor<f32> = factorize_with(
        Cycle::random(10_000, 1).gram(),
        Config::default(),
        no_iterations,
    )
    .unwrap();
    let block = &factor.blocks()[0];
    assert_eq!(block.scaling_report(), None);
    assert!(block.floats());
}

/// A stated ground of `10⁻¹¹ a_00` grounds the ring at the row its cycle closes on, from either end of its column.
#[test]
fn light_ground_on_long_ring_grounds() {
    let ring = || Cycle::random(100_000, 1).grounded(0, 1e-11);
    for ring in [ring(), ring().swapped()] {
        let factor: Factor = factorize(ring.gram()).unwrap();
        assert!(!factor.blocks()[0].floats());
    }
}

/// A stated ground of `10⁻¹³ a_00` grounds a path of `10⁵` rounded links, from either end of its column.
#[test]
fn light_ground_on_long_path_grounds() {
    let path = || Cycle::random(100_000, 1).open().grounded(0, 1e-13);
    for path in [path(), path().swapped()] {
        let factor: Factor = factorize(path.gram()).unwrap();
        assert!(!factor.blocks()[0].floats());
    }
}

/// An `f32` ring missing by `10⁻²` is past `√ε`: it falls to `relax`, which here may not iterate, so it is reported.
#[test]
fn f32_ring_missing_past_its_tolerance_is_reported() {
    let no_iterations = ScalingPolicy::new(1e-9, 0).unwrap();
    let ring = Cycle::random(10_000, 1).missing(1e-2);
    let factor: Factor<f32> =
        factorize_with(ring.gram(), Config::default(), no_iterations).unwrap();
    let report = factor.blocks()[0].scaling_report();
    assert!(report.is_some_and(|report| !report.settled), "{report:?}");
}

/// A ring missing by `10⁻¹⁰` floats, reported; a stated ground of `10⁻¹¹ a_00` on its closing row grounds it.
#[test]
fn near_null_ring_floats_unless_a_light_ground_is_stated() {
    let ring = || Cycle::random(100_000, 1).missing(1e-10);
    let factor: Factor = factorize(ring().gram()).unwrap();
    let block = &factor.blocks()[0];
    assert!(block.scaling_report().is_some());
    assert!(block.floats());
    let factor: Factor = factorize(ring().grounded(0, 1e-11).gram()).unwrap();
    assert!(!factor.blocks()[0].floats());
}

/// Worker slopes `(t − t̄ + offset)·r` beside year intercepts `r`, the last worker's year 3 off by `δ`, and that worker grounded by `ground·a_ii`.
fn slope(workers: u32, years: u32, offset: f64, delta: f64, ground: f64) -> [Vec<f64>; 2] {
    let mean = f64::from(years - 1) / 2.0;
    let r = |w: u32, t: u32| (f64::from(w * years + t) * 0.618_034).fract() + 0.5;
    let (mut x, mut y) = (vec![], vec![]);
    for w in 0..workers {
        for t in 0..years {
            x.push((f64::from(t) - mean + offset) * r(w, t));
            y.push(r(w, t));
        }
    }
    let last = x.len() - years as usize;
    x[last + 3] *= 1.0 + delta;
    let a_ii: f64 = x[last..].iter().map(|x| x * x).sum();
    x.push((ground * a_ii).sqrt());
    y.push(0.0);
    [x, y]
}

/// A near-aliased slope floats; a stated ground on one worker grounds it, also beside an odd count's middle year, split off by its zero slopes.
#[test]
fn light_stated_ground_beside_a_near_aliased_slope_grounds() {
    let cases = [
        (400u32, 10u32, 0.3, 1e-13, 1e-13),
        (40_000, 10, 0.3, 1e-10, 1e-11),
        (4000, 11, 0.0, 1e-12, 1e-11),
    ];
    for (workers, years, offset, delta, ground) in cases {
        let a: Vec<u32> = (0..workers * years).map(|k| k / years).collect();
        let b: Vec<u32> = (0..workers * years).map(|k| workers + k % years).collect();
        let (a, b) = ([a, vec![workers - 1]].concat(), [b, vec![0]].concat());
        let n = (workers + years) as usize;
        for (ground, floats) in [(0.0, true), (ground, false)] {
            let [x, y] = slope(workers, years, offset, delta, ground);
            let factor = factorize(FactorWidth2::new(n, (&a, &x), (&b, &y)).unwrap()).unwrap();
            let floated = factor.blocks().iter().any(|block| block.floats());
            assert_eq!(floated, floats, "{workers} {years} {delta} {ground}");
        }
    }
}

/// Grounds of `10ε a_ii`, each under its row's floor, sum past the block's `5ε tr M` band: the ring grounds though its null vector closes.
#[test]
fn stated_grounds_summing_past_the_band_ground_a_closed_ring() {
    let ring = (0..1000).fold(Cycle::random(1000, 1), |ring, i| {
        ring.grounded(i, 10.0 * f64::EPSILON)
    });
    let factor: Factor = factorize(ring.gram()).unwrap();
    assert!(!factor.blocks()[0].floats());
}

/// `petals` cycles of four columns through hub 0.
struct Flower {
    n: usize,
    a: Vec<u32>,
    b: Vec<u32>,
    x: Vec<f64>,
    y: Vec<f64>,
}

impl Flower {
    /// Each column `x = u_b·r`, `y = −u_a·r` from `value`'s draws `u` then `r`.
    fn new(petals: u32, mut value: impl FnMut() -> f64) -> Self {
        let u: Vec<f64> = (0..=3 * petals).map(|_| value()).collect();
        let (mut a, mut b, mut x, mut y) = (vec![], vec![], vec![], vec![]);
        for p in 0..petals {
            let ring = [0, 3 * p + 1, 3 * p + 2, 3 * p + 3, 0];
            for (&i, &j) in ring.iter().zip(&ring[1..]) {
                let r = value();
                a.push(i);
                b.push(j);
                x.push(u[j as usize] * r);
                y.push(-u[i as usize] * r);
            }
        }
        let n = 3 * petals as usize + 1;
        Self { n, a, b, x, y }
    }

    /// Values in `[0.5, 1.5)`.
    fn random(petals: u32) -> Self {
        let mut lcg = Lcg(1);
        Self::new(petals, move || lcg.unit() + 0.5)
    }

    /// Every value `±1`: rows already dominant.
    fn unit(petals: u32) -> Self {
        Self::new(petals, || 1.0)
    }

    /// The last petal's closing column scaled by `1 + miss`.
    fn missing(mut self, miss: f64) -> Self {
        *self.x.last_mut().unwrap() *= 1.0 + miss;
        self
    }

    fn gram(&self) -> FactorWidth2<'_, f64, u32> {
        FactorWidth2::new(self.n, (&self.a, &self.x), (&self.b, &self.y)).unwrap()
    }
}

/// A short cycle missing by `10⁻¹³` in a block of `60001` vertices clamps its petal's row past band, so it is reported; the hub's band covers its degree.
#[test]
fn short_cycle_miss_in_a_large_block_is_reported() {
    for (miss, reported) in [(0.0, false), (-1e-13, true)] {
        let factor: Factor = factorize(Flower::random(20_000).missing(miss).gram()).unwrap();
        let block = &factor.blocks()[0];
        assert!(block.floats(), "{miss}");
        assert_eq!(block.scaling_report().is_some(), reported, "{miss}");
    }
}

/// A cycle missing past `√ε` states the ring definite: it grounds whatever the scaling iteration leaves.
#[test]
fn ring_missing_past_its_tolerance_grounds() {
    for seed in 1..=6 {
        for miss in [2e-8, -1e-7, 1e-6] {
            let ring = Cycle::<f64>::random(1000, seed).missing(miss);
            let factor: Factor = factorize(ring.gram()).unwrap();
            assert!(!factor.blocks()[0].floats(), "{seed} {miss}");
        }
    }
}

/// A dominant flower whose closing column misses within `√ε` floats like a scaled one: its cycles are read once its band fails.
#[test]
fn dominant_flower_missing_within_its_tolerance_floats() {
    for miss in [1e-12, -1e-10, 1e-10, 1e-9] {
        let factor: Factor = factorize(Flower::unit(2000).missing(miss).gram()).unwrap();
        assert!(factor.blocks()[0].floats(), "{miss}");
    }
}

/// Near-unit values keep rows dominant yet past their band; closing every cycle, the block floats.
#[test]
fn dominant_singular_block_off_its_band_floats() {
    for seed in 1..=10 {
        let mut lcg = Lcg(seed);
        let n = 100u32;
        let u: Vec<f64> = (0..n).map(|_| 1.0 + 1e-13 * (lcg.unit() - 0.5)).collect();
        let (a, b): (Vec<u32>, Vec<u32>) = (0..n - 1).map(|i| (i, i + 1)).unzip();
        let r: Vec<f64> = a.iter().map(|_| lcg.unit() + 0.5).collect();
        let x: Vec<f64> = (b.iter().zip(&r))
            .map(|(&j, r)| u[j as usize] * r)
            .collect();
        let y: Vec<f64> = (a.iter().zip(&r))
            .map(|(&i, r)| -u[i as usize] * r)
            .collect();
        let path = FactorWidth2::new(n as usize, (&a, &x), (&b, &y)).unwrap();
        let factor: Factor = factorize(path).unwrap();
        assert!(factor.blocks()[0].floats(), "{seed}");
    }
}
