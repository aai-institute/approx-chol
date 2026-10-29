use super::cycle::Cycle;
use super::dense::dense;
use super::dense_residual::apply;
use super::gram::{dense_csr, matrix};
use approx_chol::input::{
    BipartiteFactorWidth2, CsrRef, FactorWidth1, FactorWidth2, Laplacian, Sddm,
};
use approx_chol::{factorize, factorize_with, Factor, Real, ScalingPolicy};

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

/// A cycle off its null vector by `1e-10` leaves a surplus twinned with the deficit clamped beside it; grounded, `x₀` was `-5e12`.
#[test]
fn gram_cycle_whose_surplus_twins_its_clamped_deficit_floats() {
    let factor = factorize(Cycle::new(20, -1e-10).gram()).unwrap();
    let block = &factor.blocks()[0];
    assert!(block.scaling_report().is_some());
    assert!(block.floats());
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

/// Worker slopes `t − t̄` beside year intercepts, one observation off by `δ`: a row's residue passes its band and the deficit clamped beside it, yet cancels in `1ᵀM1`.
#[test]
fn near_aliased_slope_residues_cancel_so_the_block_floats() {
    for delta in [1e-9, -1e-9] {
        let (workers, years) = (400u32, 10u32);
        let mean = f64::from(years - 1) / 2.0;
        let (mut rows, mut cols, mut x) = (vec![], vec![], vec![]);
        for w in 0..workers {
            for t in 0..years {
                rows.push(w);
                cols.push(t);
                x.push(f64::from(t) - mean);
            }
        }
        x[3] += delta;
        let gram = BipartiteFactorWidth2::new(
            FactorWidth1::new(workers as usize, &rows, Some(&x)).unwrap(),
            FactorWidth1::new(years as usize, &cols, None).unwrap(),
        )
        .unwrap();
        let factor = factorize(gram).unwrap();
        assert_eq!(factor.blocks().len(), 1);
        assert!(factor.blocks()[0].floats(), "{delta}");
    }
}
