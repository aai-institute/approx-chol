use super::gram::{dense_csr, matrix};
use approx_chol::input::{CsrRef, FactorWidth2, Laplacian, Sddm};
use approx_chol::{factorize, Factor};

fn grounded(factor: &Factor) -> bool {
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

/// Past the default `ExactBelow { max_dim: 24 }`, so the grounded block is factored by AC.
#[test]
fn weakly_grounded_vertex_on_a_long_path_solves_as_its_sddm_view() {
    let path: Vec<(u32, u32)> = (0..39).map(|i| (i, i + 1)).collect();
    weak_vertex_solves_as_its_sddm_view(40, &path);
}
