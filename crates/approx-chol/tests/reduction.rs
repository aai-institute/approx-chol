#[path = "reduction/band_agreement.rs"]
mod band_agreement;
#[path = "common/cycle.rs"]
mod cycle;
#[path = "common/dense.rs"]
mod dense;
#[path = "common/dense_residual.rs"]
mod dense_residual;
#[path = "common/gram.rs"]
mod gram;
#[path = "reduction/surplus_grounds.rs"]
mod surplus_grounds;

use approx_chol::input::{
    BipartiteFactorWidth2, CsrRef, FactorWidth1, FactorWidth2, Laplacian, Sddm, UpperCsrRef,
};
use approx_chol::{factorize_with, Backend, Config, Factor, ScalingPolicy};
use dense::dense;
use dense_residual::solve_residual;
use gram::{dense_csr, matrix};

fn floats(factor: &Factor) -> Vec<bool> {
    factor.blocks().iter().map(|block| block.floats()).collect()
}

#[test]
fn csr_and_upper_share_reduction_and_solve_bits() {
    let (a, b) = ([0, 1, 3, 4, 3], [1, 2, 4, 5, 5]);
    let (x, y) = ([1.0; 5], [-1.0, -1.0, 1.0, 1.0, 1.0]);
    let matrix = matrix(7, &a, &b, &x, &y);
    let mut ptrs = [vec![0u32], vec![0u32]];
    let mut indices = [Vec::new(), Vec::new()];
    let mut values = [Vec::new(), Vec::new()];
    for (i, row) in matrix.iter().enumerate() {
        for (j, &value) in row.iter().enumerate() {
            if value != 0.0 {
                for upper in 0..2 {
                    if upper == 0 || j >= i {
                        indices[upper].push(j as u32);
                        values[upper].push(value);
                    }
                }
            }
        }
        for upper in 0..2 {
            ptrs[upper].push(indices[upper].len() as u32);
        }
    }
    let csr = CsrRef::new(&ptrs[0], &indices[0], &values[0], 7).unwrap();
    let upper = UpperCsrRef::new(&ptrs[1], &indices[1], &values[1], 7).unwrap();
    let rhs = [1.0; 7];
    for config in [
        Config {
            backend: Backend::Approximate,
            ..Config::default()
        },
        dense(),
    ] {
        let full: Factor = factorize_with(csr, config, ScalingPolicy::default()).unwrap();
        let upper: Factor = factorize_with(upper, config, ScalingPolicy::default()).unwrap();
        assert_eq!(format!("{full:?}"), format!("{upper:?}"));
        assert_eq!(floats(&full), [true, false]);
        let a = full.solve(&rhs).unwrap();
        let b = upper.solve(&rhs).unwrap();
        assert_eq!(
            a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(a[6], 0.0);
    }
    solve_residual(
        &factorize_with(csr, dense(), ScalingPolicy::default()).unwrap(),
        &matrix,
    );
}

#[test]
fn gram_separates_folded_scaled_and_frustrated_components() {
    let (a, b) = ([0, 1, 3, 4, 3], [1, 2, 4, 5, 5]);
    let (x, y) = ([1.3, 2.7, 1.0, 1.0, 1.0], [-2.7, -4.1, 1.0, 1.0, 1.0]);
    let gram = FactorWidth2::new(7, (&a, &x), (&b, &y)).unwrap();
    let reduced = factorize_with(gram, dense(), ScalingPolicy::new(0.0, 0).unwrap()).unwrap();
    assert_eq!(floats(&reduced), [true, false]);
    assert!(reduced
        .blocks()
        .iter()
        .all(|c| c.scaling_report().is_none()));
    solve_residual(&reduced, &matrix(7, &a, &b, &x, &y));
}

#[test]
fn shape_ground_and_signature_match_their_gram_matrices() {
    let (a, b) = ([0, 1, 2, 0], [1, 2, 3, 3]);
    let weights = [1.0, 4.0, 9.0, 16.0];
    let (x, y) = ([1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]);
    let ground = [0.25, 0.0, 0.0, 0.0];
    let laplacian = Laplacian::weighted(4, &a, &b, &weights).unwrap();
    let floating = factorize_with(laplacian, dense(), ScalingPolicy::default()).unwrap();
    assert!(floating.blocks()[0].floats());
    let grounded = factorize_with(
        Sddm::new(laplacian, &ground).unwrap(),
        dense(),
        ScalingPolicy::default(),
    )
    .unwrap();
    assert!(!grounded.blocks()[0].floats());
    let mut expected = matrix(4, &a, &b, &x, &y);
    solve_residual(&floating, &expected);
    expected[0][0] += ground[0];
    solve_residual(&grounded, &expected);
    let y = [1.0, -2.0, 3.0, -4.0];
    let signed = matrix(4, &a, &b, &x, &y);
    let (ptrs, columns, values) = dense_csr(&signed);
    let csr = CsrRef::new(&ptrs, &columns, &values, 4).unwrap();
    let factor = factorize_with(csr, dense(), ScalingPolicy::default()).unwrap();
    assert!(factor.blocks()[0].floats());
    solve_residual(&factor, &signed);
}

#[test]
fn bipartite_incidence_and_valued_grams_preserve_both_sides() {
    let (left, right) = ([0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]);
    let (ones, minus) = ([1.0; 6], [-1.0; 6]);
    let gram = |values_l, values_r| {
        let l = FactorWidth1::new(3, &left, values_l).unwrap();
        let r = FactorWidth1::new(2, &right, values_r).unwrap();
        let gram = BipartiteFactorWidth2::new(l, r).unwrap();
        factorize_with(gram, dense(), ScalingPolicy::default()).unwrap()
    };
    let incidence = gram(None, None);
    let valued = gram(Some(&ones[..]), Some(&ones[..]));
    let laplacian = gram(None, Some(&minus[..]));
    for sgdd in [&incidence, &valued, &laplacian] {
        assert!(sgdd.blocks()[0].floats());
    }
    let right = right.map(|j| j + 3);
    let positive = matrix(5, &left, &right, &ones, &ones);
    let negative = matrix(5, &left, &right, &ones, &minus);
    solve_residual(&incidence, &positive);
    solve_residual(&valued, &positive);
    solve_residual(&laplacian, &negative);
}

#[test]
fn exhausted_scaling_reports_on_grams_and_rows() {
    let (a, b) = ([0, 1, 0], [1, 2, 2]);
    let (x, y) = ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
    let gram = FactorWidth2::new(3, (&a, &x), (&b, &y)).unwrap();
    let policy = ScalingPolicy::new(0.0, 0).unwrap();
    let factor: Factor = factorize_with(gram, dense(), policy).unwrap();
    assert!(factor.blocks()[0].scaling_report().is_some());
    let matrix = matrix(3, &a, &b, &x, &y);
    let ptrs = [0, 3, 6, 9];
    let indices = [0, 1, 2, 0, 1, 2, 0, 1, 2];
    let values: Vec<f64> = matrix.into_iter().flatten().collect();
    let csr = CsrRef::new(&ptrs, &indices, &values, 3).unwrap();
    let factor: Factor = factorize_with(csr, dense(), policy).unwrap();
    assert!(factor.blocks()[0].scaling_report().is_some());
}
