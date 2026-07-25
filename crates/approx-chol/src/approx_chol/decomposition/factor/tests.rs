use super::*;

#[test]
fn dense_factor_solves_and_projects_to_zero_mean() {
    let factor = BlockFactor::dense(3, vec![4.0, 1.0, 1.0, 3.0], &[0, 1]).unwrap();
    // Zero-sum, so the range projection is a no-op and the solve reduces to the
    // anchor-deleted 2x2 system.
    let mut rhs = [6.0_f64, 7.0, -13.0];
    factor.solve_recovered(&mut rhs, 2, None);
    assert!(rhs[0].abs() < 1e-12);
    assert!((rhs[1] - 1.0).abs() < 1e-12);
    assert!((rhs[2] + 1.0).abs() < 1e-12);
}

#[test]
fn permutation_gather_matches_its_definition_and_scatter_inverts_it() {
    // A 2-cycle would make gather and scatter identical, hiding a reversed rotation.
    let forward = [2u32, 0, 1];
    let permutation = Permutation::from_forward(&forward).expect("not the identity");

    let original = [10.0_f64, 20.0, 30.0];
    let mut values = original;
    permutation.gather(&mut values);
    for (position, &source) in forward.iter().enumerate() {
        assert_eq!(values[position], original[source as usize]);
    }

    permutation.scatter(&mut values);
    assert_eq!(values, original);
}

#[test]
fn permutation_of_identity_is_none() {
    assert!(Permutation::from_forward(&[0, 1, 2, 3]).is_none());
    assert!(Permutation::from_forward(&[]).is_none());
}

#[test]
fn dense_factor_reports_non_positive_pivot() {
    let error = BlockFactor::dense(3, vec![1.0, 1.0, 1.0, 1.0], &[4, 7]).unwrap_err();
    assert_eq!(
        error,
        Error::DenseFactorizationFailed {
            vertex: 7,
            failure: DenseFailure::NonPositivePivot,
        }
    );
}
