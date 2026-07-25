use super::*;

#[test]
fn dense_factor_solves_and_anchors_last_coordinate() {
    let factor = SingleFactor::dense(3, vec![4.0, 1.0, 1.0, 3.0], &[0, 1]).unwrap();
    let mut rhs = [6.0_f64, 7.0, 99.0];
    factor.solve_recovered(&mut rhs, 3);
    assert!((rhs[0] - 1.0).abs() < 1e-12);
    assert!((rhs[1] - 2.0).abs() < 1e-12);
    assert_eq!(rhs[2], 0.0);
}

#[test]
fn dense_factor_reports_non_positive_pivot() {
    let error = SingleFactor::dense(3, vec![1.0, 1.0, 1.0, 1.0], &[4, 7]).unwrap_err();
    assert_eq!(
        error,
        Error::DenseFactorizationFailed {
            vertex: 7,
            failure: DenseFailure::NonPositivePivot,
        }
    );
}
