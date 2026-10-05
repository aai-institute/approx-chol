use super::compensated_sum;

/// Each order puts the dropped `1.0` in a different branch of the compensation.
#[test]
fn compensated_sum_recovers_the_term_a_plain_fold_drops() {
    for values in [[1e16, 1.0, -1e16], [1.0, 1e16, -1e16]] {
        assert_eq!(values.iter().sum::<f64>(), 0.0, "plain fold keeps the term");
        assert_eq!(compensated_sum(&values), 1.0);
    }
}
