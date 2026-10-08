use super::*;
use crate::{DenseFailure, UnusablePivot};

/// A block that will not fit falls back whatever the policy says.
#[test]
fn only_an_unusable_pivot_answers_to_the_failure_policy() {
    let unusable = UnusablePivot {
        vertex: 30,
        failure: DenseFailure::NonPositivePivot,
    };
    let pivot = Fallback::InvalidPivot(unusable);
    let too_large = Fallback::WillNotFit { dim: 9 };
    let cases = [
        (
            "pivot, falling back",
            pivot,
            ExactFailure::FallBackToApproximate,
            Ok(pivot),
        ),
        ("pivot, erroring", pivot, ExactFailure::Error, Err(unusable)),
        (
            "will not fit, falling back",
            too_large,
            ExactFailure::FallBackToApproximate,
            Ok(too_large),
        ),
        (
            "will not fit, erroring",
            too_large,
            ExactFailure::Error,
            Ok(too_large),
        ),
    ];
    for (label, fallback, on_failure, expected) in cases {
        assert_eq!(on_failure.accept(fallback), expected, "{label}");
    }
}
