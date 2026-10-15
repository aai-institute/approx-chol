use proptest::strategy::Strategy;
use proptest::test_runner::{Config, TestCaseResult, TestRunner};

/// `proptest!` for a body inside an `rstest` case, so each entry point is its own test.
pub fn check<S: Strategy>(strategy: S, test: impl Fn(S::Value) -> TestCaseResult) {
    // The runner is shared by every suite, so a persisted seed would replay into the wrong one.
    let config = Config {
        failure_persistence: None,
        ..Config::default()
    };
    if let Err(error) = TestRunner::new(config).run(&strategy, test) {
        panic!("{error}");
    }
}
