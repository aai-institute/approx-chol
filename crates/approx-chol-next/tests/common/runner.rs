use proptest::strategy::Strategy;
use proptest::test_runner::{Config, TestCaseResult, TestRunner};

use crate::entry::Entry;
use crate::graph::Graph;

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

/// The graphs `entry` states, drawn as `graphs` draws them.
pub fn stated_by(
    entry: Entry,
    graphs: impl Strategy<Value = Graph>,
) -> impl Strategy<Value = Graph> {
    graphs.prop_filter("the entry point cannot state this graph", move |graph| {
        entry.states(graph)
    })
}
