use approx_chol_next::{CliqueRule, Config, Multiplicity, SplitCount};

use crate::graph::Graph;

/// Every Cholesky dense, every Schur complement exact.
pub fn exact() -> Config {
    Config {
        complement: CliqueRule::Exact,
        dense_up_to: usize::MAX,
        ..Config::default()
    }
}

/// `exact` where `graph` is small enough to factor dense, then `dense_up_to = 0` under AC and AC2, each complement exact or sampled either way.
pub fn modes_for(graph: &Graph) -> Vec<Config> {
    let split = Multiplicity::Split(SplitCount::new(2).expect("two copies split"));
    let multiplicities = [Multiplicity::Single, split];
    let complements = [CliqueRule::Exact]
        .into_iter()
        .chain(multiplicities.map(CliqueRule::Sampled));
    let sampled = complements.flat_map(|complement| {
        multiplicities.map(|approximate| Config {
            seed: 7,
            complement,
            approximate,
            dense_up_to: 0,
        })
    });
    let dense = (graph.n <= 512).then(exact);
    dense.into_iter().chain(sampled).collect()
}
