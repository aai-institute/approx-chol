use std::num::NonZeroU32;

use approx_chol::{Config, Sampling, SchurMode};

/// `exact_max_dim: 0` under AC or AC2, each complement exact or sampled either way.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sparse {
    AcExactComplement,
    AcComplementAc,
    AcComplementAc2,
    Ac2ExactComplement,
    Ac2ComplementAc,
    Ac2ComplementAc2,
}

/// `k` clique-tree copies per edge.
pub fn clique_tree(k: u32) -> Sampling {
    Sampling::CliqueTree {
        copies: NonZeroU32::new(k).expect("nonzero"),
    }
}

impl Sparse {
    pub fn config(self) -> Config {
        let copies = clique_tree;
        let (complement, sampling) = match self {
            Sparse::AcExactComplement => (SchurMode::Exact, copies(1)),
            Sparse::AcComplementAc => (SchurMode::Same, copies(1)),
            Sparse::AcComplementAc2 => (
                SchurMode::Approximate {
                    sampling: copies(2),
                },
                copies(1),
            ),
            Sparse::Ac2ExactComplement => (SchurMode::Exact, copies(2)),
            Sparse::Ac2ComplementAc => (
                SchurMode::Approximate {
                    sampling: copies(1),
                },
                copies(2),
            ),
            Sparse::Ac2ComplementAc2 => (SchurMode::Same, copies(2)),
        };
        Config {
            seed: 7,
            sampling,
            exact_max_dim: 0,
            complement,
            ..Config::default()
        }
    }
}
