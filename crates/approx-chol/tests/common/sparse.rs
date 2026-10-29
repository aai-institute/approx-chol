use approx_chol::{Config, SchurMode};

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

impl Sparse {
    pub fn config(self) -> Config {
        let (complement, split_merge) = match self {
            Sparse::AcExactComplement => (SchurMode::Exact, None),
            Sparse::AcComplementAc => (SchurMode::Same, None),
            Sparse::AcComplementAc2 => (
                SchurMode::Approximate {
                    split_merge: Some(2),
                },
                None,
            ),
            Sparse::Ac2ExactComplement => (SchurMode::Exact, Some(2)),
            Sparse::Ac2ComplementAc => (SchurMode::Approximate { split_merge: None }, Some(2)),
            Sparse::Ac2ComplementAc2 => (SchurMode::Same, Some(2)),
        };
        Config {
            seed: 7,
            split_merge,
            exact_max_dim: 0,
            complement,
        }
    }
}
