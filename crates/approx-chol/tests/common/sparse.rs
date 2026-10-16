use approx_chol::{CliqueRule, Config, Multiplicity, SplitCount};

/// `dense_up_to = 0` under AC or AC2, each complement exact or sampled either way.
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
        let (single, split) = (
            Multiplicity::Single,
            Multiplicity::Split(SplitCount::new(2).expect("two copies split")),
        );
        let (complement, approximate) = match self {
            Sparse::AcExactComplement => (CliqueRule::Exact, single),
            Sparse::AcComplementAc => (CliqueRule::Sampled(single), single),
            Sparse::AcComplementAc2 => (CliqueRule::Sampled(split), single),
            Sparse::Ac2ExactComplement => (CliqueRule::Exact, split),
            Sparse::Ac2ComplementAc => (CliqueRule::Sampled(single), split),
            Sparse::Ac2ComplementAc2 => (CliqueRule::Sampled(split), split),
        };
        Config {
            seed: 7,
            complement,
            approximate,
            dense_up_to: 0,
        }
    }
}
