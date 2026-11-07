use approx_chol::Config;

use crate::common::dense::dense;
use crate::common::sparse::Sparse;

/// A graph small enough to factor dense runs `Dense` beside every sparse mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Dense,
    Sparse(Sparse),
}

impl Mode {
    pub fn config(self) -> Config {
        match self {
            Mode::Dense => dense(),
            Mode::Sparse(sparse) => sparse.config(),
        }
    }
}
