use approx_chol::Config;

use crate::entry::{by_type, Reduced};

impl Reduced {
    /// Per irreducible block, whether it solves as singular.
    pub fn floats(&self) -> Vec<bool> {
        let factored = self.factorize(&Config::default());
        by_type!(&factored.factor, factor => factor.blocks().iter().map(|block| block.floats()).collect())
    }
}
