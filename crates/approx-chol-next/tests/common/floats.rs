use crate::entry::{by_type, Reduced};

impl Reduced {
    /// Per irreducible block, whether it solves as singular.
    pub fn floats(&self) -> Vec<bool> {
        by_type!(&self.sgdd, sgdd => sgdd.blocks().iter().map(|block| block.floats()).collect())
    }
}
